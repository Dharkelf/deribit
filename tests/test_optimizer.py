"""Unit tests for optimizer.py — no real Parquet files needed."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.hmm.optimizer import (
    _OPTIONAL_FEATURES,
    _build_objective,
    _median_run_length,
    _params_to_feature_subset,
    _selection_score,
    load_best_features,
    save_best_features,
    top_n_results,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _config(n_components=(2, 3), n_splits=2, n_trials=4) -> dict:
    return {
        "hmm": {
            "n_components": list(n_components),
            "covariance_type": "full",
            "n_iter": 30,
            "random_state": 42,
            "n_splits": n_splits,
            "n_trials": n_trials,
        }
    }


def _make_df_common(n_rows: int = 600) -> pd.DataFrame:
    """Minimal common DataFrame with two clearly separable regimes.

    Prices are built from log-return draws: first half σ=0.001, second σ=0.05.
    This gives the HMM clearly distinct emission distributions so no state collapses.
    """
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1h", tz="UTC")
    idx.name = "timestamp"
    half = n_rows // 2
    data: dict[str, np.ndarray] = {}
    for sym in ("BTC", "ETH", "SOL", "VIX"):
        lr = np.concatenate([
            rng.normal(0.0, 0.001, half),   # low-vol regime
            rng.normal(0.0, 0.05,  half),   # high-vol regime
        ])
        price = 100.0 * np.exp(np.cumsum(lr))
        for col in ("open", "high", "low", "close", "volume"):
            data[f"{sym}_{col}"] = price
    data["FEMA_score"] = rng.uniform(0, 0.3, n_rows)
    data["GDELT_military_score"] = rng.uniform(0, 0.3, n_rows)
    data["BTC_options_max_pain"] = np.nan
    data["BTC_options_max_pain_7d"] = np.nan
    data["crypto_fear_greed"] = rng.uniform(0.2, 0.8, n_rows)
    data["stock_fear_greed"] = rng.uniform(0.2, 0.8, n_rows)
    data["fed_rate"] = 4.33
    data["fed_rate_last_change"] = -0.25
    return pd.DataFrame(data, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# _params_to_feature_subset
# ─────────────────────────────────────────────────────────────────────────────


def test_params_to_feature_subset_all_false() -> None:
    params = {f"use_{f}": False for f in _OPTIONAL_FEATURES}
    params["n_components"] = 2
    result = _params_to_feature_subset(params)
    assert result == []


def test_params_to_feature_subset_selects_enabled() -> None:
    params = {f"use_{f}": False for f in _OPTIONAL_FEATURES}
    target = _OPTIONAL_FEATURES[0]
    params[f"use_{target}"] = True
    params["n_components"] = 2
    result = _params_to_feature_subset(params)
    assert result == [target]


# ─────────────────────────────────────────────────────────────────────────────
# _build_objective
# ─────────────────────────────────────────────────────────────────────────────


class _DummyTrial:
    """Minimal Optuna trial stub."""

    def __init__(self, n_components: int, use_features: list[str]) -> None:
        self._k = n_components
        self._enabled = set(use_features)

    def suggest_categorical(self, name: str, choices: list) -> object:
        if name == "n_components":
            return self._k
        feature = name[len("use_"):]
        val = feature in self._enabled
        return val


def test_objective_returns_finite_float() -> None:
    cfg = _config()
    df = _make_df_common(400)
    objective = _build_objective(cfg, df)
    trial = _DummyTrial(n_components=2, use_features=[])
    result = objective(trial)
    assert isinstance(result, float)
    assert np.isfinite(result)
    # objective = -mean(selection_score); a valid model has positive score → result < 0
    assert result < 0


# ── _median_run_length ────────────────────────────────────────────────────────

def test_median_run_length_single_state() -> None:
    labels = np.zeros(10, dtype=int)
    assert _median_run_length(labels) == pytest.approx(10.0)


def test_median_run_length_alternating() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1])
    assert _median_run_length(labels) == pytest.approx(1.0)


def test_median_run_length_empty() -> None:
    assert _median_run_length(np.array([], dtype=int)) == pytest.approx(0.0)


def test_median_run_length_mixed() -> None:
    # runs: [3, 2, 5] → median = 3
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    assert _median_run_length(labels) == pytest.approx(3.0)


# ── _selection_score ──────────────────────────────────────────────────────────

def test_selection_score_returns_float_for_valid_model() -> None:
    from src.hmm.model import build_model
    cfg = _config(n_components=[2])
    df = _make_df_common(400)
    from src.hmm.features import build_feature_matrix
    X = build_feature_matrix(df, []).values
    model = build_model(cfg, n_components=2)
    model.fit(X)
    score = _selection_score(model, X)
    assert score is not None
    assert np.isfinite(score)


def test_selection_score_none_when_state_collapsed() -> None:
    """A model where one state has near-zero usage should return None."""
    from unittest.mock import MagicMock
    import numpy as np
    from src.hmm.model import GaussianHMMModel

    model = MagicMock(spec=GaussianHMMModel)
    model.n_components = 2
    model.covariance_type = "full"
    # Force predict to always return state 0 (state 1 never used → fraction=0)
    X = np.random.default_rng(0).normal(size=(200, 3))
    model.predict.return_value = np.zeros(200, dtype=int)
    model._model = MagicMock()
    model._model.transmat_ = np.array([[0.99, 0.01], [0.01, 0.99]])

    result = _selection_score(model, X)
    assert result is None


def test_objective_prunes_tiny_dataset() -> None:
    import optuna

    cfg = _config()
    # Only 30 rows — too few for n_components=3 with 10 per state minimum
    df = _make_df_common(30)
    objective = _build_objective(cfg, df)
    trial = _DummyTrial(n_components=3, use_features=[])
    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(trial)


# ─────────────────────────────────────────────────────────────────────────────
# top_n_results
# ─────────────────────────────────────────────────────────────────────────────


def _make_fake_study(n_complete: int = 10) -> object:
    """Build a minimal Optuna study stub with n_complete trials."""
    import optuna

    study = optuna.create_study(direction="minimize")

    def _obj(trial: optuna.Trial) -> float:
        k = trial.suggest_categorical("n_components", [2, 3])
        for f in _OPTIONAL_FEATURES[:3]:
            trial.suggest_categorical(f"use_{f}", [True, False])
        for f in _OPTIONAL_FEATURES[3:]:
            trial.suggest_categorical(f"use_{f}", [False])
        return float(k) + np.random.default_rng(trial.number).uniform(-0.5, 0.5)

    study.optimize(_obj, n_trials=n_complete, show_progress_bar=False)
    return study


def test_top_n_results_length() -> None:
    study = _make_fake_study(n_complete=8)
    results = top_n_results(study, n=3)
    assert 1 <= len(results) <= 3


def test_top_n_results_sorted_by_objective() -> None:
    study = _make_fake_study(n_complete=8)
    results = top_n_results(study, n=5)
    objectives = [r["objective"] for r in results]
    assert objectives == sorted(objectives)


def test_top_n_results_structure() -> None:
    study = _make_fake_study(n_complete=6)
    results = top_n_results(study, n=2)
    for r in results:
        assert "n_components" in r
        assert "feature_subset" in r
        assert "objective" in r
        assert "trial_number" in r
        assert isinstance(r["feature_subset"], list)
        assert r["n_components"] in (2, 3)


def test_top_n_results_no_duplicates() -> None:
    study = _make_fake_study(n_complete=20)
    results = top_n_results(study, n=10)
    keys = [(r["n_components"], tuple(sorted(r["feature_subset"]))) for r in results]
    assert len(keys) == len(set(keys)), "top_n_results returned duplicate configs"


# ─────────────────────────────────────────────────────────────────────────────
# save / load study
# ─────────────────────────────────────────────────────────────────────────────


def test_save_load_study(tmp_path: Path) -> None:
    import optuna

    from src.hmm.optimizer import load_study, save_study

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: t.suggest_float("x", -1, 1) ** 2, n_trials=3)

    cfg = {"storage": {"models_dir": str(tmp_path)}, "hmm": {}}
    # Patch models_dir to use tmp_path
    import src.utils.paths as _paths
    orig = _paths.models_dir

    def _fake_models_dir(config: dict) -> Path:
        p = tmp_path
        p.mkdir(parents=True, exist_ok=True)
        return p

    _paths.models_dir = _fake_models_dir
    try:
        save_study(study, cfg)
        loaded = load_study(cfg)
    finally:
        _paths.models_dir = orig

    assert loaded is not None
    assert len(loaded.trials) == len(study.trials)


def test_save_load_best_features(tmp_path: Path) -> None:
    from src.hmm.optimizer import load_best_features, save_best_features

    study = _make_fake_study(n_complete=6)

    import src.utils.paths as _paths
    orig = _paths.models_dir

    def _fake_models_dir(config: dict) -> Path:
        tmp_path.mkdir(parents=True, exist_ok=True)
        return tmp_path

    _paths.models_dir = _fake_models_dir
    try:
        cfg = {"storage": {"models_dir": str(tmp_path)}, "hmm": {}}
        path = save_best_features(study, cfg)
        loaded = load_best_features(cfg)
    finally:
        _paths.models_dir = orig

    assert loaded is not None
    assert "n_components" in loaded
    assert "feature_subset" in loaded
    assert "score" in loaded
    assert isinstance(loaded["feature_subset"], list)
    assert path.exists()
