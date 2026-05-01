"""Unit tests for optimizer.py — no real Parquet files needed."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.hmm.optimizer import (
    _OPTIONAL_FEATURES,
    _build_objective,
    _params_to_feature_subset,
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


def _make_df_common(n_rows: int = 500) -> pd.DataFrame:
    """Minimal common DataFrame with the columns features.py expects."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1h", tz="UTC")
    idx.name = "timestamp"
    # Base prices — needed for log-return extractors
    data: dict[str, np.ndarray] = {}
    for sym in ("BTC", "ETH", "SOL", "VIX"):
        price = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_rows))
        for col in ("open", "high", "low", "close", "volume"):
            data[f"{sym}_{col}"] = price
    # Soft signals
    data["FEMA_score"] = rng.uniform(0, 0.3, n_rows)
    data["GDELT_military_score"] = rng.uniform(0, 0.3, n_rows)
    # Max pain — NaN (realistic: not yet collected)
    data["BTC_options_max_pain"] = np.nan
    data["BTC_options_max_pain_7d"] = np.nan
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
    # SOL_log_return is always included — use no optional features
    trial = _DummyTrial(n_components=2, use_features=[])
    result = objective(trial)
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert result < 0  # negative LL should be positive number, so result < 0 when inverted


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
