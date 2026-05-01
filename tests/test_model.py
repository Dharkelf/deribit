"""Unit tests for GaussianHMMModel — no real Parquet files needed."""

import numpy as np
import pytest

from src.hmm.model import GaussianHMMModel, build_model, select_n_components


def _make_X(n_samples: int = 600, n_features: int = 4, seed: int = 0) -> np.ndarray:
    """Synthetic two-regime observation matrix."""
    rng = np.random.default_rng(seed)
    # regime 0: low-volatility, regime 1: high-volatility
    X0 = rng.normal(loc=0.0,  scale=0.01, size=(n_samples // 2, n_features))
    X1 = rng.normal(loc=0.02, scale=0.04, size=(n_samples // 2, n_features))
    return np.vstack([X0, X1])


def _config(n_components: list[int] | None = None) -> dict:
    return {
        "hmm": {
            "n_components": n_components or [2, 3],
            "covariance_type": "full",
            "n_iter": 50,
            "random_state": 42,
        }
    }


# ── fit / predict ─────────────────────────────────────────────────────────────

def test_fit_returns_self() -> None:
    model = GaussianHMMModel(n_components=2, n_iter=50)
    X = _make_X()
    assert model.fit(X) is model


def test_predict_shape() -> None:
    X = _make_X()
    model = GaussianHMMModel(n_components=2, n_iter=50).fit(X)
    labels = model.predict(X)
    assert labels.shape == (len(X),)
    assert set(np.unique(labels)).issubset({0, 1})


def test_predict_proba_shape_and_sums_to_one() -> None:
    X = _make_X()
    model = GaussianHMMModel(n_components=2, n_iter=50).fit(X)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ── score / BIC ───────────────────────────────────────────────────────────────

def test_score_is_finite() -> None:
    X = _make_X()
    model = GaussianHMMModel(n_components=2, n_iter=50).fit(X)
    assert np.isfinite(model.score(X))


def test_bic_is_finite() -> None:
    X = _make_X()
    model = GaussianHMMModel(n_components=2, n_iter=50).fit(X)
    assert np.isfinite(model.bic(X))


def test_bic_penalises_extra_components() -> None:
    # On small synthetic data a 2-component model should beat 10-component
    # (severely over-parametrised) by BIC.
    X = _make_X(n_samples=300, n_features=2)
    bic2 = GaussianHMMModel(n_components=2, n_iter=50).fit(X).bic(X)
    bic10 = GaussianHMMModel(n_components=10, n_iter=50).fit(X).bic(X)
    assert bic2 < bic10, f"BIC(2)={bic2:.1f} should be < BIC(10)={bic10:.1f}"


# ── regime_stats ──────────────────────────────────────────────────────────────

def test_regime_stats_keys_and_frequencies() -> None:
    X = _make_X()
    model = GaussianHMMModel(n_components=2, n_iter=50).fit(X)
    stats = model.regime_stats(X)
    assert set(stats.keys()) == {0, 1}
    total_freq = sum(v["frequency"] for v in stats.values())
    assert abs(total_freq - 1.0) < 1e-6
    total_obs = sum(v["n_observations"] for v in stats.values())
    assert total_obs == len(X)


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    X = _make_X()
    model = GaussianHMMModel(n_components=2, n_iter=50).fit(X)
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = GaussianHMMModel.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


# ── factory ───────────────────────────────────────────────────────────────────

def test_build_model_uses_config() -> None:
    cfg = _config()
    model = build_model(cfg, n_components=3)
    assert model.n_components == 3
    assert model.covariance_type == "full"
    assert model.random_state == 42
    assert model.n_iter == 50


# ── select_n_components ───────────────────────────────────────────────────────

def test_select_n_components_returns_fitted_model() -> None:
    X = _make_X(n_samples=400, n_features=3)
    best = select_n_components(_config(n_components=[2, 3]), X)
    assert best.n_components in (2, 3)
    # must be fitted — predict should not raise
    labels = best.predict(X)
    assert labels.shape == (len(X),)


def test_select_n_components_single_candidate() -> None:
    X = _make_X()
    best = select_n_components(_config(n_components=[2]), X)
    assert best.n_components == 2
