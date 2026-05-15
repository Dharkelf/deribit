"""Smoke tests for src/hmm/visualize.py — no display required (Agg backend)."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.hmm.model import GaussianHMMModel
from src.hmm.visualize import _kstep_forecast


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_model_and_X() -> tuple[GaussianHMMModel, pd.DataFrame]:
    """Build a numerically stable 2-state HMM on structured bimodal data."""
    rng = np.random.default_rng(99)
    half = 600
    # Two clearly separated regimes → full covariance fits cleanly
    X = np.vstack([
        np.column_stack([rng.normal(0.01, 0.005, half), rng.normal(1.0, 0.3, half)]),
        np.column_stack([rng.normal(-0.01, 0.005, half), rng.normal(-1.0, 0.3, half)]),
    ])
    model = GaussianHMMModel(
        n_components=2, covariance_type="full", n_iter=50, random_state=0
    ).fit(X)
    n_rows = len(X)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1h", tz="UTC")
    X_df = pd.DataFrame(X, columns=["SOL_log_return", "feat_b"], index=idx)
    return model, X_df


# ── _kstep_forecast ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hmm_fixture() -> tuple[GaussianHMMModel, pd.DataFrame]:
    return _make_model_and_X()


def test_kstep_forecast_returns_correct_length(hmm_fixture) -> None:
    model, X_df = hmm_fixture
    k = 24
    future_ts, exp, lo, hi = _kstep_forecast(model, X_df, k, sol_last=100.0)
    assert len(future_ts) == k
    assert len(exp) == k
    assert len(lo) == k
    assert len(hi) == k


def test_kstep_forecast_prices_positive(hmm_fixture) -> None:
    model, X_df = hmm_fixture
    _, exp, lo, hi = _kstep_forecast(model, X_df, 24, sol_last=100.0)
    assert np.all(exp > 0)
    assert np.all(lo > 0)
    assert np.all(hi > 0)


def test_kstep_forecast_lo_le_exp_le_hi(hmm_fixture) -> None:
    model, X_df = hmm_fixture
    _, exp, lo, hi = _kstep_forecast(model, X_df, 24, sol_last=100.0)
    assert np.all(lo <= exp + 1e-9)
    assert np.all(exp <= hi + 1e-9)


def test_kstep_forecast_timestamps_start_after_last_known(hmm_fixture) -> None:
    model, X_df = hmm_fixture
    future_ts, _, _, _ = _kstep_forecast(model, X_df, 5, sol_last=100.0)
    assert future_ts[0] == X_df.index[-1] + pd.Timedelta(hours=1)


def test_kstep_forecast_uncertainty_grows_over_horizon(hmm_fixture) -> None:
    model, X_df = hmm_fixture
    _, _, lo, hi = _kstep_forecast(model, X_df, 48, sol_last=100.0)
    width_early = hi[0] - lo[0]
    width_late = hi[-1] - lo[-1]
    assert width_late >= width_early - 1e-9
