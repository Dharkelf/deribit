"""Unit tests for src.backtest.metrics, src.backtest.strategy, and backtest engine logic."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    annualized_return,
    directional_accuracy,
    mae,
    max_drawdown,
    rmse,
    sharpe,
)
from src.backtest.strategy import RegimeStrategy


# ── metrics ───────────────────────────────────────────────────────────────────


def test_rmse_zero_on_perfect_prediction():
    a = np.array([1.0, 2.0, 3.0])
    assert rmse(a, a) == pytest.approx(0.0)


def test_rmse_known_value():
    a = np.array([1.0, 2.0])
    p = np.array([2.0, 3.0])  # errors = [1, 1] → RMSE = 1
    assert rmse(a, p) == pytest.approx(1.0)


def test_rmse_ignores_nan():
    a = np.array([1.0, np.nan, 3.0])
    p = np.array([1.0, 9.0,   3.0])
    assert rmse(a, p) == pytest.approx(0.0)


def test_mae_known_value():
    a = np.array([0.0, 2.0, 4.0])
    p = np.array([1.0, 1.0, 3.0])  # |errors| = [1, 1, 1] → MAE = 1
    assert mae(a, p) == pytest.approx(1.0)


def test_directional_accuracy_perfect():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    pred   = np.array([1.0, 2.0, 3.0, 4.0])
    assert directional_accuracy(actual, pred) == pytest.approx(1.0)


def test_directional_accuracy_worst():
    actual = np.array([1.0, 2.0, 3.0])
    pred   = np.array([3.0, 2.0, 1.0])  # predicted opposite direction
    assert directional_accuracy(actual, pred) == pytest.approx(0.0)


def test_directional_accuracy_too_short():
    assert np.isnan(directional_accuracy(np.array([1.0]), np.array([1.0])))


def test_sharpe_positive_for_positive_returns():
    rng     = np.random.default_rng(42)
    returns = 0.001 + rng.standard_normal(1000) * 0.0001  # positive mean, non-zero std
    assert sharpe(returns) > 0


def test_sharpe_nan_for_zero_std():
    returns = np.ones(100)
    assert np.isnan(sharpe(returns))


def test_max_drawdown_no_drawdown():
    equity = np.array([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_max_drawdown_known_value():
    equity = np.array([1.0, 1.5, 1.0, 1.5])  # drawdown = (1.0 - 1.5) / 1.5
    dd = max_drawdown(equity)
    assert dd == pytest.approx((1.0 - 1.5) / 1.5, rel=1e-5)


def test_annualized_return_flat():
    returns = np.zeros(8760)  # one year flat
    assert annualized_return(returns) == pytest.approx(0.0, abs=1e-6)


# ── strategy ─────────────────────────────────────────────────────────────────


def _make_inputs(
    n: int = 20,
    regime: str = "Bullish",
) -> tuple[pd.Series, pd.Series]:
    idx    = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    lr     = pd.Series(np.full(n, 0.001), index=idx, name="sol_lr")
    labels = pd.Series([regime] * n, index=idx, name="regime")
    return lr, labels


def test_strategy_bullish_is_half_long():
    lr, labels = _make_inputs(regime="Bullish")
    result = RegimeStrategy().apply(lr, labels)
    assert (result["position"] == 0.5).all()


def test_strategy_neutral_is_cash():
    lr, labels = _make_inputs(regime="Neutral")
    result = RegimeStrategy().apply(lr, labels)
    assert (result["position"] == 0.0).all()
    assert (result["strategy_lr"] == 0.0).all()


def test_strategy_equity_starts_at_one():
    lr, labels = _make_inputs(regime="Bullish")
    result = RegimeStrategy().apply(lr, labels)
    # equity_strategy[0] = exp(position[0] * lr[0]) = exp(0.5 * 0.001)
    assert result["equity_strategy"].iloc[0] == pytest.approx(np.exp(0.5 * 0.001), rel=1e-6)


def test_strategy_bnh_equals_long():
    lr, labels = _make_inputs(regime="Strong Bullish")
    result = RegimeStrategy().apply(lr, labels)
    assert (result["strategy_lr"].values == result["bnh_lr"].values).all()


def test_strategy_per_regime_pnl():
    idx    = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    lr     = pd.Series([0.01, 0.01, -0.01, -0.01], index=idx)
    labels = pd.Series(["Bullish", "Bullish", "Bearish", "Bearish"], index=idx)
    result = RegimeStrategy().apply(lr, labels)
    pnl    = RegimeStrategy().per_regime_pnl(result)
    assert "Bullish"  in pnl.index
    assert "Bearish"  in pnl.index
    assert pnl.loc["Bullish", "hours"] == 2


def test_strategy_unknown_regime_maps_to_zero():
    lr, _ = _make_inputs()
    labels = pd.Series(["UnknownRegime"] * 20, index=lr.index)
    result = RegimeStrategy().apply(lr, labels)
    assert (result["position"] == 0.0).all()


# ── engine helpers ────────────────────────────────────────────────────────────
# Tests for the core logic patterns used in engine.run() without invoking the
# full pipeline (which requires trained models and Parquet data on disk).


def _make_regime_series(n: int = 500) -> pd.Series:
    idx    = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    labels = (["Bullish"] * 200 + ["Neutral"] * 150 + ["Bearish"] * 150)[:n]
    return pd.Series(labels, index=idx, name="regime")


def test_fold_count_matches_range():
    """Fold index arithmetic: range(min_train_h, N - horizon_h, step_h)."""
    N          = 1000
    min_train  = 168   # 7 days × 24h
    step       = 168
    horizon    = 24
    fold_idxs  = list(range(min_train, N - horizon, step))
    # All fold start indices are within bounds for a horizon window
    assert all(t + horizon <= N for t in fold_idxs)
    # First fold starts exactly at min_train
    assert fold_idxs[0] == min_train
    # Steps are exactly step_h apart
    assert all(fold_idxs[i + 1] - fold_idxs[i] == step for i in range(len(fold_idxs) - 1))


def test_fold_count_zero_when_data_too_short():
    """No folds generated when N ≤ min_train_h + horizon_h."""
    N, min_train, step, horizon = 50, 48, 24, 24
    fold_idxs = list(range(min_train, N - horizon, step))
    # 50 - 24 = 26 < 48 → empty
    assert len(fold_idxs) == 0


def test_timestamp_regime_lookup_hit():
    """Timestamp in label_series returns correct regime."""
    labels = _make_regime_series(500)
    ts     = labels.index[250]
    result = labels.get(ts, labels.iloc[-1])
    assert result == labels.iloc[250]


def test_timestamp_regime_lookup_miss_falls_back():
    """Timestamp NOT in label_series (different grid) falls back to last known."""
    labels  = _make_regime_series(500)
    missing = labels.index[0] - pd.Timedelta(hours=1)   # before series start
    fallback_idx = min(0, len(labels) - 1)
    result  = labels.get(missing, labels.iloc[fallback_idx])
    assert result == labels.iloc[0]


def test_timestamp_regime_lookup_x_df_ahead_of_x_hmm():
    """Simulate the row-count mismatch: X_df starts earlier than X_hmm.

    When _filter_24h_features removes 168h lag features, X_df gains ~167 extra
    rows at the start compared to X_hmm.  The fold index t into X_df therefore
    maps to a timestamp that is NOT at position t in X_hmm.  Timestamp-based
    lookup must find the regime correctly regardless of the offset.
    """
    offset = 167  # rows X_df starts earlier than X_hmm
    N      = 600

    # X_df index: starts offset hours earlier
    x_df_idx = pd.date_range("2024-01-01", periods=N, freq="1h", tz="UTC")
    # X_hmm (label_series) index: starts offset hours later
    x_hmm_idx = x_df_idx[offset:]
    labels = pd.Series(["Neutral"] * len(x_hmm_idx), index=x_hmm_idx, name="regime")
    labels.iloc[0] = "Bullish"  # mark first entry distinctly

    t       = offset   # fold index in X_df corresponds to x_hmm_idx[0]
    ts_at_t = x_df_idx[t]

    # Positional lookup (old broken approach): would hit x_hmm_idx[offset] = Neutral
    positional_regime = labels.iloc[min(t, len(labels) - 1)]
    # Timestamp lookup (correct): hits x_hmm_idx[0] = Bullish
    ts_regime = labels.get(ts_at_t, labels.iloc[min(t, len(labels) - 1)])

    assert ts_regime   == "Bullish"   # correct
    assert positional_regime == "Neutral"  # what the old code returned (wrong)
