"""Unit tests for src.backtest.metrics and src.backtest.strategy."""

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
