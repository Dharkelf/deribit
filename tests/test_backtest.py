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


# ── trading hours filter ──────────────────────────────────────────────────────


def _make_mixed_hours(n: int = 48) -> tuple[pd.Series, pd.Series]:
    """Return (log_returns, regime_labels) spanning two days, hours 0–23."""
    idx    = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    lr     = pd.Series(np.full(n, 0.001), index=idx)
    labels = pd.Series(["Bullish"] * n, index=idx)
    return lr, labels


def test_trading_hours_off_hours_position_is_zero():
    """Hours outside [8, 18) must have position zeroed."""
    lr, labels = _make_mixed_hours()
    result = RegimeStrategy().apply(lr, labels, trading_hours=(8, 18))
    off = result[result["off_hours"]]
    assert (off["position"] == 0.0).all()


def test_trading_hours_on_hours_position_unchanged():
    """Hours inside [8, 18) must carry the normal regime position."""
    lr, labels = _make_mixed_hours()
    result = RegimeStrategy().apply(lr, labels, trading_hours=(8, 18))
    on = result[~result["off_hours"]]
    assert (on["position"] == 0.5).all()   # Bullish = 0.5


def test_trading_hours_correct_hour_count():
    """For a 48-hour window, 24 hours fall inside [8, 18) (10 per day × 2)."""
    lr, labels = _make_mixed_hours(48)
    result = RegimeStrategy().apply(lr, labels, trading_hours=(8, 18))
    assert result["off_hours"].sum() == 28   # 14 off-hours per day × 2
    assert (~result["off_hours"]).sum() == 20  # 10 on-hours per day × 2


def test_trading_hours_none_all_hours_active():
    """Without a filter every row should have off_hours=False."""
    lr, labels = _make_mixed_hours()
    result = RegimeStrategy().apply(lr, labels, trading_hours=None)
    assert not result["off_hours"].any()


def test_trading_hours_off_hours_strategy_lr_is_zero():
    """off_hours rows must contribute zero return regardless of market move."""
    lr     = pd.Series(
        [0.05] * 48,
        index=pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC"),
    )
    labels = pd.Series(["Strong Bullish"] * 48, index=lr.index)
    result = RegimeStrategy().apply(lr, labels, trading_hours=(8, 18))
    off = result[result["off_hours"]]
    assert np.allclose(off["strategy_lr"].values, 0.0, atol=1e-12)


def test_trading_hours_applied_before_trailing_stop():
    """Off-hours returns must not influence trailing stop equity tracking.

    With [8, 18) filter and large off-hours losses, the trailing stop should
    not fire (because off-hours positions are zero, so equity stays flat).
    """
    idx = pd.date_range("2026-01-01 00:00", periods=24, freq="1h", tz="UTC")
    # Large losses at hours 0–7 and 18–23 (off-hours); small gains at 8–17
    lr_vals = np.where(
        (idx.hour < 8) | (idx.hour >= 18),
        -0.10,   # −10 % per hour in off-hours
        0.001,   # +0.1 % per hour in on-hours
    )
    lr     = pd.Series(lr_vals, index=idx)
    labels = pd.Series(["Strong Bullish"] * 24, index=idx)

    result = RegimeStrategy().apply(
        lr, labels, trailing_stop_pct=15, trading_hours=(8, 18)
    )
    # Stop must NOT have fired (off-hours losses are invisible to stop tracking)
    assert not result["stopped"].any()


def test_strategy_stopped_column_absent_without_stop():
    lr, labels = _make_inputs()
    result = RegimeStrategy().apply(lr, labels)
    assert "stopped" in result.columns
    assert not result["stopped"].any()


def test_trailing_stop_none_is_equivalent_to_no_stop():
    lr, labels = _make_inputs(n=50, regime="Bullish")
    r_no_stop = RegimeStrategy().apply(lr, labels)
    r_none    = RegimeStrategy().apply(lr, labels, trailing_stop_pct=None)
    assert (r_no_stop["position"].values == r_none["position"].values).all()


def test_trailing_stop_fires_on_large_drawdown():
    """Sustained losses should trigger the stop, forcing position to zero."""
    idx    = pd.date_range("2026-01-01", periods=40, freq="1h", tz="UTC")
    # Strong Bullish (+1 position) with large losses every step → stop fires
    lr     = pd.Series([-0.05] * 40, index=idx)      # −5 % per hour compounded
    labels = pd.Series(["Strong Bullish"] * 40, index=idx)
    result = RegimeStrategy().apply(lr, labels, trailing_stop_pct=15)
    # After enough losses the stop must have fired: some hours stopped
    assert result["stopped"].any()
    # All hours after stop fires must have position 0 (within same regime phase)
    stopped_idx = result.index[result["stopped"]]
    assert (result.loc[stopped_idx, "position"] == 0.0).all()


def test_trailing_stop_resets_on_regime_change():
    """After a regime change the position should resume (stop cleared)."""
    idx = pd.date_range("2026-01-01", periods=60, freq="1h", tz="UTC")
    lr  = pd.Series([-0.05] * 60, index=idx)

    # First 30 hours: Strong Bullish (stop fires) → next 30: Bullish (stop clears)
    labels = pd.Series(
        ["Strong Bullish"] * 30 + ["Bullish"] * 30, index=idx
    )
    result = RegimeStrategy().apply(lr, labels, trailing_stop_pct=15)

    # Stop must fire during first regime phase
    assert result.loc[result.index[:30], "stopped"].any()

    # After regime change the first few hours of "Bullish" must NOT be stopped
    # (stop resets at boundary)
    assert not result.loc[result.index[30], "stopped"]


def test_trailing_stop_does_not_fire_on_small_drawdown():
    """A 1 % drawdown should never fire a 15 % trailing stop."""
    idx    = pd.date_range("2026-01-01", periods=20, freq="1h", tz="UTC")
    lr     = pd.Series([-0.001] * 20, index=idx)   # −0.1 % per hour
    labels = pd.Series(["Bullish"] * 20, index=idx)
    result = RegimeStrategy().apply(lr, labels, trailing_stop_pct=15)
    assert not result["stopped"].any()


def test_trailing_stop_equity_is_non_decreasing_when_stopped():
    """While stopped (position=0) equity must stay flat."""
    idx    = pd.date_range("2026-01-01", periods=60, freq="1h", tz="UTC")
    lr     = pd.Series([-0.05] * 60, index=idx)
    labels = pd.Series(["Strong Bullish"] * 60, index=idx)
    result = RegimeStrategy().apply(lr, labels, trailing_stop_pct=15)

    stopped_rows = result[result["stopped"]]
    if not stopped_rows.empty:
        # strategy_lr must be 0 while stopped (position=0); use allclose to tolerate -0.0
        assert np.allclose(stopped_rows["strategy_lr"].values, 0.0, atol=1e-12)


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


# ── discrete trading ──────────────────────────────────────────────────────────

def _make_discrete_inputs(
    n: int = 48,
    regime: str = "Bullish",
    lr_val: float = 0.001,
    start: str = "2026-01-01 06:00",
) -> tuple[pd.Series, pd.Series]:
    idx    = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    lr     = pd.Series([lr_val] * n, index=idx)
    labels = pd.Series([regime] * n, index=idx)
    return lr, labels


def test_discrete_holds_for_min_hours_weak_regime():
    """Weak regime → position fixed for min_hold_h hours then re-evaluated."""
    lr, labels = _make_discrete_inputs(n=12, regime="Bullish", lr_val=0.001)
    result = RegimeStrategy().apply(
        lr, labels,
        discrete_trading=(3, 6),
        trading_window=(6, 19),
    )
    pos = result["position"].values
    # Hours 0-2 (06:00-08:00): first hold, position=0.5
    assert np.all(pos[:3] == pytest.approx(0.5))
    # Hours 3-5 (09:00-11:00): second hold, same regime → same position
    assert np.all(pos[3:6] == pytest.approx(0.5))


def test_discrete_holds_for_max_hours_strong_regime():
    """Strong regime → position fixed for max_hold_h hours."""
    lr, labels = _make_discrete_inputs(n=12, regime="Strong Bullish", lr_val=0.001)
    result = RegimeStrategy().apply(
        lr, labels,
        discrete_trading=(3, 6),
        trading_window=(6, 19),
    )
    pos = result["position"].values
    # First 6 hours: one hold block, all position=1.0
    assert np.all(pos[:6] == pytest.approx(1.0))


def test_discrete_zeros_outside_trading_window():
    """Position must be 0 outside the trading window."""
    # Start at 04:00 UTC — first 2 hours are outside [6, 19)
    lr, labels = _make_discrete_inputs(n=20, regime="Bullish", start="2026-01-01 04:00")
    result = RegimeStrategy().apply(
        lr, labels,
        discrete_trading=(3, 6),
        trading_window=(6, 19),
    )
    off = result["off_hours"].values
    pos = result["position"].values
    # Hours before window start must be off_hours and position=0
    assert np.all(off[:2])
    assert np.all(pos[:2] == pytest.approx(0.0))


def test_discrete_position_zero_after_window_ends():
    """After window closes (19:00 UTC) position must be 0."""
    # 13 hours starting at 10:00 → 10,11,...,22 UTC; window closes at 19
    lr, labels = _make_discrete_inputs(n=13, regime="Bullish", start="2026-01-01 10:00")
    result = RegimeStrategy().apply(
        lr, labels,
        discrete_trading=(3, 6),
        trading_window=(6, 19),
    )
    pos = result["position"].values
    # Last entries (19:00+) must be 0
    hours = result.index.hour.values
    assert np.all(pos[hours >= 19] == pytest.approx(0.0))


def test_discrete_stop_fires_mid_hold():
    """Trailing stop must fire within a hold period on large losses."""
    # Strong Bullish, position=+1.0, large negative returns → stop must fire
    lr, labels = _make_discrete_inputs(
        n=12, regime="Strong Bullish", lr_val=-0.10,
    )
    result = RegimeStrategy().apply(
        lr, labels,
        discrete_trading=(6, 6),
        trading_window=(6, 19),
        trailing_stop_pct=15,
    )
    # Stop must fire during the 6-hour hold
    assert result["stopped"].any()
    # Once stopped, strategy_lr must be 0
    stopped_rows = result[result["stopped"]]
    assert np.allclose(stopped_rows["strategy_lr"].values, 0.0, atol=1e-12)


def test_discrete_peak_resets_at_new_trade():
    """Peak resets at each trade entry: stop from prior trade does not carry over."""
    # Two 3-hour holds: first has large loss (stop fires), second has small gain
    # If peak did NOT reset, second trade would be stopped immediately
    n      = 12
    idx    = pd.date_range("2026-01-01 06:00", periods=n, freq="1h", tz="UTC")
    lr_val = [-0.10] * 6 + [0.001] * 6   # first 6h: big loss; next 6h: small gain
    lr     = pd.Series(lr_val, index=idx)
    labels = pd.Series(["Strong Bullish"] * n, index=idx)

    result = RegimeStrategy().apply(
        lr, labels,
        discrete_trading=(6, 6),
        trading_window=(6, 19),
        trailing_stop_pct=15,
    )
    # Second hold (hours 6-11): peak resets → stop should NOT be active immediately
    second_hold = result.iloc[6:]
    # At least the first hour of second hold must not be stopped
    assert not result.iloc[6]["stopped"]


def test_discrete_neutral_regime_zero_position():
    """Neutral regime → position=0 in discrete mode too."""
    lr, labels = _make_discrete_inputs(n=12, regime="Neutral", lr_val=0.001)
    result = RegimeStrategy().apply(
        lr, labels,
        discrete_trading=(3, 6),
        trading_window=(6, 19),
    )
    pos = result["position"]
    assert np.all(pos[result["off_hours"] == False].values == pytest.approx(0.0))  # noqa: E712


def test_discrete_hourly_mode_unchanged():
    """When discrete_trading=None, hourly mode is used (no regression)."""
    lr, labels = _make_discrete_inputs(n=24, regime="Bullish", lr_val=0.001)
    r_hourly   = RegimeStrategy().apply(lr, labels)
    r_discrete = RegimeStrategy().apply(
        lr, labels, discrete_trading=(3, 6), trading_window=(6, 19)
    )
    # In hourly mode every hour is active; discrete mode has off-hours zeros
    assert r_hourly["position"].sum() > r_discrete["position"].sum()


def test_long_only_no_short_positions():
    """long_only=True: positions must never be negative."""
    idx    = pd.date_range("2026-01-01 06:00", periods=24, freq="1h", tz="UTC")
    lr     = pd.Series([0.001] * 24, index=idx)
    labels = pd.Series(
        ["Strong Bearish", "Bearish", "Neutral", "Bullish", "Strong Bullish"] * 4 + ["Neutral"] * 4,
        index=idx,
    )
    result = RegimeStrategy().apply(
        lr, labels,
        discrete_trading=(3, 6),
        trading_window=(6, 19),
        long_only=True,
    )
    assert (result["position"] >= 0.0).all()


def test_long_only_bullish_enters_position():
    """long_only=True: Bullish regime must produce positive position."""
    idx    = pd.date_range("2026-01-01 06:00", periods=6, freq="1h", tz="UTC")
    lr     = pd.Series([0.001] * 6, index=idx)
    labels = pd.Series(["Strong Bullish"] * 6, index=idx)
    result = RegimeStrategy().apply(
        lr, labels,
        discrete_trading=(3, 6),
        trading_window=(6, 19),
        long_only=True,
    )
    in_window = result[~result["off_hours"]]
    assert (in_window["position"] > 0.0).any()


# ── Option C composite gate ───────────────────────────────────────────────────


def test_xgb_conflict_halves_position():
    """XGB direction conflict → position scaled to 0.5× of HMM value."""
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    lr = pd.Series([0.001] * 4, index=idx)
    labels = pd.Series(["Bullish"] * 4, index=idx)  # HMM position = +0.5
    # XGB says down (-1) → conflicts with Bullish (+0.5)
    xgb_sig = pd.Series([-1.0] * 4, index=idx)
    result = RegimeStrategy().apply(lr, labels, xgb_signal=xgb_sig)
    # All hours in conflict → position should be 0.5 × 0.5 = 0.25
    assert result["position"].iloc[0] == pytest.approx(0.25)


def test_xgb_agreement_keeps_full_position():
    """XGB direction agreement → position unchanged."""
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    lr = pd.Series([0.001] * 4, index=idx)
    labels = pd.Series(["Bullish"] * 4, index=idx)  # HMM position = +0.5
    xgb_sig = pd.Series([1.0] * 4, index=idx)       # XGB agrees → up
    result = RegimeStrategy().apply(lr, labels, xgb_signal=xgb_sig)
    assert result["position"].iloc[0] == pytest.approx(0.5)


def test_persistence_scales_position():
    """High persistence (p=1.0) → factor=1.0; low (p=0.0) → factor=0.5."""
    idx = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    lr = pd.Series([0.001, 0.001], index=idx)
    labels = pd.Series(["Bullish", "Bullish"], index=idx)  # pos = +0.5
    pers = pd.Series([1.0, 0.0], index=idx)
    result = RegimeStrategy().apply(lr, labels, persistence=pers)
    assert result["position"].iloc[0] == pytest.approx(0.5 * 1.0)   # 0.5 + 0.5*1 = 1.0
    assert result["position"].iloc[1] == pytest.approx(0.5 * 0.5)   # 0.5 + 0.5*0 = 0.5


def test_allowed_hours_zeroes_positions_outside_set():
    """Hourly mode: hours not in allowed_hours must have position=0."""
    idx = pd.date_range("2026-01-05 00:00", periods=24, freq="1h", tz="UTC")  # Monday
    lr = pd.Series([0.001] * 24, index=idx)
    labels = pd.Series(["Bullish"] * 24, index=idx)
    result = RegimeStrategy().apply(lr, labels, allowed_hours=[6, 7, 8])
    assert (result["position"][~result.index.hour.isin([6, 7, 8])] == 0.0).all()
    assert (result["position"][result.index.hour.isin([6, 7, 8])] == 0.5).all()


def test_discrete_allowed_hours_skips_entry_outside_set():
    """Discrete mode: entry must not happen at hour outside allowed_hours."""
    # 24h window, only hour 6 allowed — all entries must be at hour 6
    idx = pd.date_range("2026-01-05 00:00", periods=48, freq="1h", tz="UTC")
    lr = pd.Series([0.001] * 48, index=idx)
    labels = pd.Series(["Bullish"] * 48, index=idx)
    result = RegimeStrategy().apply(
        lr, labels,
        discrete_trading=(3, 6),
        allowed_hours=[6, 7, 8],
    )
    # No position outside allowed hours
    outside = result[~result.index.hour.isin([6, 7, 8])]
    assert (outside["position"] == 0.0).all()
