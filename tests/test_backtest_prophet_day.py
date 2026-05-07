"""Unit tests for src/backtest/prophet_day.py."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.prophet_day import (
    apply_stop_loss,
    build_day_returns,
    find_sell_hour_by_ci,
    sample_days,
    select_candidates,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_sol(start: str = "2023-01-01", periods: int = 24 * 10) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    prices = 100.0 * np.exp(
        np.cumsum(np.random.default_rng(0).normal(0, 0.01, periods))
    )
    return pd.Series(prices, index=idx)


# ── build_day_returns ─────────────────────────────────────────────────────────


def test_build_day_returns_shape():
    sol = _make_sol(periods=24 * 5)
    dr = build_day_returns(sol, sell_hours=[15, 18])
    assert "move_h15" in dr.columns
    assert "move_h18" in dr.columns
    assert "move_best" in dr.columns
    assert len(dr) > 0


def test_build_day_returns_known_value():
    # Construct a SOL series with known prices at 23:00 and 18:00 next day
    idx = pd.date_range("2023-06-01", periods=48, freq="1h", tz="UTC")
    prices = pd.Series(100.0, index=idx)
    # entry: 2023-06-01 23:00 → price 100
    # exit : 2023-06-02 18:00 = entry + 19h → force price to 110
    prices.iloc[23] = 100.0  # 23:00 Jun 1
    prices.iloc[42] = 110.0  # 18:00 Jun 2 = 23:00 + 19h

    dr = build_day_returns(prices, sell_hours=[15, 18])
    assert not dr.empty
    row = dr.iloc[0]
    expected = float(np.log(110.0 / 100.0))
    assert abs(row["move_h18"] - expected) < 1e-9


def test_build_day_returns_move_best():
    sol = _make_sol(periods=24 * 5)
    dr = build_day_returns(sol, sell_hours=[15, 16, 17, 18])
    cols = [f"move_h{h}" for h in [15, 16, 17, 18]]
    for _, row in dr.dropna().iterrows():
        assert abs(row["move_best"]) >= abs(row[cols].values).max() - 1e-12


# ── find_sell_hour_by_ci ──────────────────────────────────────────────────────


def test_find_sell_hour_by_ci_selects_tightest():
    np_exp = np.array([90.0] * 24)
    np_lo = np.array([85.0] * 24)
    np_hi = np.array([95.0] * 24)
    # Override hour 16 to have a much tighter CI
    np_lo[16] = 89.0
    np_hi[16] = 91.0

    sell_hour, ci = find_sell_hour_by_ci(
        np_exp, np_lo, np_hi, sell_hours=[15, 16, 17, 18]
    )
    assert sell_hour == 16
    assert ci < (95.0 - 85.0) / 90.0


def test_find_sell_hour_by_ci_fallback_on_nan():
    np_exp = np.full(24, np.nan)
    np_lo = np.full(24, np.nan)
    np_hi = np.full(24, np.nan)
    sell_hour, ci = find_sell_hour_by_ci(
        np_exp, np_lo, np_hi, sell_hours=[15, 16, 17, 18]
    )
    assert sell_hour == 15  # fallback = first in list
    assert ci == float("inf")


def test_find_sell_hour_by_ci_equal_width_returns_first():
    np_exp = np.full(24, 100.0)
    np_lo = np.full(24, 90.0)
    np_hi = np.full(24, 110.0)
    sell_hour, _ = find_sell_hour_by_ci(
        np_exp, np_lo, np_hi, sell_hours=[15, 16, 17, 18]
    )
    assert sell_hour == 15  # all equal → first wins


# ── apply_stop_loss ───────────────────────────────────────────────────────────


def _make_sol_dict(entry_ts: pd.Timestamp, prices: list[float]) -> dict:
    idx = pd.date_range(entry_ts, periods=len(prices), freq="1h", tz="UTC")
    return dict(zip(idx, [float(p) for p in prices]))


def test_apply_stop_loss_triggered():
    entry = pd.Timestamp("2024-03-01 23:00", tz="UTC")
    sell = entry + pd.Timedelta(hours=19)  # 18:00 next day
    prices = [100.0] + [100.0] * 5 + [88.0] + [100.0] * 13  # drop at hour 6
    sol_dict = _make_sol_dict(entry, prices)
    exit_ts, exit_price, was_stopped, stop_hour = apply_stop_loss(
        sol_dict, entry, sell, stop_pct=0.10
    )
    assert was_stopped
    assert exit_price == pytest.approx(88.0)
    assert stop_hour == (entry + pd.Timedelta(hours=6)).hour


def test_apply_stop_loss_not_triggered():
    entry = pd.Timestamp("2024-03-01 23:00", tz="UTC")
    sell = entry + pd.Timedelta(hours=19)
    prices = [100.0] + [99.0] * 18 + [105.0]  # never drops 10%
    sol_dict = _make_sol_dict(entry, prices)
    exit_ts, exit_price, was_stopped, stop_hour = apply_stop_loss(
        sol_dict, entry, sell, stop_pct=0.10
    )
    assert not was_stopped
    assert exit_ts == sell
    assert stop_hour is None


def test_apply_stop_loss_exactly_at_threshold_not_triggered():
    entry = pd.Timestamp("2024-04-01 23:00", tz="UTC")
    sell = entry + pd.Timedelta(hours=5)
    # Exactly −10 % is NOT > threshold → no stop
    prices = [100.0, 100.0, 100.0, 90.0, 100.0, 100.0]
    sol_dict = _make_sol_dict(entry, prices)
    _, _, was_stopped, _ = apply_stop_loss(sol_dict, entry, sell, stop_pct=0.10)
    assert not was_stopped  # 90/100 - 1 = -0.10, which is NOT < -0.10


# ── sample_days ───────────────────────────────────────────────────────────────


def test_sample_days_respects_n():
    rng = np.random.default_rng(42)
    candidates = pd.date_range("2024-01-02", periods=100, freq="7D", tz="UTC").tolist()
    labels = pd.Series("Bullish", index=candidates)
    feat_df = pd.DataFrame(
        {
            "SOL_vol_168h": rng.uniform(0.01, 0.05, 100),
            "VIX_zscore": rng.normal(0, 1, 100),
        },
        index=candidates,
    )
    sampled = sample_days(candidates, labels, feat_df, n=20)
    assert len(sampled) <= 20
    assert len(sampled) > 0


def test_sample_days_stratifies_by_year():
    candidates_24 = pd.date_range(
        "2024-01-15 23:00", periods=20, freq="14D", tz="UTC"
    ).tolist()
    candidates_25 = pd.date_range(
        "2025-01-15 23:00", periods=20, freq="14D", tz="UTC"
    ).tolist()
    candidates = candidates_24 + candidates_25
    labels = pd.Series("Bullish", index=candidates)
    rng = np.random.default_rng(1)
    feat_df = pd.DataFrame(
        {
            "SOL_vol_168h": rng.uniform(0.01, 0.05, 40),
            "VIX_zscore": rng.normal(0, 1, 40),
        },
        index=candidates,
    )
    sampled = sample_days(candidates, labels, feat_df, n=10)
    years = {t.year for t in sampled}
    assert 2024 in years
    assert 2025 in years


# ── select_candidates ─────────────────────────────────────────────────────────


def test_select_candidates_filters_weekends():
    # Build hourly feat_df spanning Mon–Sun at 23:00
    base = pd.Timestamp("2024-01-01 23:00", tz="UTC")  # Monday
    ts_list = [base + pd.Timedelta(days=i) for i in range(7)]
    idx = pd.DatetimeIndex(ts_list)
    feat_df = pd.DataFrame(
        {
            "SOL_vol_168h": [0.05] * 7,
            "BTC_momentum": [0.1] * 7,
            "VIX_zscore": [1.0] * 7,
        },
        index=idx,
    )
    labels = pd.Series("Bullish", index=idx)
    result = select_candidates(feat_df, labels, vol_median=0.01)
    for ts in result:
        assert ts.dayofweek < 5  # no Saturday or Sunday


def test_select_candidates_regime_filter():
    base = pd.Timestamp("2024-03-04 23:00", tz="UTC")  # Monday
    idx = pd.DatetimeIndex([base, base + pd.Timedelta(days=1)])
    feat_df = pd.DataFrame(
        {
            "SOL_vol_168h": [0.05, 0.05],
            "BTC_momentum": [0.1, 0.1],
            "VIX_zscore": [1.0, 1.0],
        },
        index=idx,
    )
    labels = pd.Series(["Bullish", "Bearish"], index=idx)
    result = select_candidates(feat_df, labels, vol_median=0.01)
    assert len(result) == 1
    assert result[0] == base
