"""Unit tests for src/backtest/timing.py analysis helpers."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.timing import (
    _block_contribution,
    _hour_weekday_heatmap,
    _weekend_hold_stats,
    _year_robustness,
)


# ── Fixture ───────────────────────────────────────────────────────────────────


def _make_strategy_df(n_weeks: int = 8) -> pd.DataFrame:
    """Synthetic hourly strategy DataFrame spanning multiple weeks and years."""
    rng = np.random.default_rng(42)
    # Start on a Monday
    idx = pd.date_range("2023-06-05 00:00", periods=n_weeks * 7 * 24, freq="1h", tz="UTC")
    n = len(idx)
    lr = rng.normal(0.0, 0.001, n)
    position = np.where(rng.random(n) > 0.3, 1, 0)
    strategy_lr = lr * position
    regime = np.where(position == 1, "Bullish", "Neutral")
    return pd.DataFrame(
        {"strategy_lr": strategy_lr, "position": position, "regime": regime},
        index=idx,
    )


# ── _hour_weekday_heatmap ─────────────────────────────────────────────────────


def test_heatmap_shape_is_7x24() -> None:
    df = _make_strategy_df()
    pivot = _hour_weekday_heatmap(df)
    assert pivot.shape == (7, 24)


def test_heatmap_weekday_index_0_to_6() -> None:
    df = _make_strategy_df()
    pivot = _hour_weekday_heatmap(df)
    assert list(pivot.index) == list(range(7))


def test_heatmap_hour_columns_0_to_23() -> None:
    df = _make_strategy_df()
    pivot = _hour_weekday_heatmap(df)
    assert list(pivot.columns) == list(range(24))


def test_heatmap_bullish_only_excludes_neutral() -> None:
    idx = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "strategy_lr": [0.001] * 48,
            "position": [1] * 48,
            "regime": ["Bullish"] * 24 + ["Bearish"] * 24,
        },
        index=idx,
    )
    pivot_all = _hour_weekday_heatmap(df, bullish_only=False)
    pivot_bull = _hour_weekday_heatmap(df, bullish_only=True)
    # bullish_only excludes Bearish rows → different values
    assert not pivot_all.equals(pivot_bull)


# ── _block_contribution ───────────────────────────────────────────────────────


def test_block_contribution_covers_all_hours() -> None:
    df = _make_strategy_df()
    agg = _block_contribution(df, block_h=3)
    expected_starts = list(range(0, 24, 3))
    assert list(agg["block_start"]) == expected_starts


def test_block_contribution_mean_lr_finite() -> None:
    df = _make_strategy_df()
    agg = _block_contribution(df, block_h=4)
    assert agg["mean_lr"].notna().all()


def test_block_contribution_n_hours_positive() -> None:
    df = _make_strategy_df()
    agg = _block_contribution(df, block_h=6)
    assert (agg["n_hours"] > 0).all()


# ── _weekend_hold_stats ───────────────────────────────────────────────────────


def test_weekend_hold_stats_keys() -> None:
    df = _make_strategy_df()
    stats = _weekend_hold_stats(df)
    for key in ("weekend_lr", "weekday_lr", "weekend_equity", "weekday_equity",
                "weekend_sharpe", "weekday_sharpe", "n_weekend_h", "n_weekday_h"):
        assert key in stats


def test_weekend_hold_stats_counts_sum_to_total() -> None:
    df = _make_strategy_df()
    stats = _weekend_hold_stats(df)
    assert stats["n_weekend_h"] + stats["n_weekday_h"] == len(df)


def test_weekend_hold_stats_friday_19_is_weekend() -> None:
    # Single row: Friday 19:00 UTC
    idx = pd.DatetimeIndex([pd.Timestamp("2024-06-07 19:00", tz="UTC")])
    df = pd.DataFrame(
        {"strategy_lr": [0.001], "position": [1], "regime": ["Bullish"]}, index=idx
    )
    stats = _weekend_hold_stats(df)
    assert stats["n_weekend_h"] == 1
    assert stats["n_weekday_h"] == 0


def test_weekend_hold_stats_monday_08_is_weekday() -> None:
    # Monday 08:00 is NOT in the weekend window (Mon < 8 is weekend)
    idx = pd.DatetimeIndex([pd.Timestamp("2024-06-10 08:00", tz="UTC")])
    df = pd.DataFrame(
        {"strategy_lr": [0.001], "position": [1], "regime": ["Bullish"]}, index=idx
    )
    stats = _weekend_hold_stats(df)
    assert stats["n_weekday_h"] == 1
    assert stats["n_weekend_h"] == 0


# ── _year_robustness ──────────────────────────────────────────────────────────


def test_year_robustness_index_contains_years() -> None:
    df = _make_strategy_df(n_weeks=60)  # ~14 months spanning 2023–2024
    result = _year_robustness(df, top_n=3)
    assert len(result) >= 1
    assert "rank1_hour" in result.columns


def test_year_robustness_hours_in_valid_range() -> None:
    df = _make_strategy_df(n_weeks=60)
    result = _year_robustness(df, top_n=5)
    for col in [c for c in result.columns if c.endswith("_hour")]:
        valid = result[col].dropna()
        assert (valid >= 0).all() and (valid <= 23).all()


def test_year_robustness_top_n_limits_columns() -> None:
    df = _make_strategy_df(n_weeks=60)
    result = _year_robustness(df, top_n=3)
    hour_cols = [c for c in result.columns if c.endswith("_hour")]
    assert len(hour_cols) <= 3
