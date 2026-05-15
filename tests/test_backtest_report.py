"""Unit tests for src.backtest.report — output generation and improvement ideas."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.report import _improvement_ideas, generate


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_fold_df(n_folds: int = 5, horizon: int = 24) -> pd.DataFrame:
    """Minimal fold DataFrame matching engine.run() output schema."""
    records = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for fold_id in range(n_folds):
        fold_start = base + pd.Timedelta(days=fold_id * 7)
        regime = ["Bullish", "Neutral", "Bearish", "Strong Bullish", "Strong Bearish"][fold_id % 5]
        start_price = 100.0 + fold_id
        for h in range(1, horizon + 1):
            ts     = fold_start + pd.Timedelta(hours=h)
            actual = start_price + h * 0.1
            pred   = actual + np.random.default_rng(fold_id + h).standard_normal() * 0.5
            records.append({
                "fold_id":     fold_id,
                "horizon_h":   h,
                "actual":      actual,
                "xgb_pred":    pred,
                "regime":      regime,
                "start_price": start_price,
            })
    df = pd.DataFrame(records)
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="1h", tz="UTC")
    return df


def _make_strategy_df(n: int = 500) -> pd.DataFrame:
    """Minimal strategy DataFrame matching strategy.apply() output schema."""
    idx      = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng      = np.random.default_rng(42)
    lr       = rng.standard_normal(n) * 0.005
    pos      = np.where(np.arange(n) % 3 == 0, 0.5, 0.0)
    strat_lr = pos * lr
    return pd.DataFrame(
        {
            "regime":          np.where(np.arange(n) % 2 == 0, "Bullish", "Neutral"),
            "position":        pos,
            "off_hours":       np.zeros(n, dtype=bool),
            "stopped":         np.zeros(n, dtype=bool),
            "strategy_lr":     strat_lr,
            "bnh_lr":          lr,
            "equity_strategy": np.exp(np.cumsum(strat_lr)),
            "equity_bnh":      np.exp(np.cumsum(lr)),
        },
        index=idx,
    )


def _minimal_config(tmp_path) -> dict:
    return {
        "storage": {"processed_dir": str(tmp_path)},
        "backtest": {
            "min_train_days": 30,
            "step_days":      7,
            "horizon_hours":  24,
            "trailing_stop_pct": 15,
            "trading_hours":  None,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# generate() — output file creation
# ─────────────────────────────────────────────────────────────────────────────


def test_generate_creates_parquet(tmp_path):
    fold_df     = _make_fold_df()
    strategy_df = _make_strategy_df()
    generate(fold_df, strategy_df, _minimal_config(tmp_path))
    assert (tmp_path / "backtest_results.parquet").exists()


def test_generate_creates_png(tmp_path):
    fold_df     = _make_fold_df()
    strategy_df = _make_strategy_df()
    generate(fold_df, strategy_df, _minimal_config(tmp_path))
    assert (tmp_path / "backtest_report.png").exists()


def test_generate_creates_markdown(tmp_path):
    fold_df     = _make_fold_df()
    strategy_df = _make_strategy_df()
    generate(fold_df, strategy_df, _minimal_config(tmp_path))
    assert (tmp_path / "BACKTEST_REPORT.md").exists()


def test_generate_parquet_roundtrip(tmp_path):
    fold_df = _make_fold_df()
    generate(fold_df, _make_strategy_df(), _minimal_config(tmp_path))
    loaded = pd.read_parquet(tmp_path / "backtest_results.parquet")
    assert set(["fold_id", "horizon_h", "actual", "xgb_pred", "regime"]).issubset(loaded.columns)
    assert len(loaded) == len(fold_df)


def test_generate_markdown_has_required_sections(tmp_path):
    generate(_make_fold_df(), _make_strategy_df(), _minimal_config(tmp_path))
    md = (tmp_path / "BACKTEST_REPORT.md").read_text()
    assert "## 1. Datenbasis"           in md
    assert "## 2. Forecast-Accuracy"   in md
    assert "## 3. Regime-Strategie"    in md
    assert "## 4. Strategie-Vergleich" in md
    assert "## 5. Jahresweise"         in md
    assert "## 6. Verbesserungsideen"  in md


def test_generate_markdown_shows_trading_hours(tmp_path):
    cfg = _minimal_config(tmp_path)
    cfg["backtest"]["trading_hours"] = [8, 18]
    generate(_make_fold_df(), _make_strategy_df(), cfg)
    md = (tmp_path / "BACKTEST_REPORT.md").read_text()
    assert "08:00" in md


def test_generate_markdown_shows_trailing_stop(tmp_path):
    generate(_make_fold_df(), _make_strategy_df(), _minimal_config(tmp_path))
    md = (tmp_path / "BACKTEST_REPORT.md").read_text()
    assert "Trailing Stop" in md or "trailing" in md.lower()


def test_generate_accepts_multi_variant_dict(tmp_path):
    """generate() with dict[str, DataFrame] writes comparison table."""
    strategies = {
        "variant_a": _make_strategy_df(),
        "variant_b": _make_strategy_df(),
    }
    generate(_make_fold_df(), strategies, _minimal_config(tmp_path))
    md = (tmp_path / "BACKTEST_REPORT.md").read_text()
    assert "## 4. Strategie-Vergleich" in md
    assert "variant_a" in md
    assert "variant_b" in md


# ─────────────────────────────────────────────────────────────────────────────
# _improvement_ideas()
# ─────────────────────────────────────────────────────────────────────────────


def test_improvement_ideas_always_has_np_reminder():
    ideas = _improvement_ideas(_make_fold_df(), _make_strategy_df())
    assert any("NeuralProphet" in idea for idea in ideas)


def test_improvement_ideas_mdd_suppressed_when_stop_active():
    strategy_df = _make_strategy_df()
    # Simulate a large drawdown but with trailing stop active
    strategy_df["equity_strategy"] = np.exp(np.linspace(0, -1.0, len(strategy_df)))  # −63 % MDD
    strategy_df["stopped"]         = True  # stop was active
    ideas = _improvement_ideas(_make_fold_df(), strategy_df)
    assert not any("Stop-Loss" in idea for idea in ideas)


def test_improvement_ideas_mdd_suggested_when_stop_inactive():
    strategy_df = _make_strategy_df()
    strategy_df["equity_strategy"] = np.exp(np.linspace(0, -1.0, len(strategy_df)))  # −63 % MDD
    strategy_df["stopped"]         = False
    ideas = _improvement_ideas(_make_fold_df(), strategy_df)
    assert any("Stop-Loss" in idea or "Drawdown" in idea for idea in ideas)


def test_improvement_ideas_no_history_warning_for_long_window():
    # Build a fold_df whose index spans > 700 days so the backfill warning is suppressed.
    n_folds  = 5
    horizon  = 24
    records  = []
    for fold_id in range(n_folds):
        regime = ["Bullish", "Neutral", "Bearish", "Strong Bullish", "Strong Bearish"][fold_id]
        fold_start = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=fold_id * 7)
        for h in range(1, horizon + 1):
            records.append({
                "fold_id":     fold_id,
                "horizon_h":   h,
                "actual":      100.0,
                "xgb_pred":    100.1,
                "regime":      regime,
                "start_price": 100.0,
            })
    fold_df = pd.DataFrame(records)
    # Index must span ≥700 days; freq="7D" × 120 rows = 840 days
    fold_df.index = pd.date_range("2021-01-01", periods=len(fold_df), freq="7D", tz="UTC")
    ideas = _improvement_ideas(fold_df, _make_strategy_df())
    assert not any("history_days" in idea for idea in ideas)
