"""Unit tests for HMM feature engineering — no real Parquet files needed."""

import numpy as np
import pandas as pd
import pytest

from src.hmm.features import (
    ALL_FEATURE_NAMES,
    BtcLagExtractor,
    DisasterExtractor,
    LogDiffReturnExtractor,
    MarketCloseExtractor,
    MaxPainExtractor,
    MilitaryExtractor,
    MomentumExtractor,
    RollingCorrelationExtractor,
    RollingVolatilityExtractor,
    VixLevelExtractor,
    build_feature_matrix,
    load_common_dataframe,
)


def _make_common_df(n: int = 400, with_max_pain: bool = True) -> pd.DataFrame:
    """Minimal common DataFrame with all required symbol + soft-signal columns."""
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    index.name = "timestamp"
    rng = np.random.default_rng(42)
    df = pd.DataFrame(index=index)
    for sym in ["BTC", "ETH", "SOL", "VIX"]:
        price = 100 + rng.standard_normal(n).cumsum()
        df[f"{sym}_close"] = np.abs(price) + 10
        for col in ["open", "high", "low", "volume"]:
            df[f"{sym}_{col}"] = df[f"{sym}_close"] * rng.uniform(0.98, 1.02, n)
    # soft signals
    df["FEMA_score"] = rng.uniform(0, 1, n)
    df["GDELT_military_score"] = rng.uniform(0, 1, n)
    if with_max_pain:
        df["BTC_options_max_pain"] = df["BTC_close"] * rng.uniform(0.8, 1.2, n)
    return df


# ── LogDiffReturnExtractor ────────────────────────────────────────────────────

def test_log_return_columns_added() -> None:
    df = LogDiffReturnExtractor().transform(_make_common_df())
    assert "SOL_log_return" in df.columns
    assert "BTC_log_return" in df.columns


def test_log_return_first_row_is_nan() -> None:
    df = LogDiffReturnExtractor().transform(_make_common_df())
    assert pd.isna(df["SOL_log_return"].iloc[0])


# ── RollingVolatilityExtractor ────────────────────────────────────────────────

def test_rolling_vol_columns_added() -> None:
    df = RollingVolatilityExtractor().transform(_make_common_df())
    assert "SOL_vol_24h" in df.columns
    assert "BTC_vol_168h" in df.columns


# ── RollingCorrelationExtractor ───────────────────────────────────────────────

def test_rolling_corr_columns_added() -> None:
    df = RollingCorrelationExtractor().transform(_make_common_df())
    assert "SOL_BTC_corr_24h" in df.columns
    assert "SOL_ETH_corr_168h" in df.columns


# ── VixLevelExtractor ─────────────────────────────────────────────────────────

def test_vix_features_added() -> None:
    df = VixLevelExtractor().transform(_make_common_df())
    assert "VIX_zscore" in df.columns
    assert "VIX_change_24h" in df.columns


# ── MomentumExtractor ─────────────────────────────────────────────────────────

def test_momentum_columns_added() -> None:
    df = MomentumExtractor().transform(_make_common_df())
    assert "SOL_momentum" in df.columns
    assert "BTC_momentum" in df.columns


# ── BtcLagExtractor ───────────────────────────────────────────────────────────

def test_btc_lag_columns_added() -> None:
    df = BtcLagExtractor().transform(_make_common_df())
    assert "BTC_log_return_lag_24h" in df.columns
    assert "BTC_log_return_lag_168h" in df.columns


def test_btc_lag_24h_is_shifted() -> None:
    df = LogDiffReturnExtractor().transform(_make_common_df())
    df = BtcLagExtractor().transform(df)
    # lag_24h at row 25 == log_return at row 1
    assert df["BTC_log_return_lag_24h"].iloc[25] == pytest.approx(
        df["BTC_log_return"].iloc[1]
    )


# ── DisasterExtractor / MilitaryExtractor ─────────────────────────────────────

def test_disaster_passthrough() -> None:
    df = _make_common_df()
    original = df["FEMA_score"].copy()
    df = DisasterExtractor().transform(df)
    pd.testing.assert_series_equal(df["FEMA_score"], original)


def test_military_passthrough() -> None:
    df = _make_common_df()
    original = df["GDELT_military_score"].copy()
    df = MilitaryExtractor().transform(df)
    pd.testing.assert_series_equal(df["GDELT_military_score"], original)


def test_max_pain_diff_computed() -> None:
    df = MaxPainExtractor().transform(_make_common_df())
    assert "max_pain_diff_usd" in df.columns
    assert "max_pain_diff_pct" in df.columns
    # diff = max_pain - btc_close — should not all be zero
    assert not (df["max_pain_diff_usd"] == 0.0).all()


def test_max_pain_nan_when_column_missing() -> None:
    df = _make_common_df(with_max_pain=False)
    df = MaxPainExtractor().transform(df)
    assert df["max_pain_diff_usd"].isna().all()
    assert df["max_pain_diff_pct"].isna().all()


def test_disaster_fills_zero_when_column_missing() -> None:
    df = _make_common_df().drop(columns=["FEMA_score"])
    df = DisasterExtractor().transform(df)
    assert "FEMA_score" in df.columns
    assert (df["FEMA_score"] == 0.0).all()


# ── MarketCloseExtractor ──────────────────────────────────────────────────────

def test_market_close_nan_when_lib_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    real_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pandas_market_calendars":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    df = MarketCloseExtractor().transform(_make_common_df())
    assert "BTC_at_XETRA_close" in df.columns
    assert df["BTC_at_XETRA_close"].isna().all()


# ── build_feature_matrix ──────────────────────────────────────────────────────

def test_build_feature_matrix_drops_nan() -> None:
    matrix = build_feature_matrix(_make_common_df(), ["BTC_log_return", "VIX_zscore"])
    assert not matrix.isnull().any().any()


def test_build_feature_matrix_always_includes_sol_return() -> None:
    matrix = build_feature_matrix(_make_common_df(), ["BTC_log_return"])
    assert "SOL_log_return" in matrix.columns


def test_build_feature_matrix_raises_on_unknown_feature() -> None:
    with pytest.raises(ValueError, match="Unknown feature"):
        build_feature_matrix(_make_common_df(), ["nonexistent_feature"])


def test_build_feature_matrix_soft_signals_in_range() -> None:
    matrix = build_feature_matrix(
        _make_common_df(),
        ["FEMA_score", "GDELT_military_score"],
    )
    assert (matrix["FEMA_score"] >= 0.0).all()
    assert (matrix["GDELT_military_score"] <= 1.0).all()


# ── load_common_dataframe ─────────────────────────────────────────────────────

def test_load_common_dataframe_raises_without_files(tmp_path: object) -> None:
    config = {"storage": {"raw_dir": str(tmp_path)}}
    with pytest.raises(FileNotFoundError):
        load_common_dataframe(config)
