"""Unit tests for HMM feature engineering — no real Parquet files needed."""

import numpy as np
import pandas as pd
import pytest

from src.hmm.features import (
    ALL_FEATURE_NAMES,
    LogReturnExtractor,
    MomentumExtractor,
    RollingCorrelationExtractor,
    RollingVolatilityExtractor,
    VixLevelExtractor,
    build_feature_matrix,
    load_common_dataframe,
)


def _make_common_df(n: int = 300) -> pd.DataFrame:
    """Minimal common DataFrame with all required symbol columns."""
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    index.name = "timestamp"
    rng = np.random.default_rng(42)
    df = pd.DataFrame(index=index)
    for sym in ["BTC", "ETH", "SOL", "VIX"]:
        price = 100 + rng.standard_normal(n).cumsum()
        df[f"{sym}_close"] = np.abs(price) + 10
        for col in ["open", "high", "low", "volume"]:
            df[f"{sym}_{col}"] = df[f"{sym}_close"] * rng.uniform(0.98, 1.02, n)
    return df


# --- LogReturnExtractor ---

def test_log_return_columns_added() -> None:
    df = LogReturnExtractor().transform(_make_common_df())
    assert "SOL_log_return" in df.columns
    assert "BTC_log_return" in df.columns


def test_log_return_first_row_is_nan() -> None:
    df = LogReturnExtractor().transform(_make_common_df())
    assert pd.isna(df["SOL_log_return"].iloc[0])


# --- RollingVolatilityExtractor ---

def test_rolling_vol_columns_added() -> None:
    df = RollingVolatilityExtractor().transform(_make_common_df())
    assert "SOL_vol_24h" in df.columns
    assert "BTC_vol_168h" in df.columns


# --- RollingCorrelationExtractor ---

def test_rolling_corr_columns_added() -> None:
    df = RollingCorrelationExtractor().transform(_make_common_df())
    assert "SOL_BTC_corr_24h" in df.columns
    assert "SOL_ETH_corr_168h" in df.columns


# --- VixLevelExtractor ---

def test_vix_features_added() -> None:
    df = VixLevelExtractor().transform(_make_common_df())
    assert "VIX_zscore" in df.columns
    assert "VIX_change_24h" in df.columns


# --- MomentumExtractor ---

def test_momentum_columns_added() -> None:
    df = MomentumExtractor().transform(_make_common_df())
    assert "SOL_momentum" in df.columns
    assert "BTC_momentum" in df.columns


# --- build_feature_matrix ---

def test_build_feature_matrix_drops_nan() -> None:
    raw = _make_common_df()
    matrix = build_feature_matrix(raw, ["BTC_log_return", "VIX_zscore"])
    assert not matrix.isnull().any().any()


def test_build_feature_matrix_always_includes_sol_return() -> None:
    raw = _make_common_df()
    matrix = build_feature_matrix(raw, ["BTC_log_return"])
    assert "SOL_log_return" in matrix.columns


def test_build_feature_matrix_raises_on_unknown_feature() -> None:
    raw = _make_common_df()
    with pytest.raises(ValueError, match="Unknown feature"):
        build_feature_matrix(raw, ["nonexistent_feature"])


def test_all_feature_names_are_producible() -> None:
    raw = _make_common_df()
    matrix = build_feature_matrix(raw, ALL_FEATURE_NAMES)
    for name in ALL_FEATURE_NAMES:
        assert name in matrix.columns


# --- load_common_dataframe raises when files missing ---

def test_load_common_dataframe_raises_without_files(tmp_path: object) -> None:
    config = {"storage": {"raw_dir": str(tmp_path)}}
    with pytest.raises(FileNotFoundError):
        load_common_dataframe(config)
