"""Unit tests for HMM feature engineering — no real Parquet files needed."""

import numpy as np
import pandas as pd
import pytest

from src.hmm.features import (
    ALL_FEATURE_NAMES,
    BtcLagExtractor,
    CryptoFearGreedExtractor,
    DisasterExtractor,
    FedRateExtractor,
    FundingRateExtractor,
    IVSkewExtractor,
    LogDiffReturnExtractor,
    MarketCloseExtractor,
    MaxPainExtractor,
    MilitaryExtractor,
    MomentumExtractor,
    OIRatioExtractor,
    RollingCorrelationExtractor,
    RollingVolatilityExtractor,
    StockFearGreedExtractor,
    VixLevelExtractor,
    build_feature_matrix,
    load_common_dataframe,
)


def _make_common_df(
    n: int = 400,
    with_max_pain: bool = True,
    with_sentiment: bool = True,
    with_fed: bool = True,
    with_funding: bool = True,
    with_oi: bool = True,
    with_iv_skew: bool = True,
) -> pd.DataFrame:
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
    if with_sentiment:
        df["crypto_fear_greed"] = rng.uniform(0, 1, n)
        df["stock_fear_greed"] = rng.uniform(0, 1, n)
    if with_fed:
        df["fed_rate"] = 4.33
        df["fed_rate_last_change"] = -0.25
    if with_funding:
        df["funding_rate_8h"] = rng.uniform(-0.001, 0.001, n)
    if with_oi:
        df["SOL_OI_BTC_ratio"] = rng.uniform(0.01, 0.05, n)
    if with_iv_skew:
        df["btc_iv_skew"] = rng.uniform(-5.0, 5.0, n)
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
    for h in (1, 2, 3, 6, 12, 18, 24):
        assert f"BTC_log_return_lag_{h}h" in df.columns
    for h in (48, 72, 168):
        assert f"BTC_log_return_lag_{h}h" not in df.columns


def test_btc_lag_24h_is_shifted() -> None:
    df = LogDiffReturnExtractor().transform(_make_common_df())
    df = BtcLagExtractor().transform(df)
    # lag_24h at row 25 == log_return at row 1
    assert df["BTC_log_return_lag_24h"].iloc[25] == pytest.approx(
        df["BTC_log_return"].iloc[1]
    )


def test_btc_lag_1h_is_shifted() -> None:
    df = LogDiffReturnExtractor().transform(_make_common_df())
    df = BtcLagExtractor().transform(df)
    # lag_1h at row 2 == log_return at row 1
    assert df["BTC_log_return_lag_1h"].iloc[2] == pytest.approx(
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


def test_max_pain_ratio_computed() -> None:
    rng = np.random.default_rng(0)
    df = _make_common_df()
    df["BTC_options_max_pain_7d"] = df["BTC_close"] * rng.uniform(0.9, 1.1, len(df))
    df = MaxPainExtractor().transform(df)
    assert "max_pain_ratio" in df.columns
    assert "max_pain_diff_pct" in df.columns
    assert "max_pain_7d_ratio" in df.columns
    assert "max_pain_7d_diff_pct" in df.columns
    # ratio = max_pain / BTC_close — should vary around 0.8–1.2 given fixture
    assert df["max_pain_ratio"].dropna().between(0.5, 2.0).all()
    assert df["max_pain_7d_ratio"].dropna().between(0.5, 2.0).all()


def test_max_pain_nan_when_column_missing() -> None:
    df = _make_common_df(with_max_pain=False)
    df = MaxPainExtractor().transform(df)
    assert df["max_pain_ratio"].isna().all()
    assert df["max_pain_diff_pct"].isna().all()
    assert df["max_pain_7d_ratio"].isna().all()
    assert df["max_pain_7d_diff_pct"].isna().all()


def test_disaster_fills_zero_when_column_missing() -> None:
    df = _make_common_df().drop(columns=["FEMA_score"])
    df = DisasterExtractor().transform(df)
    assert "FEMA_score" in df.columns
    assert (df["FEMA_score"] == 0.0).all()


# ── MarketCloseExtractor ──────────────────────────────────────────────────────

def test_market_close_nan_when_lib_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)  # type: ignore[arg-type]
    df = MarketCloseExtractor().transform(_make_common_df())
    assert "BTC_at_XETRA_close" in df.columns
    assert df["BTC_at_XETRA_close"].isna().all()


def test_market_close_columns_non_nan_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MarketCloseExtractor produces non-NaN columns when mcal returns a valid schedule."""
    import sys
    from unittest.mock import MagicMock

    n = 400
    df = _make_common_df(n=n)

    # Mock schedule: 10 daily closes within the DataFrame's date range
    mock_schedule = pd.DataFrame({
        "market_close": (
            pd.date_range("2024-01-02 16:00", periods=10, freq="D", tz="UTC")
            # Use microsecond resolution to simulate real mcal output
            .astype("datetime64[us, UTC]")
        ),
    })
    mock_cal = MagicMock()
    mock_cal.schedule.return_value = mock_schedule

    mock_mcal = MagicMock()
    mock_mcal.get_calendar.return_value = mock_cal

    monkeypatch.setitem(sys.modules, "pandas_market_calendars", mock_mcal)

    result = MarketCloseExtractor().transform(df)

    for ex in ("XETRA", "NYSE", "TSE"):
        assert f"BTC_at_{ex}_close" in result.columns
        assert f"BTC_return_since_{ex}_close" in result.columns
        assert not result[f"BTC_at_{ex}_close"].isna().all(), (
            f"BTC_at_{ex}_close is all-NaN"
        )


def test_market_close_uses_correct_calendar_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exchange labels map to the correct mcal calendar identifiers."""
    import sys
    from unittest.mock import MagicMock

    mock_schedule = pd.DataFrame({
        "market_close": (
            pd.date_range("2024-01-02 16:00", periods=5, freq="D", tz="UTC")
            .astype("datetime64[us, UTC]")
        ),
    })
    mock_cal = MagicMock()
    mock_cal.schedule.return_value = mock_schedule

    mock_mcal = MagicMock()
    mock_mcal.get_calendar.return_value = mock_cal

    monkeypatch.setitem(sys.modules, "pandas_market_calendars", mock_mcal)

    MarketCloseExtractor().transform(_make_common_df())

    called_names = {call.args[0] for call in mock_mcal.get_calendar.call_args_list}
    assert "XETR" in called_names,  "XETRA should map to mcal key 'XETR'"
    assert "NYSE" in called_names
    assert "XTKS" in called_names, "TSE should map to mcal key 'XTKS'"


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


# ── CryptoFearGreedExtractor ──────────────────────────────────────────────────

def test_crypto_fear_greed_passthrough() -> None:
    df = _make_common_df()
    original = df["crypto_fear_greed"].copy()
    df = CryptoFearGreedExtractor().transform(df)
    pd.testing.assert_series_equal(df["crypto_fear_greed"], original)


def test_crypto_fear_greed_nan_when_column_missing() -> None:
    df = _make_common_df(with_sentiment=False)
    df = CryptoFearGreedExtractor().transform(df)
    assert "crypto_fear_greed" in df.columns
    assert df["crypto_fear_greed"].isna().all()


# ── StockFearGreedExtractor ───────────────────────────────────────────────────

def test_stock_fear_greed_passthrough() -> None:
    df = _make_common_df()
    original = df["stock_fear_greed"].copy()
    df = StockFearGreedExtractor().transform(df)
    pd.testing.assert_series_equal(df["stock_fear_greed"], original)


def test_stock_fear_greed_nan_when_column_missing() -> None:
    df = _make_common_df(with_sentiment=False)
    df = StockFearGreedExtractor().transform(df)
    assert "stock_fear_greed" in df.columns
    assert df["stock_fear_greed"].isna().all()


# ── FedRateExtractor ──────────────────────────────────────────────────────────

def test_fed_rate_passthrough() -> None:
    df = _make_common_df()
    df = FedRateExtractor().transform(df)
    assert "fed_rate" in df.columns
    assert "fed_rate_last_change" in df.columns
    assert (df["fed_rate"] == 4.33).all()
    assert (df["fed_rate_last_change"] == -0.25).all()


def test_fed_rate_nan_when_column_missing() -> None:
    df = _make_common_df(with_fed=False)
    df = FedRateExtractor().transform(df)
    assert df["fed_rate"].isna().all()
    assert df["fed_rate_last_change"].isna().all()


def test_new_sentiment_features_in_build_matrix() -> None:
    """Sentiment and Fed features flow through build_feature_matrix without error."""
    matrix = build_feature_matrix(
        _make_common_df(),
        ["crypto_fear_greed", "stock_fear_greed", "fed_rate", "fed_rate_last_change"],
    )
    for col in ("crypto_fear_greed", "stock_fear_greed", "fed_rate", "fed_rate_last_change"):
        assert col in matrix.columns
    assert not matrix.isnull().any().any()


# ── FundingRateExtractor ──────────────────────────────────────────────────────

def test_funding_rate_passthrough() -> None:
    df = _make_common_df()
    original = df["funding_rate_8h"].copy()
    df = FundingRateExtractor().transform(df)
    pd.testing.assert_series_equal(df["funding_rate_8h"], original)


def test_funding_rate_ema24h_computed() -> None:
    df = FundingRateExtractor().transform(_make_common_df())
    assert "funding_rate_ema24h" in df.columns
    assert not df["funding_rate_ema24h"].isna().all()
    # EMA must stay bounded within the range of the raw input
    lo, hi = df["funding_rate_8h"].min(), df["funding_rate_8h"].max()
    assert df["funding_rate_ema24h"].dropna().between(lo * 2, hi * 2).all()


def test_funding_rate_nan_when_column_missing() -> None:
    df = _make_common_df(with_funding=False)
    df = FundingRateExtractor().transform(df)
    assert "funding_rate_8h" in df.columns
    assert df["funding_rate_8h"].isna().all()
    assert "funding_rate_ema24h" in df.columns


def test_funding_rate_in_build_matrix() -> None:
    matrix = build_feature_matrix(_make_common_df(), ["funding_rate_8h", "funding_rate_ema24h"])
    assert "funding_rate_8h" in matrix.columns
    assert "funding_rate_ema24h" in matrix.columns
    assert not matrix.isnull().any().any()


# ── OIRatioExtractor ──────────────────────────────────────────────────────────

def test_oi_ratio_passthrough() -> None:
    df = _make_common_df()
    original = df["SOL_OI_BTC_ratio"].copy()
    df = OIRatioExtractor().transform(df)
    pd.testing.assert_series_equal(df["SOL_OI_BTC_ratio"], original)


def test_oi_ratio_nan_when_column_missing() -> None:
    df = _make_common_df(with_oi=False)
    df = OIRatioExtractor().transform(df)
    assert "SOL_OI_BTC_ratio" in df.columns
    assert df["SOL_OI_BTC_ratio"].isna().all()


def test_oi_ratio_in_build_matrix() -> None:
    matrix = build_feature_matrix(_make_common_df(), ["SOL_OI_BTC_ratio"])
    assert "SOL_OI_BTC_ratio" in matrix.columns
    assert not matrix["SOL_OI_BTC_ratio"].isna().any()


# ── IVSkewExtractor ───────────────────────────────────────────────────────────

def test_iv_skew_passthrough() -> None:
    df = _make_common_df()
    original = df["btc_iv_skew"].copy()
    df = IVSkewExtractor().transform(df)
    pd.testing.assert_series_equal(df["btc_iv_skew"], original)


def test_iv_skew_nan_when_column_missing() -> None:
    df = _make_common_df(with_iv_skew=False)
    df = IVSkewExtractor().transform(df)
    assert "btc_iv_skew" in df.columns
    assert df["btc_iv_skew"].isna().all()


def test_iv_skew_in_build_matrix() -> None:
    matrix = build_feature_matrix(_make_common_df(), ["btc_iv_skew"])
    assert "btc_iv_skew" in matrix.columns
    assert not matrix["btc_iv_skew"].isna().any()


# ── build_feature_matrix duplicate-column guard ───────────────────────────────

def test_build_feature_matrix_raises_on_duplicate_columns() -> None:
    """build_feature_matrix must raise when the df contains duplicate column names."""
    from src.hmm.features import FeatureExtractor, build_feature_matrix

    class _DupExtractor(FeatureExtractor):
        @property
        def feature_names(self) -> list[str]:
            return ["extra_dup"]

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            # pd.concat is the only way to inject real duplicate column names
            extra = pd.DataFrame({"extra_dup": 1.0}, index=df.index)
            return pd.concat([df, extra, extra], axis=1)

    import src.hmm.features as feat_mod
    original_extractors = list(feat_mod.ALL_EXTRACTORS)
    feat_mod.ALL_EXTRACTORS.append(_DupExtractor())
    try:
        with pytest.raises(ValueError, match="Duplicate columns"):
            build_feature_matrix(_make_common_df(), ["extra_dup"])
    finally:
        feat_mod.ALL_EXTRACTORS[:] = original_extractors
