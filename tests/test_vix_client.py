"""Unit tests for VixClient — yfinance is mocked."""

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from src.collector.vix_client import VixClient


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _make_daily_df(dates: list[str]) -> pd.DataFrame:
    index = pd.to_datetime(dates, utc=True)
    return pd.DataFrame(
        {"Open": 20.0, "High": 22.0, "Low": 19.0, "Close": 21.0},
        index=index,
    )


@patch("src.collector.vix_client.yf.download")
def test_fetch_returns_hourly_dataframe(mock_download: object) -> None:
    mock_download.return_value = _make_daily_df(["2024-01-02", "2024-01-03"])
    client = VixClient()
    df = client.fetch_ohlcv(_utc(2024, 1, 2), _utc(2024, 1, 4))

    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None
    assert df.index.name == "timestamp"  # consistent index name for common DataFrame
    # daily → hourly: each day yields 24 rows after ffill
    assert len(df) >= 24


@patch("src.collector.vix_client.yf.download")
def test_fetch_empty_response(mock_download: object) -> None:
    mock_download.return_value = pd.DataFrame()
    client = VixClient()
    df = client.fetch_ohlcv(_utc(2024, 1, 2), _utc(2024, 1, 3))

    assert df.empty


@patch("src.collector.vix_client.yf.download")
def test_volume_is_zero(mock_download: object) -> None:
    mock_download.return_value = _make_daily_df(["2024-01-02"])
    client = VixClient()
    df = client.fetch_ohlcv(_utc(2024, 1, 2), _utc(2024, 1, 3))

    assert (df["volume"] == 0.0).all()
