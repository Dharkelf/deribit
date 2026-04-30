"""Unit tests for DeribitClient — API calls are mocked."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collector.deribit_client import DeribitClient


def _make_response(ticks: list[int]) -> dict:
    n = len(ticks)
    return {
        "jsonrpc": "2.0",
        "result": {
            "status": "ok",
            "ticks": ticks,
            "open":   [100.0] * n,
            "high":   [110.0] * n,
            "low":    [90.0]  * n,
            "close":  [105.0] * n,
            "volume": [1000.0] * n,
        },
    }


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


@patch("src.collector.deribit_client.httpx.Client")
def test_fetch_ohlcv_returns_dataframe(mock_client_cls: MagicMock) -> None:
    ticks = [1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000]
    mock_response = MagicMock()
    mock_response.json.return_value = _make_response(ticks)
    mock_response.raise_for_status = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client_cls.return_value)
    mock_client_cls.return_value.get.return_value = mock_response

    client = DeribitClient(resolution_minutes=60)
    client._http = mock_client_cls.return_value

    df = client.fetch_ohlcv("BTC-PERPETUAL", _utc(2023, 11, 14), _utc(2023, 11, 15))

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 3
    assert df.index.tz is not None  # UTC-aware


@patch("src.collector.deribit_client.httpx.Client")
def test_fetch_ohlcv_empty_response(mock_client_cls: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {"status": "ok", "ticks": []},
    }
    mock_response.raise_for_status = MagicMock()
    mock_client_cls.return_value.get.return_value = mock_response

    client = DeribitClient(resolution_minutes=60)
    client._http = mock_client_cls.return_value

    df = client.fetch_ohlcv("BTC-PERPETUAL", _utc(2023, 11, 14), _utc(2023, 11, 15))

    assert df.empty


@patch("src.collector.deribit_client.httpx.Client")
def test_fetch_ohlcv_deduplicates(mock_client_cls: MagicMock) -> None:
    ticks = [1_700_000_000_000, 1_700_000_000_000, 1_700_003_600_000]
    mock_response = MagicMock()
    mock_response.json.return_value = _make_response(ticks)
    mock_response.raise_for_status = MagicMock()
    mock_client_cls.return_value.get.return_value = mock_response

    client = DeribitClient(resolution_minutes=60)
    client._http = mock_client_cls.return_value

    df = client.fetch_ohlcv("BTC-PERPETUAL", _utc(2023, 11, 14), _utc(2023, 11, 15))

    assert df.index.is_unique
