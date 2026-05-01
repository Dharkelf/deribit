"""Unit tests for GdeltClient — HTTP calls are mocked."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.collector.gdelt_client import GdeltClient


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _make_gdelt_response(n_days: int = 10) -> dict:
    from datetime import timedelta
    base = datetime(2024, 3, 1)
    return {
        "timeline": [
            {
                "data": [
                    {
                        "date": (base + timedelta(days=i)).strftime("%Y%m%d%H%M%S"),
                        "value": float(100 + i * 10),
                    }
                    for i in range(n_days)
                ]
            }
        ]
    }


@patch("src.collector.gdelt_client.httpx.Client")
def test_fetch_daily_score_returns_dataframe(mock_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_gdelt_response(10)
    mock_resp.raise_for_status = MagicMock()
    mock_cls.return_value.get.return_value = mock_resp

    client = GdeltClient()
    client._http = mock_cls.return_value
    result = client.fetch_daily_score(_utc(2024, 3, 1), _utc(2024, 3, 31))

    assert "GDELT_military_score" in result.columns
    assert (result["GDELT_military_score"] >= 0.0).all()
    assert (result["GDELT_military_score"] <= 1.0).all()


@patch("src.collector.gdelt_client.httpx.Client")
def test_fetch_daily_score_zeros_on_api_error(mock_cls: MagicMock) -> None:
    mock_cls.return_value.get.side_effect = Exception("connection refused")

    client = GdeltClient()
    client._http = mock_cls.return_value
    result = client.fetch_daily_score(_utc(2024, 3, 1), _utc(2024, 3, 7))

    assert (result["GDELT_military_score"] == 0.0).all()


@patch("src.collector.gdelt_client.httpx.Client")
def test_fetch_uses_correct_endpoint(mock_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_gdelt_response(5)
    mock_resp.raise_for_status = MagicMock()
    mock_cls.return_value.get.return_value = mock_resp

    client = GdeltClient()
    client._http = mock_cls.return_value
    client.fetch_daily_score(_utc(2024, 3, 1), _utc(2024, 3, 7))

    called_path = mock_cls.return_value.get.call_args[0][0]
    assert called_path == "/api/v2/doc/doc", f"wrong endpoint: {called_path}"


@patch("src.collector.gdelt_client.httpx.Client")
def test_fetch_daily_score_zeros_on_empty_timeline(mock_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"timeline": []}
    mock_resp.raise_for_status = MagicMock()
    mock_cls.return_value.get.return_value = mock_resp

    client = GdeltClient()
    client._http = mock_cls.return_value
    result = client.fetch_daily_score(_utc(2024, 3, 1), _utc(2024, 3, 7))

    assert (result["GDELT_military_score"] == 0.0).all()
