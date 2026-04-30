"""Unit tests for FemaClient — HTTP calls are mocked."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collector.fema_client import FemaClient


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _make_api_response(n_records: int = 5) -> dict:
    records = [
        {
            "disasterNumber": i,
            "declarationDate": f"2024-0{(i % 9) + 1}-01T00:00:00Z",
            "incidentEndDate": None,
        }
        for i in range(1, n_records + 1)
    ]
    return {"DisasterDeclarations": records}


@patch("src.collector.fema_client.httpx.Client")
def test_fetch_daily_score_returns_dataframe(mock_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_api_response(5)
    mock_resp.raise_for_status = MagicMock()
    mock_cls.return_value.get.return_value = mock_resp

    client = FemaClient()
    client._http = mock_cls.return_value
    result = client.fetch_daily_score(_utc(2024, 3, 1), _utc(2024, 3, 31))

    assert "FEMA_score" in result.columns
    assert len(result) == 31
    assert (result["FEMA_score"] >= 0.0).all()
    assert (result["FEMA_score"] <= 1.0).all()


@patch("src.collector.fema_client.httpx.Client")
def test_fetch_daily_score_returns_zeros_on_empty(mock_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"DisasterDeclarations": []}
    mock_resp.raise_for_status = MagicMock()
    mock_cls.return_value.get.return_value = mock_resp

    client = FemaClient()
    client._http = mock_cls.return_value
    result = client.fetch_daily_score(_utc(2024, 3, 1), _utc(2024, 3, 7))

    assert (result["FEMA_score"] == 0.0).all()


@patch("src.collector.fema_client.httpx.Client")
def test_fetch_daily_score_returns_zeros_on_api_error(mock_cls: MagicMock) -> None:
    mock_cls.return_value.get.side_effect = Exception("timeout")

    client = FemaClient()
    client._http = mock_cls.return_value
    result = client.fetch_daily_score(_utc(2024, 3, 1), _utc(2024, 3, 7))

    assert (result["FEMA_score"] == 0.0).all()
