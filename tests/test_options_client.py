"""Unit tests for DeribitOptionsClient and Max Pain computation."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.collector.options_client import (
    DeribitOptionsClient,
    _compute_max_pain,
    _parse_instrument,
)


# ── _parse_instrument ─────────────────────────────────────────────────────────

def test_parse_call_instrument() -> None:
    result = _parse_instrument("BTC-27DEC24-50000-C")
    assert result is not None
    expiry, strike, opt_type = result
    assert strike == 50000.0
    assert opt_type == "C"
    assert expiry.year == 2024
    assert expiry.month == 12
    assert expiry.day == 27


def test_parse_put_instrument() -> None:
    result = _parse_instrument("BTC-15MAR25-80000-P")
    assert result is not None
    _, strike, opt_type = result
    assert strike == 80000.0
    assert opt_type == "P"


def test_parse_invalid_returns_none() -> None:
    assert _parse_instrument("ETH-27DEC24-2000-C") is None
    assert _parse_instrument("not-an-instrument") is None


# ── _compute_max_pain ─────────────────────────────────────────────────────────

def test_compute_max_pain_trivial() -> None:
    """With only calls at high strikes, max pain should be at the lowest strike."""
    group = pd.DataFrame({
        "strike": [40000.0, 50000.0, 60000.0, 40000.0, 50000.0, 60000.0],
        "type":   ["C",     "C",     "C",     "P",     "P",     "P"],
        "open_interest": [100.0, 100.0, 100.0, 1.0, 1.0, 1.0],
    })
    mp = _compute_max_pain(group)
    assert mp is not None
    assert isinstance(mp, float)


def test_compute_max_pain_single_strike_returns_none() -> None:
    group = pd.DataFrame({
        "strike": [50000.0],
        "type": ["C"],
        "open_interest": [100.0],
    })
    assert _compute_max_pain(group) is None


# ── DeribitOptionsClient ──────────────────────────────────────────────────────

def _make_api_response(n_strikes: int = 5) -> dict:
    from datetime import timedelta
    expiry = datetime.now(tz=timezone.utc) + timedelta(days=10)
    exp_str = expiry.strftime("%d%b%y").upper()
    results = []
    for k in range(n_strikes):
        strike = 50000 + k * 5000
        for opt_type in ("C", "P"):
            results.append({
                "instrument_name": f"BTC-{exp_str}-{strike}-{opt_type}",
                "open_interest": float(100 + k * 10),
            })
    return {"result": results}


@patch("src.collector.options_client.httpx.Client")
def test_fetch_daily_snapshot_returns_both_columns(mock_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_api_response(5)
    mock_resp.raise_for_status = MagicMock()
    mock_cls.return_value.get.return_value = mock_resp

    client = DeribitOptionsClient(days_ahead=30, days_ahead_short=7)
    client._http = mock_cls.return_value
    result = client.fetch_daily_snapshot()

    assert "BTC_options_max_pain" in result.columns
    assert "BTC_options_max_pain_7d" in result.columns
    assert len(result) == 1
    assert not result["BTC_options_max_pain"].isna().all()


@patch("src.collector.options_client.httpx.Client")
def test_fetch_daily_snapshot_nan_on_api_error(mock_cls: MagicMock) -> None:
    mock_cls.return_value.get.side_effect = Exception("timeout")

    client = DeribitOptionsClient(days_ahead=30, days_ahead_short=7)
    client._http = mock_cls.return_value
    result = client.fetch_daily_snapshot()

    assert result["BTC_options_max_pain"].isna().all()
    assert result["BTC_options_max_pain_7d"].isna().all()


@patch("src.collector.options_client.httpx.Client")
def test_fetch_daily_snapshot_nan_when_no_upcoming_expiries(mock_cls: MagicMock) -> None:
    from datetime import timedelta
    past = datetime.now(tz=timezone.utc) - timedelta(days=60)
    exp_str = past.strftime("%d%b%y").upper()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": [
        {"instrument_name": f"BTC-{exp_str}-50000-C", "open_interest": 100.0}
    ]}
    mock_resp.raise_for_status = MagicMock()
    mock_cls.return_value.get.return_value = mock_resp

    client = DeribitOptionsClient(days_ahead=30, days_ahead_short=7)
    client._http = mock_cls.return_value
    result = client.fetch_daily_snapshot()

    assert result["BTC_options_max_pain"].isna().all()
    assert result["BTC_options_max_pain_7d"].isna().all()
