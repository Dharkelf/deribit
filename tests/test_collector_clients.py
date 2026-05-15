"""Unit tests for DeribitFundingRateClient, DeribitOIClient, DeribitIVSkewClient.

All HTTP calls are mocked so no network access is required.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.collector.funding_client import DeribitFundingRateClient, DeribitOIClient
from src.collector.iv_skew_client import DeribitIVSkewClient, _parse_option


# ── _parse_option ─────────────────────────────────────────────────────────────


def test_parse_option_valid_call() -> None:
    result = _parse_option("BTC-27DEC24-50000-C")
    assert result is not None
    expiry, strike, opt_type = result
    assert opt_type == "C"
    assert strike == 50000.0
    assert expiry.year == 2024
    assert expiry.month == 12
    assert expiry.day == 27


def test_parse_option_valid_put() -> None:
    result = _parse_option("BTC-15JAN25-30000-P")
    assert result is not None
    _, strike, opt_type = result
    assert opt_type == "P"
    assert strike == 30000.0


def test_parse_option_invalid_returns_none() -> None:
    assert _parse_option("not-an-option") is None
    assert _parse_option("") is None
    assert _parse_option("BTC-27XXX24-50000-C") is None


# ── DeribitFundingRateClient ──────────────────────────────────────────────────


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _make_funding_response(n: int = 3) -> dict:
    """Fake Deribit funding rate API response with `n` 8h records."""
    base_ms = int(_utc(2024, 1, 1).timestamp() * 1000)
    records = [
        {"timestamp": base_ms + i * 8 * 3600 * 1000, "interest_8h": 0.0001 * (i + 1)}
        for i in range(n)
    ]
    return {"result": records}


def _mock_get_side_effects(*response_dicts: dict) -> list[MagicMock]:
    """Build a list of mock responses: each call returns the next response."""
    mocks = []
    for d in response_dicts:
        m = MagicMock()
        m.raise_for_status.return_value = None
        m.json.return_value = d
        mocks.append(m)
    return mocks


def test_funding_rate_fetch_hourly_shape() -> None:
    # First call returns 5 records; second call returns empty → loop exits.
    with patch("src.collector.funding_client.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.side_effect = _mock_get_side_effects(
            _make_funding_response(5),
            {"result": []},
        )
        client = DeribitFundingRateClient()
        client._http = instance
        df = client.fetch_hourly(start=_utc(2024, 1, 1), end=_utc(2024, 1, 3))

    assert not df.empty
    assert "funding_rate_8h" in df.columns
    assert len(df) >= 5
    assert df.index.tz is not None


def test_funding_rate_empty_response_returns_empty_df() -> None:
    with patch("src.collector.funding_client.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.side_effect = _mock_get_side_effects({"result": []})
        client = DeribitFundingRateClient()
        client._http = instance
        df = client.fetch_hourly(start=_utc(2024, 1, 1), end=_utc(2024, 1, 2))

    assert df.empty
    assert "funding_rate_8h" in df.columns


def test_funding_rate_no_duplicates_in_hourly_output() -> None:
    with patch("src.collector.funding_client.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.side_effect = _mock_get_side_effects(
            _make_funding_response(10),
            {"result": []},
        )
        client = DeribitFundingRateClient()
        client._http = instance
        df = client.fetch_hourly(start=_utc(2024, 1, 1), end=_utc(2024, 1, 5))

    assert not df.index.duplicated().any()


# ── DeribitOIClient ───────────────────────────────────────────────────────────


def _make_oi_response(currency: str) -> dict:
    if currency == "USDC":
        return {
            "result": [
                {
                    "instrument_name": "SOL_USDC-PERPETUAL",
                    "open_interest": 1_000_000.0,
                    "underlying_price": 150.0,
                    "mark_price": 150.0,
                }
            ]
        }
    else:  # BTC
        return {
            "result": [
                {
                    "instrument_name": "BTC-PERPETUAL",
                    "open_interest": 5_000_000_000.0,
                    "underlying_price": 60000.0,
                    "mark_price": 60000.0,
                }
            ]
        }


def test_oi_snapshot_returns_ratio() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None

    def _side_effect(path: str, params: dict) -> MagicMock:
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = _make_oi_response(params.get("currency", "BTC"))
        return r

    with patch("src.collector.funding_client.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.side_effect = _side_effect

        client = DeribitOIClient()
        client._http = instance
        df = client.fetch_daily_snapshot()

    assert "SOL_OI_BTC_ratio" in df.columns
    assert "SOL_oi_usd" in df.columns
    assert "BTC_oi_usd" in df.columns
    assert len(df) == 1
    ratio = float(df["SOL_OI_BTC_ratio"].iloc[0])
    assert ratio > 0


def test_oi_snapshot_api_error_returns_nan() -> None:
    with patch("src.collector.funding_client.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.side_effect = Exception("API error")

        client = DeribitOIClient()
        client._http = instance
        df = client.fetch_daily_snapshot()

    assert len(df) == 1
    assert np.isnan(float(df["SOL_OI_BTC_ratio"].iloc[0]))


# ── DeribitIVSkewClient ───────────────────────────────────────────────────────


def _make_options_response() -> dict:
    """Fake options chain with ATM BTC calls and puts expiring in 7 days."""
    from datetime import timedelta

    today = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    expiry = today + timedelta(days=7)
    day_str = f"{expiry.day:02d}"
    mon_str = expiry.strftime("%b").upper()
    yr_str = str(expiry.year)[2:]
    strike = 60000

    instruments = []
    for opt_type in ("C", "P"):
        iv = 40.0 if opt_type == "C" else 45.0  # put IV > call IV → positive skew
        instruments.append({
            "instrument_name": f"BTC-{day_str}{mon_str}{yr_str}-{strike}-{opt_type}",
            "mark_iv": iv,
            "underlying_price": 60000.0,
        })
    return {"result": instruments}


def test_iv_skew_positive_for_puts_more_expensive() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = _make_options_response()

    with patch("src.collector.iv_skew_client.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.return_value = mock_resp

        client = DeribitIVSkewClient()
        client._http = instance
        df = client.fetch_daily_snapshot()

    assert "btc_iv_skew" in df.columns
    assert len(df) == 1
    skew = float(df["btc_iv_skew"].iloc[0])
    assert skew > 0  # put_iv(45) > call_iv(40)


def test_iv_skew_api_error_returns_nan() -> None:
    with patch("src.collector.iv_skew_client.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.side_effect = Exception("timeout")

        client = DeribitIVSkewClient()
        client._http = instance
        df = client.fetch_daily_snapshot()

    assert len(df) == 1
    assert np.isnan(float(df["btc_iv_skew"].iloc[0]))


def test_iv_skew_empty_chain_returns_nan() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"result": []}

    with patch("src.collector.iv_skew_client.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.get.return_value = mock_resp

        client = DeribitIVSkewClient()
        client._http = instance
        df = client.fetch_daily_snapshot()

    assert np.isnan(float(df["btc_iv_skew"].iloc[0]))
