"""Unit tests for fear_greed_client.py — no network access required."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collector.fear_greed_client import fetch_crypto_fear_greed, fetch_stock_fear_greed


# ── helpers ───────────────────────────────────────────────────────────────────


def _mock_response(json_data: object, status_code: int = 200) -> MagicMock:
    m = MagicMock()
    m.status_code = status_code
    m.raise_for_status.return_value = None
    m.json.return_value = json_data
    return m


def _crypto_payload(n: int = 5) -> dict:
    """Fake alternative.me response with n daily records."""
    import time
    base = int(time.mktime((2024, 1, 1, 0, 0, 0, 0, 0, 0)))
    return {
        "data": [
            {"timestamp": str(base + i * 86400), "value": str(50 + i)}
            for i in range(n)
        ]
    }


def _stock_payload(n: int = 5) -> dict:
    """Fake CNN response with n daily records."""
    import time
    base = int(time.mktime((2024, 1, 1, 0, 0, 0, 0, 0, 0))) * 1000  # ms
    return {
        "fear_and_greed_historical": {
            "data": [
                {"x": base + i * 86_400_000, "y": 40.0 + i}
                for i in range(n)
            ]
        }
    }


# ── fetch_crypto_fear_greed ───────────────────────────────────────────────────


def test_crypto_fg_returns_dataframe() -> None:
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(_crypto_payload(10))
        df = fetch_crypto_fear_greed(days=10)

    assert isinstance(df, pd.DataFrame)
    assert "crypto_fear_greed" in df.columns
    assert len(df) == 10


def test_crypto_fg_values_normalised_0_to_1() -> None:
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(_crypto_payload(5))
        df = fetch_crypto_fear_greed()

    assert (df["crypto_fear_greed"] >= 0.0).all()
    assert (df["crypto_fear_greed"] <= 1.0).all()


def test_crypto_fg_index_is_utc() -> None:
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(_crypto_payload(3))
        df = fetch_crypto_fear_greed()

    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"


def test_crypto_fg_network_error_returns_none() -> None:
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.side_effect = Exception("connection refused")
        result = fetch_crypto_fear_greed()

    assert result is None


def test_crypto_fg_empty_data_returns_none() -> None:
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response({"data": []})
        result = fetch_crypto_fear_greed()

    assert result is None


def test_crypto_fg_malformed_items_skipped() -> None:
    payload = {
        "data": [
            {"timestamp": "1704067200", "value": "55"},   # valid
            {"timestamp": "bad", "value": "60"},           # bad ts — skipped
            {"timestamp": "1704153600", "value": "xyz"},   # bad value — skipped
            {"timestamp": "1704240000", "value": "70"},    # valid
        ]
    }
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(payload)
        df = fetch_crypto_fear_greed()

    assert df is not None
    assert len(df) == 2


def test_crypto_fg_sorted_ascending() -> None:
    import time
    base = int(time.mktime((2024, 1, 1, 0, 0, 0, 0, 0, 0)))
    payload = {
        "data": [
            {"timestamp": str(base + 2 * 86400), "value": "60"},
            {"timestamp": str(base), "value": "50"},
            {"timestamp": str(base + 86400), "value": "55"},
        ]
    }
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(payload)
        df = fetch_crypto_fear_greed()

    assert df is not None
    assert list(df.index) == sorted(df.index)


# ── fetch_stock_fear_greed ────────────────────────────────────────────────────


def test_stock_fg_returns_dataframe() -> None:
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(_stock_payload(8))
        df = fetch_stock_fear_greed()

    assert isinstance(df, pd.DataFrame)
    assert "stock_fear_greed" in df.columns
    assert len(df) == 8


def test_stock_fg_values_normalised_0_to_1() -> None:
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(_stock_payload(5))
        df = fetch_stock_fear_greed()

    assert df is not None
    assert (df["stock_fear_greed"] >= 0.0).all()
    assert (df["stock_fear_greed"] <= 1.0).all()


def test_stock_fg_index_is_utc() -> None:
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(_stock_payload(3))
        df = fetch_stock_fear_greed()

    assert df is not None
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"


def test_stock_fg_network_error_returns_none() -> None:
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.side_effect = Exception("timeout")
        result = fetch_stock_fear_greed()

    assert result is None


def test_stock_fg_unexpected_json_structure_returns_none() -> None:
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response({"wrong_key": []})
        result = fetch_stock_fear_greed()

    assert result is None


def test_stock_fg_empty_history_returns_none() -> None:
    payload = {"fear_and_greed_historical": {"data": []}}
    with patch("src.collector.fear_greed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(payload)
        result = fetch_stock_fear_greed()

    assert result is None
