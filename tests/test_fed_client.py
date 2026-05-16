"""Unit tests for fed_client.py — no network access required."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collector.fed_client import fetch_fed_rate


# ── helpers ───────────────────────────────────────────────────────────────────


def _mock_response(text: str, status_code: int = 200) -> MagicMock:
    m = MagicMock()
    m.status_code = status_code
    m.raise_for_status.return_value = None
    m.text = text
    return m


def _fred_csv(rates: list[tuple[str, float]]) -> str:
    """Build a minimal FRED DFF CSV string."""
    lines = ["observation_date,DFF"]
    for date, rate in rates:
        lines.append(f"{date},{rate}")
    return "\n".join(lines)


# ── fetch_fed_rate ────────────────────────────────────────────────────────────


def test_fed_rate_returns_dataframe() -> None:
    csv = _fred_csv([
        ("2024-01-01", 5.33),
        ("2024-01-02", 5.33),
        ("2024-01-03", 5.33),
    ])
    with patch("src.collector.fed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(csv)
        df = fetch_fed_rate()

    assert isinstance(df, pd.DataFrame)
    assert "fed_rate" in df.columns
    assert "fed_rate_last_change" in df.columns
    assert len(df) == 3


def test_fed_rate_index_is_utc() -> None:
    csv = _fred_csv([("2024-01-01", 5.33), ("2024-01-02", 5.33)])
    with patch("src.collector.fed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(csv)
        df = fetch_fed_rate()

    assert df is not None
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"


def test_fed_rate_last_change_detects_cut() -> None:
    """A 0.25pp cut should be recorded as -0.25 and forward-filled."""
    csv = _fred_csv([
        ("2024-09-16", 5.33),
        ("2024-09-17", 5.33),
        ("2024-09-18", 5.08),  # -0.25pp cut
        ("2024-09-19", 5.08),
        ("2024-09-20", 5.08),
    ])
    with patch("src.collector.fed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(csv)
        df = fetch_fed_rate()

    assert df is not None
    last_change = df["fed_rate_last_change"].iloc[-1]
    assert abs(last_change - (-0.25)) < 1e-6


def test_fed_rate_noise_below_threshold_ignored() -> None:
    """Daily rate fluctuations <0.10pp must not be recorded as policy moves."""
    csv = _fred_csv([
        ("2024-01-01", 5.33),
        ("2024-01-02", 5.34),   # +0.01 noise
        ("2024-01-03", 5.33),   # -0.01 noise
        ("2024-01-04", 5.33),
    ])
    with patch("src.collector.fed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(csv)
        df = fetch_fed_rate()

    assert df is not None
    # No real FOMC move → all last_change values should be 0.0 (fillna default)
    assert (df["fed_rate_last_change"] == 0.0).all()


def test_fed_rate_network_error_returns_none() -> None:
    with patch("src.collector.fed_client.requests.get") as mock_get:
        mock_get.side_effect = Exception("timeout")
        result = fetch_fed_rate()

    assert result is None


def test_fed_rate_empty_csv_returns_none() -> None:
    """A CSV with only the header and no data rows should return None."""
    csv = "observation_date,DFF\n"
    with patch("src.collector.fed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(csv)
        result = fetch_fed_rate()

    assert result is None


def test_fed_rate_all_na_returns_none() -> None:
    """FRED uses '.' for missing values; all-missing should return None."""
    csv = _fred_csv([("2024-01-01", "."), ("2024-01-02", ".")])
    with patch("src.collector.fed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(csv)
        result = fetch_fed_rate()

    assert result is None


def test_fed_rate_hike_recorded_positive() -> None:
    csv = _fred_csv([
        ("2022-03-16", 0.08),
        ("2022-03-17", 0.33),  # +0.25pp hike
        ("2022-03-18", 0.33),
    ])
    with patch("src.collector.fed_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response(csv)
        df = fetch_fed_rate()

    assert df is not None
    last_change = df["fed_rate_last_change"].iloc[-1]
    assert last_change > 0.0
