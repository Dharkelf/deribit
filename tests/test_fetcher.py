"""Unit tests for src/collector/fetcher.py — all I/O is mocked."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collector.fetcher import _time_range, _fetch_funding_rate, _fetch_oi_snapshot


# ── _time_range ───────────────────────────────────────────────────────────────


def _make_repo(last_ts: pd.Timestamp | None = None) -> MagicMock:
    repo = MagicMock()
    repo.last_timestamp.return_value = last_ts
    return repo


def test_time_range_fresh_uses_history_days() -> None:
    repo = _make_repo(last_ts=None)
    start, end = _time_range(repo, "BTC", history_days=365)
    delta = end - start
    assert 364 <= delta.days <= 366


def test_time_range_incremental_advances_by_one_hour() -> None:
    last = pd.Timestamp("2024-06-01 12:00", tz="UTC")
    repo = _make_repo(last_ts=last)
    start, end = _time_range(repo, "BTC", history_days=365)
    assert start == last.to_pydatetime() + timedelta(hours=1)
    assert start.tzinfo is not None


def test_time_range_end_is_utc_and_truncated() -> None:
    repo = _make_repo(last_ts=None)
    _, end = _time_range(repo, "SOL", history_days=30)
    assert end.tzinfo == timezone.utc
    assert end.minute == 0
    assert end.second == 0
    assert end.microsecond == 0


# ── _fetch_funding_rate ───────────────────────────────────────────────────────


def _make_funding_df(n: int = 5) -> pd.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    idx = pd.date_range(base, periods=n, freq="1h", tz="UTC", name="timestamp")
    return pd.DataFrame({"funding_rate_8h": [0.0001] * n}, index=idx)


def test_fetch_funding_rate_appends_on_success() -> None:
    repo = _make_repo(last_ts=None)

    with patch("src.collector.fetcher.DeribitFundingRateClient") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.fetch_hourly.return_value = _make_funding_df(5)
        _fetch_funding_rate(repo, history_days=30)

    repo.append.assert_called_once()
    repo.save_sample.assert_called_once()


def test_fetch_funding_rate_skips_on_exception() -> None:
    repo = _make_repo(last_ts=None)

    with patch("src.collector.fetcher.DeribitFundingRateClient") as MockClient:
        MockClient.return_value.__enter__.side_effect = Exception("API down")
        _fetch_funding_rate(repo, history_days=30)

    repo.append.assert_not_called()


def test_fetch_funding_rate_skips_empty_df() -> None:
    repo = _make_repo(last_ts=None)

    with patch("src.collector.fetcher.DeribitFundingRateClient") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.fetch_hourly.return_value = pd.DataFrame(columns=["funding_rate_8h"])
        _fetch_funding_rate(repo, history_days=30)

    repo.append.assert_not_called()


# ── _fetch_oi_snapshot ────────────────────────────────────────────────────────


def _make_oi_df() -> pd.DataFrame:
    today = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    idx = pd.DatetimeIndex([today], name="timestamp")
    return pd.DataFrame(
        {"SOL_oi_usd": [1e9], "BTC_oi_usd": [5e10], "SOL_OI_BTC_ratio": [0.02]},
        index=idx,
    )


def test_fetch_oi_snapshot_skips_when_up_to_date() -> None:
    today = datetime.now(tz=timezone.utc)
    repo = _make_repo(last_ts=pd.Timestamp(today))

    with patch("src.collector.fetcher.DeribitOIClient") as MockClient:
        _fetch_oi_snapshot(repo)
        MockClient.assert_not_called()

    repo.append.assert_not_called()


def test_fetch_oi_snapshot_appends_when_stale() -> None:
    yesterday = pd.Timestamp(datetime.now(tz=timezone.utc) - timedelta(days=1))
    repo = _make_repo(last_ts=yesterday)

    with patch("src.collector.fetcher.DeribitOIClient") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.fetch_daily_snapshot.return_value = _make_oi_df()
        _fetch_oi_snapshot(repo)

    repo.append.assert_called_once()
    repo.save_sample.assert_called_once()
