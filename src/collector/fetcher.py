"""Fetcher — orchestrates DeribitClient, VixClient and ParquetRepository.

For each symbol: checks the last stored timestamp, fetches only missing
candles (incremental update), and appends to Parquet.
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.collector.deribit_client import DeribitClient
from src.collector.repository import ParquetRepository
from src.collector.vix_client import VixClient
from src.utils.paths import raw_dir

logger = logging.getLogger(__name__)


def run(config: dict) -> None:
    """Entry point called from main.py. Fetches all symbols and saves Parquet."""
    repo = ParquetRepository(raw_dir(config))
    history_days = config["collector"]["history_days"]
    resolution = config["collector"]["resolution"]

    _fetch_deribit(config, repo, history_days, resolution)
    _fetch_vix(config, repo, history_days)


def _fetch_deribit(
    config: dict,
    repo: ParquetRepository,
    history_days: int,
    resolution_minutes: int,
) -> None:
    with DeribitClient(resolution_minutes=resolution_minutes) as client:
        for entry in config["symbols"]["deribit"]:
            instrument: str = entry["instrument"]
            symbol: str = entry["symbol"]
            start, end = _time_range(repo, symbol, history_days)
            logger.info("Fetching %s (%s) from %s to %s", symbol, instrument, start.date(), end.date())
            df = client.fetch_ohlcv(instrument, start, end)
            repo.append(symbol, df)


def _fetch_vix(
    config: dict,
    repo: ParquetRepository,
    history_days: int,
) -> None:
    symbol = "VIX"
    ticker: str = config["symbols"]["vix"]
    start, end = _time_range(repo, symbol, history_days)
    logger.info("Fetching VIX (%s) from %s to %s", ticker, start.date(), end.date())
    client = VixClient(ticker=ticker)
    df = client.fetch_ohlcv(start, end)
    repo.append(symbol, df)


def _time_range(
    repo: ParquetRepository,
    symbol: str,
    history_days: int,
) -> tuple[datetime, datetime]:
    """Return (start, end) in UTC. Start is last stored timestamp or history_days ago."""
    end = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    last = repo.last_timestamp(symbol)
    if last is not None:
        start = last.to_pydatetime() + timedelta(hours=1)
    else:
        start = end - timedelta(days=history_days)
    return start, end
