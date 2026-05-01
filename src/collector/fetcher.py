"""Fetcher — orchestrates all data clients and ParquetRepository.

For each symbol: checks the last stored timestamp, fetches only missing
data (incremental update), and appends to Parquet.

Sources:
  - Deribit REST API  → BTC, ETH, SOL (OHLCV, hourly)
  - yfinance          → VIX (daily → forward-filled hourly)
  - FEMA OpenFEMA     → US disaster severity score (daily)
  - GDELT DOC 2.0     → US military activity score (daily)
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.collector.deribit_client import DeribitClient
from src.collector.fema_client import FemaClient
from src.collector.gdelt_client import GdeltClient
from src.collector.options_client import DeribitOptionsClient
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
    _fetch_fema(repo, history_days)
    _fetch_gdelt(repo, history_days)
    _fetch_options_max_pain(config, repo)


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
            logger.info(
                "Fetching %s (%s) from %s to %s",
                symbol, instrument, start.date(), end.date(),
            )
            df = client.fetch_ohlcv(instrument, start, end)
            repo.append(symbol, df)
            repo.save_sample(symbol)


def _fetch_vix(
    config: dict,
    repo: ParquetRepository,
    history_days: int,
) -> None:
    symbol = "VIX"
    ticker: str = config["symbols"]["vix"]
    start, end = _time_range(repo, symbol, history_days)
    logger.info("Fetching VIX (%s) from %s to %s", ticker, start.date(), end.date())
    df = VixClient(ticker=ticker).fetch_ohlcv(start, end)
    repo.append(symbol, df)
    repo.save_sample(symbol)


def _fetch_fema(repo: ParquetRepository, history_days: int) -> None:
    symbol = "FEMA"
    start, end = _time_range(repo, symbol, history_days)
    logger.info("Fetching FEMA disaster score from %s to %s", start.date(), end.date())
    with FemaClient() as client:
        df = client.fetch_daily_score(start, end)
    repo.append(symbol, df)
    repo.save_sample(symbol)


def _fetch_gdelt(repo: ParquetRepository, history_days: int) -> None:
    symbol = "GDELT"
    start, end = _time_range(repo, symbol, history_days)
    logger.info("Fetching GDELT military score from %s to %s", start.date(), end.date())
    with GdeltClient() as client:
        df = client.fetch_daily_score(start, end)
    if df is None:
        logger.warning("GDELT fetch failed — skipping persist to avoid overwriting with zeros")
        return
    repo.append(symbol, df)
    repo.save_sample(symbol)


def _fetch_options_max_pain(config: dict, repo: ParquetRepository) -> None:
    symbol = "BTC_OPTIONS_MAX_PAIN"
    days_ahead: int = config.get("options", {}).get("max_pain_days_ahead", 30)
    last = repo.last_timestamp(symbol)
    today = datetime.now(tz=timezone.utc).date()
    if last is not None and last.date() >= today:
        logger.info("Options max pain already up to date")
        return
    logger.info("Fetching BTC options max pain (next %d days)", days_ahead)
    with DeribitOptionsClient(days_ahead=days_ahead) as client:
        df = client.fetch_daily_snapshot()
    repo.append(symbol, df)
    repo.save_sample(symbol)


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
