"""Fetcher — orchestrates all data clients and ParquetRepository.

For each symbol: checks the last stored timestamp, fetches only missing
data (incremental update), and appends to Parquet.

Sources:
  - Deribit REST API  → BTC, ETH, SOL (OHLCV, hourly)
  - yfinance          → VIX (daily → forward-filled hourly)
  - FEMA OpenFEMA     → US disaster severity score (daily)
  - GDELT DOC 2.0     → US military activity score (daily)
  - alternative.me    → Crypto Fear & Greed Index (daily, [0,1])
  - CNN dataviz       → Stock Fear & Greed Index (daily, [0,1])
  - FRED DFF          → US Federal Funds Rate + last change (daily)
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.collector.deribit_client import DeribitClient
from src.collector.fema_client import FemaClient
from src.collector.fear_greed_client import fetch_crypto_fear_greed, fetch_stock_fear_greed
from src.collector.fed_client import fetch_fed_rate
from src.collector.funding_client import DeribitFundingRateClient, DeribitOIClient
from src.collector.gdelt_client import GdeltClient
from src.collector.iv_skew_client import DeribitIVSkewClient
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
    _fetch_crypto_fear_greed(repo, history_days)
    _fetch_stock_fear_greed(repo)
    _fetch_fed_rate(repo)
    _fetch_funding_rate(repo, history_days)
    _fetch_oi_snapshot(repo)
    _fetch_iv_skew(config, repo)


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
    opts = config.get("options", {})
    days_ahead: int       = opts.get("max_pain_days_ahead", 30)
    days_ahead_short: int = opts.get("max_pain_days_ahead_short", 7)
    last = repo.last_timestamp(symbol)
    today = datetime.now(tz=timezone.utc).date()
    if last is not None and last.date() >= today:
        logger.info("Options max pain already up to date")
        return
    logger.info(
        "Fetching BTC options max pain (7d and %dd windows)", days_ahead
    )
    with DeribitOptionsClient(days_ahead=days_ahead, days_ahead_short=days_ahead_short) as client:
        df = client.fetch_daily_snapshot()
    repo.append(symbol, df)
    repo.save_sample(symbol)


def _fetch_crypto_fear_greed(repo: ParquetRepository, history_days: int) -> None:
    symbol = "CRYPTO_FEAR_GREED"
    start, _ = _time_range(repo, symbol, history_days)
    days_needed = (datetime.now(tz=timezone.utc) - start).days + 1
    if days_needed <= 0:
        logger.info("Crypto Fear & Greed already up to date")
        return
    logger.info("Fetching Crypto Fear & Greed (%d days)", days_needed)
    df = fetch_crypto_fear_greed(days=days_needed)
    if df is None:
        logger.warning("Crypto F&G fetch failed — skipping persist")
        return
    repo.append(symbol, df)
    repo.save_sample(symbol)


def _fetch_stock_fear_greed(repo: ParquetRepository) -> None:
    symbol = "STOCK_FEAR_GREED"
    last = repo.last_timestamp(symbol)
    today = datetime.now(tz=timezone.utc).date()
    if last is not None and last.date() >= today:
        logger.info("Stock Fear & Greed already up to date")
        return
    logger.info("Fetching Stock Fear & Greed (CNN)")
    df = fetch_stock_fear_greed()
    if df is None:
        logger.warning("Stock F&G fetch failed — skipping persist")
        return
    repo.append(symbol, df)
    repo.save_sample(symbol)


def _fetch_fed_rate(repo: ParquetRepository) -> None:
    symbol = "FED_RATE"
    last = repo.last_timestamp(symbol)
    today = datetime.now(tz=timezone.utc).date()
    if last is not None and last.date() >= today:
        logger.info("Fed rate already up to date")
        return
    logger.info("Fetching US Federal Funds Rate (FRED DFF)")
    df = fetch_fed_rate()
    if df is None:
        logger.warning("Fed rate fetch failed — skipping persist")
        return
    repo.append(symbol, df)
    repo.save_sample(symbol)


def _fetch_funding_rate(repo: ParquetRepository, history_days: int) -> None:
    symbol = "FUNDING_RATE_SOL"
    start, end = _time_range(repo, symbol, history_days)
    logger.info("Fetching SOL funding rate from %s to %s", start.date(), end.date())
    try:
        with DeribitFundingRateClient(instrument="SOL_USDC-PERPETUAL") as client:
            df = client.fetch_hourly(start, end)
    except Exception as exc:
        logger.warning("Funding rate fetch failed (%s) — skipping persist", exc)
        return
    if df.empty:
        logger.warning("Funding rate fetch returned no data — skipping persist")
        return
    repo.append(symbol, df)
    repo.save_sample(symbol)


def _fetch_oi_snapshot(repo: ParquetRepository) -> None:
    symbol = "OI_RATIO"
    last = repo.last_timestamp(symbol)
    today = datetime.now(tz=timezone.utc).date()
    if last is not None and last.date() >= today:
        logger.info("OI ratio already up to date")
        return
    logger.info("Fetching SOL/BTC OI snapshot")
    with DeribitOIClient() as client:
        df = client.fetch_daily_snapshot()
    repo.append(symbol, df)
    repo.save_sample(symbol)


def _fetch_iv_skew(config: dict, repo: ParquetRepository) -> None:
    symbol = "BTC_IV_SKEW"
    last = repo.last_timestamp(symbol)
    today = datetime.now(tz=timezone.utc).date()
    if last is not None and last.date() >= today:
        logger.info("BTC IV skew already up to date")
        return
    days_ahead: int = config.get("iv_skew", {}).get("days_ahead", 14)
    logger.info("Fetching BTC IV skew (within %d days)", days_ahead)
    with DeribitIVSkewClient(currency="BTC", days_ahead=days_ahead) as client:
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
