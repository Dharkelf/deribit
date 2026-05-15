"""VixClient — fetches VIX daily data via yfinance and resamples to hourly.

VIX is only available at daily resolution; values are forward-filled to
align with the hourly Deribit candles.

Primary source: yfinance (^VIX).
Fallback: FRED VIXCLS CSV — used when yfinance returns an empty response
(Yahoo Finance API outages are occasional; FRED is more reliable).
"""

import logging
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

_TICKER   = "^VIX"
_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


def _fetch_vix_fred(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch VIX from FRED VIXCLS as fallback. Returns daily DataFrame."""
    try:
        r = requests.get(_FRED_URL, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        df = df[df.iloc[:, 0] != "."]
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"
        df = df.rename(columns={df.columns[0]: "close"})
        df["close"] = df["close"].astype(float)
        df["open"] = df["high"] = df["low"] = df["close"]
        df["volume"] = 0.0
        def _to_utc(dt: datetime) -> pd.Timestamp:
            ts = pd.Timestamp(dt)
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

        df = df.loc[(_to_utc(start) <= df.index) & (df.index <= _to_utc(end))]
        logger.info("FRED VIX fallback: %d daily rows", len(df))
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as exc:
        logger.warning("FRED VIX fallback failed: %s", exc)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


class VixClient:
    """Repository for VIX time-series data."""

    def __init__(self, ticker: str = _TICKER) -> None:
        self._ticker = ticker

    def fetch_ohlcv(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch VIX between *start* and *end*, resampled to 1-hour frequency.

        Daily values are forward-filled into hourly slots so the DataFrame
        aligns with Deribit candles. Volume is always 0 (VIX has none).
        """
        raw = yf.download(
            self._ticker,
            start=start.date(),
            end=end.date(),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            logger.warning("yfinance returned no data for %s — trying FRED fallback", self._ticker)
            raw = _fetch_vix_fred(start, end)

        if raw.empty:
            logger.warning("VIX unavailable from both yfinance and FRED")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Normalise: yfinance returns Title-case columns; FRED fallback already lowercase
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.index = pd.to_datetime(raw.index, utc=True)
        raw.index.name = "timestamp"

        if "Close" in raw.columns:
            raw = raw[["Open", "High", "Low", "Close"]].rename(
                columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
            )
            raw["volume"] = 0.0

        # resample to hourly and filter to the requested range.
        # No .dropna() here — if the first VIX daily entry arrives after `start`
        # (e.g. weekend), early slots have NaN which load_common_dataframe fills
        # via its own reindex+ffill pass.
        hourly = (
            raw.resample("1h")
            .ffill()
            .loc[start:end]
        )
        hourly.index.name = "timestamp"

        logger.info("Fetched %d hourly VIX rows (%s → %s)", len(hourly), start.date(), end.date())
        return hourly
