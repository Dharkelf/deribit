"""VixClient — fetches VIX daily data via yfinance and resamples to hourly.

VIX is only available at daily resolution; values are forward-filled to
align with the hourly Deribit candles.
"""

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_TICKER = "^VIX"


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
            logger.warning("yfinance returned no data for %s", self._ticker)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # flatten multi-level columns that yfinance sometimes returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.index = pd.to_datetime(raw.index, utc=True)
        raw = raw[["Open", "High", "Low", "Close"]].rename(
            columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
        )
        raw["volume"] = 0.0

        hourly = (
            raw.resample("1h")
            .ffill()
            .dropna()
        )

        logger.info("Fetched %d hourly VIX rows (%s → %s)", len(hourly), start.date(), end.date())
        return hourly
