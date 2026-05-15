"""DeribitIVSkewClient — computes options IV skew from Deribit public options chain.

IV skew = mean(ATM put IV) − mean(ATM call IV) for near-term expiries.

Interpretation:
  Positive skew: puts more expensive → crash protection demand → bearish sentiment
  Negative skew: calls more expensive → upside expectation → bullish sentiment
  Near zero:     balanced → neutral

Uses BTC options (most liquid on Deribit) as a crypto-wide fear/greed indicator.
ATM filter: strikes within 10 % of current underlying price.
Near-term filter: expiries within days_ahead calendar days.

Endpoint: GET /public/get_book_summary_by_currency?currency=BTC&kind=option
  Response includes mark_iv (percentage, e.g. 42.87 = 42.87 % annualised IV).
"""

import logging
import re
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.deribit.com/api/v2"
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_EXPIRY_RE = re.compile(r"(\w+)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])$")


class DeribitIVSkewClient:
    """Repository for BTC options IV skew — crypto fear/greed proxy."""

    def __init__(self, currency: str = "BTC", days_ahead: int = 14) -> None:
        self._currency = currency
        self._days_ahead = days_ahead
        self._http = httpx.Client(base_url=_BASE_URL, timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "DeribitIVSkewClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_daily_snapshot(self) -> pd.DataFrame:
        """Compute IV skew from current options chain; return single-row DataFrame.

        Column: btc_iv_skew (pct-points; positive = fear, negative = greed)
        Returns NaN row on API failure or insufficient near-term options.
        """
        today = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        index = pd.DatetimeIndex([today], name="timestamp")
        col = "btc_iv_skew"

        summaries = self._fetch_summaries()
        if summaries.empty:
            logger.warning("IV skew: empty options chain for %s", self._currency)
            return pd.DataFrame({col: [np.nan]}, index=index)

        cutoff = today + pd.Timedelta(days=self._days_ahead)
        near = summaries[
            (summaries["expiry"] >= today) & (summaries["expiry"] <= cutoff)
        ]
        if near.empty:
            logger.warning(
                "IV skew: no %s options expiring within %d days",
                self._currency,
                self._days_ahead,
            )
            return pd.DataFrame({col: [np.nan]}, index=index)

        underlying = float(near["underlying_price"].median())
        atm_mask = (
            (near["strike"] >= underlying * 0.90)
            & (near["strike"] <= underlying * 1.10)
            & near["mark_iv"].notna()
            & (near["mark_iv"] > 0)
        )
        atm = near[atm_mask]

        if atm.empty or atm["type"].nunique() < 2:
            logger.warning("IV skew: insufficient ATM %s options", self._currency)
            return pd.DataFrame({col: [np.nan]}, index=index)

        put_iv = float(atm.loc[atm["type"] == "P", "mark_iv"].mean())
        call_iv = float(atm.loc[atm["type"] == "C", "mark_iv"].mean())
        skew = put_iv - call_iv

        logger.info(
            "%s IV skew: put_iv=%.2f%%  call_iv=%.2f%%  skew=%.4f  (underlying=%.0f)",
            self._currency,
            put_iv,
            call_iv,
            skew,
            underlying,
        )
        return pd.DataFrame({col: [skew]}, index=index)

    def _fetch_summaries(self) -> pd.DataFrame:
        try:
            resp = self._http.get(
                "/public/get_book_summary_by_currency",
                params={"currency": self._currency, "kind": "option"},
            )
            resp.raise_for_status()
            results: list[dict] = resp.json().get("result", [])
        except Exception as exc:
            logger.warning("Deribit IV skew API error: %s", exc)
            return pd.DataFrame()

        records = []
        for item in results:
            parsed = _parse_option(item.get("instrument_name", ""))
            if parsed is None:
                continue
            expiry, strike, opt_type = parsed
            records.append(
                {
                    "expiry": expiry,
                    "strike": strike,
                    "type": opt_type,
                    "mark_iv": item.get("mark_iv"),
                    "underlying_price": item.get("underlying_price"),
                }
            )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
        df["mark_iv"] = pd.to_numeric(df["mark_iv"], errors="coerce")
        df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
        return df


def _parse_option(name: str) -> tuple[datetime, float, str] | None:
    """Parse 'BTC-27DEC24-50000-C' → (expiry_datetime, strike, type)."""
    m = _EXPIRY_RE.match(name)
    if not m:
        return None
    _, day, mon_str, yr, strike_str, opt_type = m.groups()
    month = _MONTH_MAP.get(mon_str)
    if month is None:
        return None
    year = 2000 + int(yr)
    expiry = datetime(year, month, int(day), 8, 0, tzinfo=timezone.utc)
    return expiry, float(strike_str), opt_type
