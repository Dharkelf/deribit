"""DeribitOptionsClient — computes BTC Max Pain from the Deribit public options chain.

Max Pain is the strike price at which all option holders (calls + puts) collectively
suffer the greatest loss at expiration. It acts as a gravitational target for price.

Data source: Deribit public API — no API key required.
  GET /public/get_book_summary_by_currency?currency=BTC&kind=option

Storage: one row per day with the mean max pain strike across all expiries
falling within the next `days_ahead` calendar days.

Feature (computed in features.py at prediction time):
  max_pain_diff     = mean_max_pain − BTC_close   (absolute USD)
  max_pain_diff_pct = (mean_max_pain − BTC_close) / BTC_close  (fraction)
"""

import logging
import re
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.deribit.com/api/v2"
_EXPIRY_RE = re.compile(r"BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])$")
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


class DeribitOptionsClient:
    """Repository for BTC Max Pain derived from Deribit options open interest."""

    def __init__(self, days_ahead: int = 30) -> None:
        self._days_ahead = days_ahead
        self._http = httpx.Client(base_url=_BASE_URL, timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "DeribitOptionsClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_daily_snapshot(self) -> pd.DataFrame:
        """Fetch current options chain and return a single-row daily snapshot.

        Returns a DataFrame with columns:
          BTC_options_max_pain   — mean max pain strike for next-month expiries
        Indexed by today's UTC date (daily resolution).
        """
        today = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        cutoff = today + pd.Timedelta(days=self._days_ahead)

        summaries = self._fetch_summaries()
        if summaries.empty:
            logger.warning("Options chain empty — max pain set to NaN")
            return pd.DataFrame(
                {"BTC_options_max_pain": np.nan}, index=pd.DatetimeIndex([today])
            )

        # Filter to expiries within the next days_ahead window
        mask = (summaries["expiry"] >= today) & (summaries["expiry"] <= cutoff)
        upcoming = summaries[mask]

        if upcoming.empty:
            logger.warning("No expiries in next %d days", self._days_ahead)
            return pd.DataFrame(
                {"BTC_options_max_pain": np.nan}, index=pd.DatetimeIndex([today])
            )

        max_pains: list[float] = []
        for expiry, group in upcoming.groupby("expiry"):
            mp = _compute_max_pain(group)
            if mp is not None:
                max_pains.append(mp)
                logger.debug("Max pain for %s: %.0f", expiry.date(), mp)

        mean_mp = float(np.mean(max_pains)) if max_pains else np.nan
        index = pd.DatetimeIndex([today], name="timestamp")
        logger.info(
            "BTC Max Pain (next %d days, %d expiries): %.0f",
            self._days_ahead, len(max_pains), mean_mp,
        )
        return pd.DataFrame({"BTC_options_max_pain": [mean_mp]}, index=index)

    # ------------------------------------------------------------------

    def _fetch_summaries(self) -> pd.DataFrame:
        try:
            resp = self._http.get(
                "/public/get_book_summary_by_currency",
                params={"currency": "BTC", "kind": "option"},
            )
            resp.raise_for_status()
            results = resp.json().get("result", [])
        except Exception as exc:
            logger.warning("Deribit options API error: %s", exc)
            return pd.DataFrame()

        records = []
        for item in results:
            parsed = _parse_instrument(item.get("instrument_name", ""))
            if parsed is None:
                continue
            expiry, strike, opt_type = parsed
            records.append({
                "expiry": expiry,
                "strike": strike,
                "type": opt_type,
                "open_interest": float(item.get("open_interest", 0.0)),
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_instrument(name: str) -> tuple[datetime, float, str] | None:
    """Parse 'BTC-27DEC24-50000-C' → (expiry_datetime, strike, type)."""
    m = _EXPIRY_RE.match(name)
    if not m:
        return None
    day, mon_str, yr, strike_str, opt_type = m.groups()
    month = _MONTH_MAP.get(mon_str)
    if month is None:
        return None
    year = 2000 + int(yr)
    expiry = datetime(year, month, int(day), 8, 0, tzinfo=timezone.utc)
    return expiry, float(strike_str), opt_type


def _compute_max_pain(group: pd.DataFrame) -> float | None:
    """Return the max pain strike for a single expiry group.

    For each candidate strike S, total pain =
      Σ_calls  max(0, S − K) × OI_call
    + Σ_puts   max(0, K − S) × OI_put
    Max pain = S that minimises total pain.
    """
    calls = group[group["type"] == "C"].set_index("strike")["open_interest"]
    puts = group[group["type"] == "P"].set_index("strike")["open_interest"]
    strikes = np.array(sorted(group["strike"].unique()))

    if len(strikes) < 2:
        return None

    total_pain = np.zeros(len(strikes))
    for i, s in enumerate(strikes):
        call_pain = np.sum(np.maximum(0, s - calls.index.values) * calls.values)
        put_pain = np.sum(np.maximum(0, puts.index.values - s) * puts.values)
        total_pain[i] = call_pain + put_pain

    return float(strikes[np.argmin(total_pain)])
