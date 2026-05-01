"""DeribitOptionsClient — computes BTC Max Pain from the Deribit public options chain.

Max Pain is the strike price at which all option holders (calls + puts) collectively
suffer the greatest loss at expiration. It acts as a gravitational target for price.

Data source: Deribit public API — no API key required.
  GET /public/get_book_summary_by_currency?currency=BTC&kind=option

Storage: one row per day with two max pain columns computed in a single API call:
  BTC_options_max_pain      — mean max pain over next days_ahead (30d) expiries
  BTC_options_max_pain_7d   — mean max pain over next days_ahead_short (7d) expiries

Feature (computed in features.py at prediction time):
  max_pain_diff_usd / max_pain_diff_pct         — 30d window vs BTC close
  max_pain_7d_diff_usd / max_pain_7d_diff_pct   — 7d window vs BTC close
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

    def __init__(self, days_ahead: int = 30, days_ahead_short: int = 7) -> None:
        self._days_ahead = days_ahead
        self._days_ahead_short = days_ahead_short
        self._http = httpx.Client(base_url=_BASE_URL, timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "DeribitOptionsClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_daily_snapshot(self) -> pd.DataFrame:
        """Fetch current options chain and return a single-row daily snapshot.

        One API call computes both windows:
          BTC_options_max_pain     — mean max pain over next days_ahead (30d) expiries
          BTC_options_max_pain_7d  — mean max pain over next days_ahead_short (7d) expiries
        Indexed by today's UTC date (daily resolution).
        """
        today = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        cutoff_long  = today + pd.Timedelta(days=self._days_ahead)
        cutoff_short = today + pd.Timedelta(days=self._days_ahead_short)
        index = pd.DatetimeIndex([today], name="timestamp")

        summaries = self._fetch_summaries()
        if summaries.empty:
            logger.warning("Options chain empty — max pain set to NaN")
            return pd.DataFrame(
                {"BTC_options_max_pain": np.nan, "BTC_options_max_pain_7d": np.nan},
                index=index,
            )

        def _mean_max_pain(cutoff: pd.Timestamp) -> float:
            mask = (summaries["expiry"] >= today) & (summaries["expiry"] <= cutoff)
            upcoming = summaries[mask]
            if upcoming.empty:
                return np.nan
            pains = [
                mp for _, g in upcoming.groupby("expiry")
                if (mp := _compute_max_pain(g)) is not None
            ]
            return float(np.mean(pains)) if pains else np.nan

        mp_30d = _mean_max_pain(cutoff_long)
        mp_7d  = _mean_max_pain(cutoff_short)

        logger.info(
            "BTC Max Pain — 7d: %.0f  30d: %.0f",
            mp_7d if not np.isnan(mp_7d) else -1,
            mp_30d if not np.isnan(mp_30d) else -1,
        )
        return pd.DataFrame(
            {"BTC_options_max_pain": [mp_30d], "BTC_options_max_pain_7d": [mp_7d]},
            index=index,
        )

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
