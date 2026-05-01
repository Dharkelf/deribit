"""US Federal Funds Rate fetcher — FRED DFF series via public CSV endpoint.

No API key required. FRED's CSV export endpoint is openly accessible.

Series used: DFF — Effective Federal Funds Rate (daily, business days only).

Two columns produced:
  fed_rate             — effective daily rate in percent (e.g. 4.33)
  fed_rate_last_change — signed value of the most recent FOMC rate change (%)
                         forward-filled so every hourly row carries the most
                         recent decision (e.g. −0.25 after a cut)
"""

import logging
from io import StringIO

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_SERIES_ID = "DFF"
_TIMEOUT = 20


def fetch_fed_rate() -> pd.DataFrame | None:
    """Fetch FRED DFF and derive last FOMC change feature.

    Returns DataFrame with UTC DatetimeIndex and columns
    'fed_rate' and 'fed_rate_last_change'.
    Returns None on failure.
    """
    try:
        resp = requests.get(
            _FRED_CSV_URL,
            params={"id": _SERIES_ID},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        df = pd.read_csv(
            StringIO(resp.text),
            parse_dates=["observation_date"],
            index_col="observation_date",
            na_values=[".", ""],
        )
    except Exception as e:
        logger.warning("Fed rate (FRED %s) fetch failed: %s", _SERIES_ID, e)
        return None

    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    df.index.name = "timestamp"
    df.columns = ["fed_rate"]
    df = df.dropna()

    if df.empty:
        logger.warning("Fed rate: empty DataFrame after parsing")
        return None

    # FOMC decisions are always ≥ 0.25 pp; effective DFF fluctuates ±0.01 as noise.
    # Only mark genuine policy moves (threshold: 0.10 pp) as last_change.
    daily_change = df["fed_rate"].diff()
    fomc_changes = daily_change.where(daily_change.abs() >= 0.10)
    df["fed_rate_last_change"] = fomc_changes.ffill().fillna(0.0)

    logger.info(
        "Fed rate: %d daily rows | current=%.4f%%  last_change=%+.4f%%",
        len(df),
        float(df["fed_rate"].iloc[-1]),
        float(df["fed_rate_last_change"].iloc[-1]),
    )
    return df
