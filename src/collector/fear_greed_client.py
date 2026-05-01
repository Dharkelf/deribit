"""Fear & Greed Index fetchers — Crypto and Stock market sentiment.

Sources
-------
Crypto F&G : alternative.me public API  — no auth required
Stock F&G  : CNN Fear & Greed dataviz endpoint — no auth required

Both return daily integer values [0, 100].
The collectors normalise to [0.0, 1.0] before persisting.
Returns None on any network / parse failure so the caller can decide whether
to skip persisting (preserving any previously stored good data).
"""

import logging
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 20

# ─── Crypto F&G ───────────────────────────────────────────────────────────────

_CRYPTO_URL = "https://api.alternative.me/fng/"


def fetch_crypto_fear_greed(days: int = 365) -> pd.DataFrame | None:
    """Fetch Crypto Fear & Greed from alternative.me.

    Returns DataFrame with UTC DatetimeIndex and column 'crypto_fear_greed' [0.0, 1.0].
    """
    try:
        resp = requests.get(
            _CRYPTO_URL,
            params={"limit": days, "format": "json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
    except Exception as e:
        logger.warning("Crypto F&G fetch failed: %s", e)
        return None

    rows = []
    for item in data:
        try:
            ts = datetime.fromtimestamp(int(item["timestamp"]), tz=timezone.utc)
            rows.append({"timestamp": ts, "crypto_fear_greed": float(item["value"]) / 100.0})
        except (KeyError, ValueError):
            continue

    if not rows:
        logger.warning("Crypto F&G: no parseable rows in response")
        return None

    df = (
        pd.DataFrame(rows)
        .set_index("timestamp")
        .sort_index()
    )
    df.index.name = "timestamp"
    logger.info("Crypto F&G: %d days fetched (last value=%.2f)", len(df), df["crypto_fear_greed"].iloc[-1])
    return df


# ─── Stock F&G (CNN) ──────────────────────────────────────────────────────────

_CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

_CNN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://edition.cnn.com/",
}


def fetch_stock_fear_greed() -> pd.DataFrame | None:
    """Fetch CNN Stock Fear & Greed historical data.

    Returns DataFrame with UTC DatetimeIndex and column 'stock_fear_greed' [0.0, 1.0].
    """
    try:
        resp = requests.get(_CNN_URL, headers=_CNN_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.warning("Stock F&G (CNN) fetch failed: %s", e)
        return None

    try:
        hist = payload["fear_and_greed_historical"]["data"]
    except (KeyError, TypeError) as e:
        logger.warning("Stock F&G unexpected response structure: %s", e)
        return None

    rows = []
    for point in hist:
        try:
            # CNN uses millisecond POSIX timestamps
            ts = datetime.fromtimestamp(point["x"] / 1000.0, tz=timezone.utc)
            rows.append({"timestamp": ts, "stock_fear_greed": float(point["y"]) / 100.0})
        except (KeyError, ValueError, TypeError):
            continue

    if not rows:
        logger.warning("Stock F&G (CNN): no parseable rows in response")
        return None

    df = (
        pd.DataFrame(rows)
        .set_index("timestamp")
        .sort_index()
    )
    df.index.name = "timestamp"
    logger.info("Stock F&G (CNN): %d days fetched (last value=%.2f)", len(df), df["stock_fear_greed"].iloc[-1])
    return df
