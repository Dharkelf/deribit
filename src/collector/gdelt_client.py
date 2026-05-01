"""GdeltClient — GDELT DOC 2.0 → daily US military activity score [0, 1].

Queries GDELT for daily news article volume about US military operations.
Score is normalized to [0, 1] via rolling 365-day max.
Raises RuntimeError on API failure so the caller decides whether to persist.
"""

import logging
import time
from datetime import datetime

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.gdeltproject.org"
_NORM_DAYS = 365
_MAX_RETRIES = 3
_RETRY_DELAY = 65  # seconds — GDELT rate-limits per minute

# CAMEO-aligned keyword query for US military force projection
_QUERY = (
    '("US military" OR "American forces" OR "US troops" OR '
    '"Pentagon" OR "US airstrike" OR "US drone strike" OR '
    '"US military operation" OR "US offensive") '
    '(operation OR strike OR deploy OR attack OR offensive OR bombing)'
)


class GdeltClient:
    """Repository for GDELT-based US military activity scores."""

    def __init__(self) -> None:
        self._http = httpx.Client(base_url=_BASE_URL, timeout=60.0)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "GdeltClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_daily_score(self, start: datetime, end: datetime) -> pd.DataFrame | None:
        """Return daily US military activity score [0, 1] from *start* to *end*.

        Returns None on API failure so the caller can skip persisting stale zeros.
        """
        date_index = pd.date_range(start.date(), end.date(), freq="D", tz="UTC")
        date_index.name = "timestamp"

        raw = self._fetch_timeline(start, end)
        if raw is None:
            logger.warning("GDELT fetch failed — returning None, caller must not persist")
            return None
        if raw.empty:
            logger.warning("GDELT returned empty timeline — score set to 0")
            return pd.DataFrame({"GDELT_military_score": 0.0}, index=date_index)

        raw = raw.reindex(date_index, fill_value=0.0)
        score = self._normalize(raw)
        logger.info("GDELT score: %d days, mean=%.3f", len(score), score.mean())
        return pd.DataFrame({"GDELT_military_score": score}, index=date_index)

    # ------------------------------------------------------------------

    def _fetch_timeline(self, start: datetime, end: datetime) -> pd.Series | None:
        params = {
            "query": _QUERY,
            "mode": "timelinevolraw",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
            "format": "json",
        }
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self._http.get("/api/v2/doc/doc", params=params)
                if resp.status_code == 429:
                    logger.warning(
                        "GDELT 429 rate-limit (attempt %d/%d) — waiting %ds",
                        attempt, _MAX_RETRIES, _RETRY_DELAY,
                    )
                    time.sleep(_RETRY_DELAY)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except httpx.HTTPStatusError:
                raise
            except Exception as exc:
                logger.warning("GDELT API error: %s", exc)
                return None
        else:
            logger.warning("GDELT gave 429 on all %d attempts", _MAX_RETRIES)
            return None

        timeline = data.get("timeline", [{}])
        if not timeline or "data" not in timeline[0]:
            return pd.Series(dtype=float)

        rows = timeline[0]["data"]
        dates = pd.to_datetime(
            [r["date"] for r in rows], format="%Y%m%dT%H%M%SZ", utc=True
        )
        values = [float(r["value"]) for r in rows]
        series = pd.Series(values, index=dates, dtype=float)
        return series.resample("1D").sum()

    def _normalize(self, raw: pd.Series) -> pd.Series:
        rolling_max = raw.rolling(_NORM_DAYS, min_periods=1).max().clip(lower=1.0)
        return (raw / rolling_max).clip(0.0, 1.0)
