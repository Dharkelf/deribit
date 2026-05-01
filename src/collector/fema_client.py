"""FemaClient — FEMA Major Disaster Declarations → daily severity score [0, 1].

Score: rolling 90-day count of active DR declarations,
normalized to [0, 1] via rolling 365-day max.
Weekend/holiday gaps are forward-filled downstream in load_common_dataframe.
"""

import logging
from datetime import datetime

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.fema.gov/api/open/v2"
_ROLLING_DAYS = 90
_NORM_DAYS = 365
_PAGE_SIZE = 1000


class FemaClient:
    """Repository for FEMA disaster severity scores."""

    def __init__(self) -> None:
        self._http = httpx.Client(base_url=_BASE_URL, timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "FemaClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_daily_score(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Return daily disaster score [0, 1] from *start* to *end* as a DataFrame.

        Fetches from (start − 90 days) so the rolling window is warm at *start*.
        """
        fetch_from = start - pd.Timedelta(days=_ROLLING_DAYS)
        declarations = self._fetch_declarations(fetch_from)
        date_index = pd.date_range(start.date(), end.date(), freq="D", tz="UTC")
        date_index.name = "timestamp"

        if declarations.empty:
            logger.warning("FEMA returned no declarations — score set to 0")
            return pd.DataFrame({"FEMA_score": 0.0}, index=date_index)

        score = self._compute_score(declarations, date_index)
        logger.info("FEMA score: %d days, mean=%.3f", len(score), score.mean())
        return pd.DataFrame({"FEMA_score": score}, index=date_index)

    # ------------------------------------------------------------------

    def _fetch_declarations(self, fetch_from: datetime) -> pd.DataFrame:
        records: list[dict] = []
        skip = 0
        fetch_from_str = fetch_from.strftime("%Y-%m-%dT00:00:00Z")

        while True:
            params = {
                "$filter": (
                    f"declarationType eq 'DR' "
                    f"and declarationDate ge '{fetch_from_str}'"
                ),
                "$select": "disasterNumber,declarationDate,incidentEndDate",
                "$orderby": "declarationDate asc",
                "$top": _PAGE_SIZE,
                "$skip": skip,
            }
            try:
                resp = self._http.get("/DisasterDeclarationsSummaries", params=params)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("FEMA API error: %s", exc)
                break

            batch = resp.json().get("DisasterDeclarationsSummaries", [])
            if not batch:
                break
            records.extend(batch)
            if len(batch) < _PAGE_SIZE:
                break
            skip += _PAGE_SIZE

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["declarationDate"] = pd.to_datetime(df["declarationDate"], utc=True)
        df["incidentEndDate"] = pd.to_datetime(
            df["incidentEndDate"], utc=True, errors="coerce"
        )
        return df

    def _compute_score(
        self,
        declarations: pd.DataFrame,
        date_index: pd.DatetimeIndex,
    ) -> pd.Series:
        window = pd.Timedelta(days=_ROLLING_DAYS)
        counts = pd.Series(0.0, index=date_index)

        for day in date_index:
            active = declarations[
                (declarations["declarationDate"] >= day - window)
                & (declarations["declarationDate"] <= day)
            ]
            counts[day] = float(len(active))

        rolling_max = counts.rolling(_NORM_DAYS, min_periods=1).max().clip(lower=1.0)
        return (counts / rolling_max).clip(0.0, 1.0)
