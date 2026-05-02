"""ParquetRepository — append-only storage for OHLCV DataFrames.

All read/write access to data/raw/*.parquet goes through this class.
Raw data is immutable: existing rows are never modified, only new rows appended.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ParquetRepository:
    """Repository pattern: isolates Parquet I/O from business logic."""

    def __init__(self, raw_dir: Path) -> None:
        self._dir = raw_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        return self._dir / f"{symbol}.parquet"

    def last_timestamp(self, symbol: str) -> pd.Timestamp | None:
        """Return the latest stored timestamp for *symbol*, or None if no data."""
        path = self._path(symbol)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        return df.index.max() if not df.empty else None

    def load(self, symbol: str) -> pd.DataFrame:
        """Load the full OHLCV DataFrame for *symbol*."""
        path = self._path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.read_parquet(path)

    def save_sample(self, symbol: str, n: int = 10) -> None:
        """Write *n* random rows from the stored Parquet to data/raw/samples/<symbol>.csv.

        CSV is used intentionally — samples are for human inspection only.
        The file is overwritten on every call so it always reflects recent data.
        """
        df = self.load(symbol)
        if df.empty:
            return
        sample = df.sample(n=min(n, len(df)), random_state=None).sort_index()
        sample_dir = self._dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        path = sample_dir / f"{symbol}.csv"
        sample.to_csv(path)
        logger.info("Sample saved: %s (%d rows) → %s", symbol, len(sample), path)

    def append(self, symbol: str, df: pd.DataFrame) -> None:
        """Append *df* to the existing Parquet file for *symbol*.

        Deduplicates on timestamp index before writing — safe to call with
        overlapping data.
        """
        if df.empty:
            return

        path = self._path(symbol)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df])
        else:
            combined = df.copy()

        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined.index.name = "timestamp"
        combined.to_parquet(path)
        logger.info("Saved %d rows for %s → %s", len(combined), symbol, path.name)
