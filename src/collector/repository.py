"""ParquetRepository — append-only storage for OHLCV DataFrames.

All read/write access to data/raw/*.parquet goes through this class.
Raw data is immutable: existing rows are never modified, only new rows appended.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file, falling back to fastparquet on pyarrow 19.x read errors.

    pyarrow 19.0.0 has a bug ('Repetition level histogram size mismatch') when
    reading files written with certain timestamp precisions. fastparquet handles
    these gracefully. The fallback is transparent to callers; the file is always
    re-saved as pyarrow on the next append(), normalising the format.
    """
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except OSError:
        logger.debug("pyarrow read failed for %s — retrying with fastparquet", path.name)
        return pd.read_parquet(path, engine="fastparquet")


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to Parquet, normalising timestamps to microsecond precision.

    Nanosecond-precision timestamps can trigger the pyarrow 19.x histogram mismatch
    bug on the subsequent read. Casting to datetime64[us, UTC] eliminates the
    precision mismatch before writing.
    """
    out = df.copy()
    if hasattr(out.index, "tz"):
        out.index = out.index.astype("datetime64[us, UTC]")
    out.to_parquet(path, engine="pyarrow")


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
        df = _read_parquet(path)
        return df.index.max() if not df.empty else None

    def load(self, symbol: str) -> pd.DataFrame:
        """Load the full OHLCV DataFrame for *symbol*."""
        path = self._path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return _read_parquet(path)

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
            existing = _read_parquet(path)
            combined = pd.concat([existing, df])
        else:
            combined = df.copy()

        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined.index.name = "timestamp"
        _write_parquet(combined, path)
        logger.info("Saved %d rows for %s → %s", len(combined), symbol, path.name)
