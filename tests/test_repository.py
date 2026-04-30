"""Unit tests for ParquetRepository — uses a tmp_path, no real data/raw."""

import pandas as pd
import pytest

from src.collector.repository import ParquetRepository


def _sample_df(timestamps: list[str]) -> pd.DataFrame:
    index = pd.to_datetime(timestamps, utc=True)
    return pd.DataFrame(
        {"open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0},
        index=index,
    )


def test_last_timestamp_none_when_no_file(tmp_path: object) -> None:
    repo = ParquetRepository(tmp_path)
    assert repo.last_timestamp("BTC") is None


def test_append_and_load_roundtrip(tmp_path: object) -> None:
    repo = ParquetRepository(tmp_path)
    df = _sample_df(["2024-01-01T00:00:00", "2024-01-01T01:00:00"])
    repo.append("BTC", df)
    loaded = repo.load("BTC")

    assert len(loaded) == 2
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert loaded.index.name == "timestamp"


def test_append_deduplicates(tmp_path: object) -> None:
    repo = ParquetRepository(tmp_path)
    df1 = _sample_df(["2024-01-01T00:00:00", "2024-01-01T01:00:00"])
    df2 = _sample_df(["2024-01-01T01:00:00", "2024-01-01T02:00:00"])  # overlap at 01:00
    repo.append("BTC", df1)
    repo.append("BTC", df2)
    loaded = repo.load("BTC")

    assert loaded.index.is_unique
    assert len(loaded) == 3


def test_last_timestamp_after_append(tmp_path: object) -> None:
    repo = ParquetRepository(tmp_path)
    df = _sample_df(["2024-01-01T00:00:00", "2024-01-01T01:00:00"])
    repo.append("BTC", df)

    last = repo.last_timestamp("BTC")
    assert last == pd.Timestamp("2024-01-01T01:00:00", tz="UTC")


def test_append_empty_df_is_noop(tmp_path: object) -> None:
    repo = ParquetRepository(tmp_path)
    repo.append("BTC", pd.DataFrame())
    assert not (tmp_path / "BTC.parquet").exists()


def test_save_sample_creates_csv(tmp_path: object) -> None:
    repo = ParquetRepository(tmp_path)
    df = _sample_df([f"2024-01-01T{h:02d}:00:00" for h in range(20)])
    repo.append("BTC", df)
    repo.save_sample("BTC", n=10)

    sample_path = tmp_path / "samples" / "BTC.csv"
    assert sample_path.exists()
    sample = pd.read_csv(sample_path, index_col=0, parse_dates=True)
    assert len(sample) == 10


def test_save_sample_capped_at_available_rows(tmp_path: object) -> None:
    repo = ParquetRepository(tmp_path)
    df = _sample_df(["2024-01-01T00:00:00", "2024-01-01T01:00:00"])
    repo.append("BTC", df)
    repo.save_sample("BTC", n=10)

    sample = pd.read_csv(tmp_path / "samples" / "BTC.csv", index_col=0)
    assert len(sample) == 2  # only 2 rows available


def test_save_sample_noop_when_no_data(tmp_path: object) -> None:
    repo = ParquetRepository(tmp_path)
    repo.save_sample("BTC")  # no parquet exists yet
    assert not (tmp_path / "samples").exists()
