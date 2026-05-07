"""Central path resolution — all file I/O derives paths from here."""

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_dir(config: dict) -> Path:
    path = project_root() / config["storage"]["raw_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir(config: dict) -> Path:
    path = project_root() / config["storage"]["models_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir(config: dict) -> Path:
    path = project_root() / config["storage"]["processed_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def symbol_parquet(config: dict, symbol: str) -> Path:
    return raw_dir(config) / f"{symbol}.parquet"
