"""Entry point — wires modules together, no business logic here."""

import argparse
import logging
import sys

import pandas as pd
import yaml


def _load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Deribit data pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["collect", "hmm"],
        default=None,
        help="Module to run (omit to run both sequentially)",
    )
    args = parser.parse_args()

    config = _load_config()
    _setup_logging(config.get("logging", {}).get("level", "INFO"))

    now_utc = pd.Timestamp.now(tz="UTC")
    config["_now_utc"]   = now_utc
    config["_today"]     = now_utc.normalize()                              # today 00:00 UTC
    config["_cutoff"]    = now_utc.normalize() - pd.Timedelta(hours=1)     # yesterday 23:00 UTC
    config["_last_hour"] = now_utc.floor("h")                              # last full hour

    # hmm always collects first so plots show data up to the current hour
    run_collect = args.command in (None, "collect", "hmm")
    run_hmm     = args.command in (None, "hmm")

    if run_collect:
        from src.collector.fetcher import run as run_collector
        run_collector(config)

    if run_hmm:
        from src.hmm.predict import run as run_hmm_pipeline
        run_hmm_pipeline(config)

        from src.hmm.predict_xgb import run as run_xgb
        run_xgb(config)

        from src.hmm.predict_prophet import run as run_prophet
        run_prophet(config)


if __name__ == "__main__":
    sys.exit(main())
