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


def _estimate_backtest_minutes(config: dict) -> str:
    """Rough estimate: ~3 s per XGB fold on Apple M-series."""
    try:
        import pandas as pd
        from src.utils.paths import raw_dir

        btc_path = raw_dir(config) / "BTC.parquet"
        if not btc_path.exists():
            return "~5"
        n_rows = len(pd.read_parquet(btc_path, columns=["close"]))
    except Exception:
        return "~5"

    bt_cfg = config.get("backtest", {})
    min_train_h = int(bt_cfg.get("min_train_days", 30)) * 24
    step_h = int(bt_cfg.get("step_days", 7)) * 24
    horizon_h = int(bt_cfg.get("horizon_hours", 24))
    n_folds = max(0, (n_rows - min_train_h - horizon_h) // step_h)
    minutes = max(1, round(n_folds * 3 / 60))
    return f"~{minutes}"


def _prompt_backtest(config: dict) -> None:
    border = "═" * 62
    est = _estimate_backtest_minutes(config)
    print(f"\n{border}")
    print("  Backtest verfügbar")
    print("  Option A: XGB Walk-Forward Forecast-Accuracy (wöchentliche Folds)")
    print("  Option B: HMM Regime-Strategie vs. Buy-and-Hold")
    print(
        "  Outputs:  backtest_results.parquet + backtest_report.png + BACKTEST_REPORT.md"
    )
    print(f"  Geschätzte Laufzeit: {est} min")
    print(f"{border}")
    try:
        answer = input("  Backtest jetzt ausführen? [j/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if answer == "j":
        _run_backtest(config)


def _run_backtest(config: dict) -> None:
    from src.backtest.engine import run as run_engine
    from src.backtest.report import generate as gen_report

    fold_df, strategies = run_engine(config)
    gen_report(fold_df, strategies, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deribit data pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["collect", "hmm", "backtest", "timing", "prophet_day", "intraday", "intraday_np"],
        default=None,
        help="Module to run (omit to run both collect + hmm sequentially)",
    )
    args = parser.parse_args()

    config = _load_config()
    _setup_logging(config.get("logging", {}).get("level", "INFO"))

    now_utc = pd.Timestamp.now(tz="UTC")
    config["_now_utc"] = now_utc
    config["_today"] = now_utc.normalize()  # today 00:00 UTC
    config["_cutoff"] = now_utc.normalize() - pd.Timedelta(
        hours=1
    )  # yesterday 23:00 UTC
    config["_last_hour"] = now_utc.floor("h")  # last full hour

    run_collect = args.command in (None, "collect", "hmm")
    run_hmm = args.command in (None, "hmm")

    if run_collect:
        from src.collector.fetcher import run as run_collector

        run_collector(config)

    if run_hmm:
        from src.hmm.visualize import run as run_visualize

        run_visualize(config)
        _prompt_backtest(config)

    if args.command == "backtest":
        _run_backtest(config)

    if args.command == "timing":
        from src.backtest.timing import run as run_timing

        run_timing(config)

    if args.command == "prophet_day":
        from src.backtest.prophet_day import run as run_prophet_day

        run_prophet_day(config)

    if args.command == "intraday":
        from src.backtest.intraday import run as run_intraday

        run_intraday(config)

    if args.command == "intraday_np":
        from src.backtest.intraday_np import run as run_intraday_np

        run_intraday_np(config)


if __name__ == "__main__":
    sys.exit(main())
