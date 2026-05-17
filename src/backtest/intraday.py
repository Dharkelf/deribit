"""Intraday SOL long strategy: buy 04:00 UTC, exit {16,18,20,22}:00 UTC.

Strategy
--------
Entry  : Market order at the 04:00 UTC candle close price.
Exit   : One of {16:00, 18:00, 20:00, 22:00} UTC, selected by XGB.
Signal : XGB recursive forecast (base + q10 + q90) trained on all data
         before the entry candle.  Exit chosen as:
           argmax over candidates of
             expected_return / (1 + ci_relative_width)
         subject to expected_gross_return > 0 (direction filter only).
Skipped: Days where no exit candidate clears the TC hurdle.
TC     : 0.01 per leg  →  0.02 round-trip per trade.
Invest : 1.0 per trade (P&L expressed as fraction of invested capital).
Period : Last 12 months of available data.
Walk-forward: XGB (base + q10 + q90, 500 trees) retrained every 7 days.
CI     : 80 % prediction interval from quantile regression (q10 / q90).
"""

import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from src.hmm.features import build_feature_matrix, load_common_dataframe
from src.hmm.optimizer import load_best_features
from src.hmm.predict_xgb import (
    _build_train_data,
    _filter_24h_features,
    _recursive_forecast,
    _train_model,
)
from src.utils.paths import processed_dir

logger = logging.getLogger(__name__)

_TC_PER_LEG: float = 0.01        # 1 % per leg
_ENTRY_HOUR: int   = 4            # 04:00 UTC
_EXIT_HOURS: list[int] = [16, 18, 20, 22]   # UTC hours for exit candidates
# 0-indexed positions in the recursive-forecast array:
# exp[j] = prediction for entry_ts + (j+1) hours
# 16:00 = entry + 12 h → j=11; 18:00 → j=13; 20:00 → j=15; 22:00 → j=17
_EXIT_IDX: list[int] = [h - _ENTRY_HOUR - 1 for h in _EXIT_HOURS]   # [11,13,15,17]
_N_STEPS:  int = 20               # recursive steps from 04:00 (covers up to 00:00+1d)
_RETRAIN_DAYS: int = 7
_N_EST:    int = 500              # trees per model (fast; 3 models per retrain)
_LOOKBACK_YEARS: int = 1          # evaluation window


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────


def _select_exit(
    entry_price: float,
    exp: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[int, int, float, float] | None:
    """Pick best exit from candidates.

    Returns (array_idx, exit_hour_utc, exp_return, ci_relative_width) or None
    when no candidate offers positive net expected return after TC.
    """
    best: tuple[int, int, float, float] | None = None
    best_score = -np.inf

    for idx, hour in zip(_EXIT_IDX, _EXIT_HOURS):
        if idx >= len(exp):
            continue
        exp_ret  = float(exp[idx] / entry_price) - 1.0
        if exp_ret <= 0:
            continue
        ci_abs   = max(float(hi[idx] - lo[idx]), 0.0)
        ci_rel   = ci_abs / max(float(exp[idx]), 1e-6)
        score    = exp_ret / (1.0 + ci_rel)
        if score > best_score:
            best_score = score
            best = (idx, hour, exp_ret, ci_rel)

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────


def _make_report(traded: pd.DataFrame, all_days: pd.DataFrame, out_path: Path) -> str:
    """Generate human-readable summary string and write INTRADAY_REPORT.md."""
    n_all    = len(all_days)
    n_traded = len(traded)
    trigger  = 100.0 * n_traded / max(n_all, 1)

    cum_pnl  = float(traded["net_return_pct"].sum()) if n_traded else 0.0
    hit_rate = float((traded["net_return_pct"] > 0).mean()) * 100 if n_traded else 0.0
    mean_net = float(traded["net_return_pct"].mean()) if n_traded else 0.0
    std_net  = float(traded["net_return_pct"].std(ddof=1)) if n_traded > 1 else 0.0

    # Annualised Sharpe: trades/year from actual frequency
    days_span = max((all_days["entry_ts"].max() - all_days["entry_ts"].min()).days, 1)
    trades_per_year = n_traded / days_span * 365
    sharpe = (mean_net / std_net * np.sqrt(trades_per_year)
              if std_net > 0 else 0.0)

    lines: list[str] = [
        "# Intraday SOL Backtest Report",
        "",
        f"Period : {all_days['entry_ts'].min().strftime('%Y-%m-%d')} → "
        f"{all_days['entry_ts'].max().strftime('%Y-%m-%d')}",
        "Strategy: Buy 04:00 UTC · Exit {16,18,20,22}:00 UTC (2-h sampling)",
        "TC      : 0.01/leg  (0.02 round-trip)  ·  Invest 1.0/trade",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Days evaluated | {n_all} |",
        f"| Trades executed | {n_traded} ({trigger:.1f}% trigger rate) |",
        f"| Hit rate | {hit_rate:.1f}% |",
        f"| Cumulative P&L | {cum_pnl:+.2f}% |",
        f"| Mean net return/trade | {mean_net:+.3f}% |",
        f"| Std net return/trade | {std_net:.3f}% |",
        f"| Sharpe (annualised) | {sharpe:.2f} |",
        f"| Trades/year (est.) | {trades_per_year:.0f} |",
        "",
    ]

    if n_traded:
        lines += ["## Exit-Hour Distribution", "", "| Exit UTC | Trades | % | Mean net | Hit rate |",
                  "|---|---|---|---|---|"]
        for hour in _EXIT_HOURS:
            sub = traded[traded["exit_hour_utc"] == hour]
            if sub.empty:
                continue
            lines.append(
                f"| {hour:02d}:00 | {len(sub)} | {100*len(sub)/n_traded:.1f}% "
                f"| {sub['net_return_pct'].mean():+.3f}% "
                f"| {100*(sub['net_return_pct']>0).mean():.1f}% |"
            )
        lines.append("")

        lines += ["## Monthly Breakdown", "",
                  "| Month | Trades | Hit rate | P&L |",
                  "|---|---|---|---|"]
        traded_ts = traded.copy()
        traded_ts["month"] = traded_ts["entry_ts"].dt.to_period("M")
        for month, grp in traded_ts.groupby("month"):
            lines.append(
                f"| {month} | {len(grp)} "
                f"| {100*(grp['net_return_pct']>0).mean():.0f}% "
                f"| {grp['net_return_pct'].sum():+.2f}% |"
            )
        lines.append("")

        # Expected vs actual return scatter summary
        exp_mean = float(traded["exp_return_pct"].mean())
        exp_corr = float(traded[["exp_return_pct", "actual_return_pct"]].corr().iloc[0, 1])
        lines += [
            "## Signal Quality",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Mean XGB expected return | {exp_mean:+.2f}% |",
            f"| Pearson corr(expected, actual) | {exp_corr:.3f} |",
            "",
        ]

    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def run(config: dict) -> pd.DataFrame:
    """Run intraday backtest. Returns full trade-log DataFrame."""
    best = load_best_features(config)
    if best is None:
        raise FileNotFoundError(
            "best_features.json missing — run 'python main.py hmm' first"
        )
    feature_subset = _filter_24h_features(best["feature_subset"])

    df_common = load_common_dataframe(config)
    X_df      = build_feature_matrix(df_common.copy(), feature_subset)
    sol_close = df_common["SOL_close"].reindex(X_df.index)

    # Evaluation window: last 12 months
    end_ts   = X_df.index[-1]
    start_ts = end_ts - pd.DateOffset(years=_LOOKBACK_YEARS)
    eval_idx = X_df.index[(X_df.index >= start_ts) & (X_df.index <= end_ts)]

    entry_timestamps = [ts for ts in eval_idx if ts.hour == _ENTRY_HOUR]
    logger.info(
        "Intraday backtest: %d potential entry days (%s → %s)",
        len(entry_timestamps),
        eval_idx[0].strftime("%Y-%m-%d"),
        eval_idx[-1].strftime("%Y-%m-%d"),
    )

    records: list[dict] = []
    last_retrain_ts: pd.Timestamp | None = None
    fold_models: tuple | None = None   # (base, q10, q90)

    for entry_ts in entry_timestamps:
        # ── Walk-forward retrain every _RETRAIN_DAYS days ────────────────────
        needs_retrain = (
            last_retrain_ts is None
            or (entry_ts - last_retrain_ts).days >= _RETRAIN_DAYS
        )
        if needs_retrain:
            X_tr = X_df.loc[X_df.index < entry_ts]
            s_tr = sol_close.loc[sol_close.index < entry_ts]
            Xd, yd = _build_train_data(X_tr, s_tr)
            if len(Xd) < 200:
                records.append({"entry_ts": entry_ts, "traded": False,
                                 "skip_reason": "insufficient_training_data"})
                continue
            base_m = _train_model(Xd, yd, n_estimators=_N_EST)
            q10_m  = _train_model(Xd, yd, quantile=0.10, n_estimators=_N_EST)
            q90_m  = _train_model(Xd, yd, quantile=0.90, n_estimators=_N_EST)
            fold_models = (base_m, q10_m, q90_m)
            last_retrain_ts = entry_ts
            logger.info(
                "  [retrain] cutoff=%s  train_rows=%d",
                entry_ts.strftime("%Y-%m-%d %H:%M"),
                len(Xd),
            )

        if fold_models is None:
            continue

        # ── Forecast from entry point ─────────────────────────────────────────
        X_at = X_df.loc[X_df.index <= entry_ts]
        s_at = sol_close.loc[sol_close.index <= entry_ts]
        if X_at.empty or s_at.empty:
            continue
        entry_price = float(s_at.iloc[-1])
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        try:
            base_m, q10_m, q90_m = fold_models
            _, exp, lo, hi = _recursive_forecast(
                base_m, q10_m, q90_m, X_at, s_at, _N_STEPS
            )
        except Exception as exc:
            logger.warning("Forecast failed at %s: %s", entry_ts, exc)
            continue

        # ── Exit selection ────────────────────────────────────────────────────
        sel = _select_exit(entry_price, exp, lo, hi)
        if sel is None:
            logger.debug("  No profitable exit on %s — skip", entry_ts.date())
            records.append({
                "entry_ts": entry_ts,
                "traded": False,
                "skip_reason": "no_upside",
                "entry_price": entry_price,
            })
            continue

        arr_idx, exit_hour, exp_ret, ci_rel = sel
        exit_ts = entry_ts + pd.Timedelta(hours=arr_idx + 1)

        # Actual exit price from SOL parquet
        if exit_ts in sol_close.index:
            exit_price = float(sol_close.loc[exit_ts])
        else:
            future = sol_close.loc[sol_close.index >= exit_ts]
            if future.empty:
                continue
            exit_price = float(future.iloc[0])
            exit_ts    = future.index[0]

        if np.isnan(exit_price) or exit_price <= 0:
            continue

        actual_ret = float(exit_price / entry_price) - 1.0
        net_ret    = actual_ret - _TC_PER_LEG * 2

        records.append({
            "entry_ts":          entry_ts,
            "traded":            True,
            "entry_price":       entry_price,
            "exit_ts":           exit_ts,
            "exit_price":        exit_price,
            "exit_hour_utc":     exit_hour,
            "exp_return_pct":    exp_ret * 100,
            "ci_rel_width":      ci_rel,
            "actual_return_pct": actual_ret * 100,
            "net_return_pct":    net_ret * 100,
            "pnl":               net_ret,
        })
        logger.info(
            "  %s  exit=%02d:00  exp=%+.1f%%  ci=%.2f  actual=%+.1f%%  net=%+.1f%%",
            entry_ts.strftime("%Y-%m-%d"),
            exit_hour,
            exp_ret * 100,
            ci_rel,
            actual_ret * 100,
            net_ret * 100,
        )

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("Intraday: no records generated")
        return df

    traded = df[df["traded"].fillna(False)]
    n_all, n_tr = len(df), len(traded)
    logger.info(
        "Intraday complete: %d days  %d trades (%.0f%% trigger)",
        n_all, n_tr, 100 * n_tr / max(n_all, 1),
    )
    if n_tr:
        cum  = traded["net_return_pct"].sum()
        hit  = 100 * (traded["net_return_pct"] > 0).mean()
        mean = traded["net_return_pct"].mean()
        logger.info(
            "  Cum P&L=%+.2f%%  hit=%.1f%%  mean/trade=%+.3f%%",
            cum, hit, mean,
        )

    # ── Persist results ───────────────────────────────────────────────────────
    out_parquet = processed_dir(config) / "intraday_backtest.parquet"
    df.to_parquet(out_parquet)
    logger.info("Intraday results → %s", out_parquet)

    out_report = processed_dir(config) / "INTRADAY_REPORT.md"
    report_txt = _make_report(traded, df, out_report)
    logger.info("Intraday report → %s", out_report)
    print("\n" + report_txt)

    return df
