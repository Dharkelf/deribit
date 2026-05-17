"""Intraday SOL long/short strategy — NeuralProphet walk-forward backtest.

Simulates running the full NP pipeline (identical to predict_prophet.py)
on a random sample of ~52 days from the last 12 months.  Each fold uses
cutoff = previous day 23:00 UTC — exactly the standard pipeline cutoff —
so the 24h forecast (yhat1..yhat24) covers the next day 00:00–23:00 UTC.

Forecast mapping  (cutoff = prev-day 23:00 UTC, n_forecasts=24)
--------------------------------------------------------------
  yhat8  → next day 07:00 UTC  (+8 h)  ← entry hour (calibration check)
  yhat12 → next day 11:00 UTC  (+12 h) ← exit candidate 1
  yhat13 → next day 12:00 UTC  (+13 h) ← exit candidate 2
  yhat14 → next day 13:00 UTC  (+14 h) ← exit candidate 3
  yhat24 → next day 23:00 UTC  (+24 h) ← daily direction anchor

Strategy (gates applied in order — all must pass)
--------------------------------------------------
Gate 1 — Daily direction: yhat24 vs entry_price sets direction (LONG/SHORT).
          Only trade LONG if yhat24 > entry; SHORT if yhat24 < entry.

Gate 2 — Duration consistency: at least _MIN_EXIT_AGREEMENT of the 3 exit
          yhats (11,12,13h) must agree with the daily direction.
          LONG : yhat > entry_price; SHORT: yhat < entry_price.
          Replaces the fixed-% magnitude threshold — consistency across
          forecast horizons is a cleaner quality measure than calibrated %.

Gate 3 — Entry calibration: |yhat8 - actual_07h| / actual_07h ≤ 4%.
          Skip if model is off-calibration at entry time.

Exit   : among direction-consistent exit yhats, pick nearest to their mean
         predicted price (same logic as before, no change).

TC     : 0.01/leg → 0.02 round-trip.
Invest : 1.0 per trade.
Period : Last 12 months of available data; ~52 folds sampled randomly (seed=42).
Model  : Full NP pipeline — n_lags=168, n_forecasts=24, epochs=60, HMM
         regime indicator regressors, HMM-selected features,
         quantiles=[0.1, 0.9].  Single seed (seed=0) for 1-hour budget.
Extra  : prev_eve_pct = SOL return 21:00 (prev day) → 07:00 entry (actual prices),
         logged as indicator column.
         daily_dir_pct = yhat24 / entry − 1 (NP full-day expectation, signed).
         exit_agreement = count of exit yhats consistent with direction (0–3).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.hmm.features import build_feature_matrix, load_common_dataframe
from src.hmm.optimizer import load_best_features
from src.hmm.predict_prophet import (
    _add_regime_proba_to_df,
    _build_future_df,
    _build_np_df,
    _train_model,
)
from src.utils.paths import processed_dir

logger = logging.getLogger(__name__)

_TC_PER_LEG:         float     = 0.01
_ENTRY_HOUR:         int       = 7        # 07:00 UTC — EU session open
_EXIT_HOURS:         list[int] = [11, 12, 13]   # 11–13:00 UTC
# 1-based yhat indices from cutoff 23:00 UTC: +12h→11:00  +13h→12:00  +14h→13:00
_EXIT_YHATS:         list[int] = [12, 13, 14]
_ENTRY_YHAT:         int       = 8    # +8h → 07:00 UTC (entry calibration)
_DAILY_ANCHOR_YHAT:  int       = 24   # +24h → 23:00 UTC (full-day direction)
_MAX_ENTRY_DEV:      float     = 0.04  # skip if |yhat8 − actual| / actual > 4%
_MIN_EXIT_AGREEMENT: int       = 2     # min exit yhats consistent with daily direction
_LOOKBACK_YEARS:     int       = 1
_N_SAMPLES:          int       = 52
_SAMPLE_SEED:        int       = 42
_SEED:               int       = 0


# ─────────────────────────────────────────────────────────────────────────────
# Forecast extraction
# ─────────────────────────────────────────────────────────────────────────────


def _get_forecasts(
    model: object,
    np_df: pd.DataFrame,
    np_feature_subset: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    """Run NP predict and extract exit + anchor forecasts.

    Returns (exp, lo, hi, yhat_entry, yhat_daily) or None on failure.
      exp/lo/hi    : shape (len(_EXIT_YHATS),) — exit-hour price forecasts.
      yhat_entry   : predicted price at 07:00 UTC (calibration check).
      yhat_daily   : predicted price at 23:00 UTC (full-day direction anchor).
    """
    try:
        future_df = _build_future_df(np_df, model, np_feature_subset)
        forecast  = model.predict(future_df)
    except Exception as exc:
        logger.warning("NP predict failed: %s", exc)
        return None

    last_ds   = np_df["ds"].iloc[-1]
    last_rows = forecast[forecast["ds"] == last_ds]
    if last_rows.empty:
        logger.warning("last_known_ds %s not in NP forecast", last_ds)
        return None
    row = last_rows.iloc[0]

    yhat_cols = sorted(
        [c for c in forecast.columns if c.startswith("yhat") and c[4:].isdigit()],
        key=lambda c: int(c[4:]),
    )
    lo_cols = sorted(
        [c for c in forecast.columns if c.endswith(" 10.0%")],
        key=lambda c: int(c.split("yhat")[1].split(" ")[0]),
    )
    hi_cols = sorted(
        [c for c in forecast.columns if c.endswith(" 90.0%")],
        key=lambda c: int(c.split("yhat")[1].split(" ")[0]),
    )

    def _get(cols: list[str], one_based: int) -> float:
        i = one_based - 1
        return float(row.get(cols[i], np.nan)) if i < len(cols) else np.nan

    exp         = np.array([_get(yhat_cols, h) for h in _EXIT_YHATS])
    lo          = np.array([_get(lo_cols,   h) for h in _EXIT_YHATS]) if lo_cols else exp.copy()
    hi          = np.array([_get(hi_cols,   h) for h in _EXIT_YHATS]) if hi_cols else exp.copy()
    yhat_entry  = _get(yhat_cols, _ENTRY_YHAT)
    yhat_daily  = _get(yhat_cols, _DAILY_ANCHOR_YHAT)
    return exp, lo, hi, yhat_entry, yhat_daily


# ─────────────────────────────────────────────────────────────────────────────
# Exit selection
# ─────────────────────────────────────────────────────────────────────────────


def _select_exit(
    entry_price: float,
    exp: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    direction: str,
) -> tuple[int, float, float, int] | None:
    """Pick exit nearest to mean among direction-consistent candidates.

    LONG : exit yhats above entry_price.
    SHORT: exit yhats below entry_price (and > 0).
    Among qualifying candidates pick closest to their mean predicted price.

    Returns (exit_hour_utc, exp_return, ci_rel_width, n_consistent) or None.
    n_consistent = number of exit yhats that agreed with direction.
    """
    if direction == "long":
        valid = [(i, h) for i, h in enumerate(_EXIT_HOURS)
                 if not np.isnan(exp[i]) and exp[i] > entry_price]
    else:
        valid = [(i, h) for i, h in enumerate(_EXIT_HOURS)
                 if not np.isnan(exp[i]) and 0 < exp[i] < entry_price]
    if not valid:
        return None

    mean_exp = float(np.mean([exp[i] for i, _ in valid]))
    best_i, best_h = min(valid, key=lambda ih: abs(exp[ih[0]] - mean_exp))

    exp_ret = float(exp[best_i] / entry_price) - 1.0
    ci_abs  = max(float(hi[best_i] - lo[best_i]), 0.0)
    ci_rel  = ci_abs / max(float(exp[best_i]), 1e-6)
    return best_h, exp_ret, ci_rel, len(valid)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────


def _make_report(traded: pd.DataFrame, all_days: pd.DataFrame, out_path: Path) -> str:
    n_all    = len(all_days)
    n_traded = len(traded)
    trigger  = 100.0 * n_traded / max(n_all, 1)

    cum_pnl  = float(traded["net_return_pct"].sum()) if n_traded else 0.0
    hit_rate = float((traded["net_return_pct"] > 0).mean()) * 100 if n_traded else 0.0
    mean_net = float(traded["net_return_pct"].mean()) if n_traded else 0.0
    std_net  = float(traded["net_return_pct"].std(ddof=1)) if n_traded > 1 else 0.0

    days_span = max((all_days["entry_ts"].max() - all_days["entry_ts"].min()).days, 1)
    trades_per_year = n_traded / days_span * 365
    sharpe = (mean_net / std_net * np.sqrt(trades_per_year)
              if std_net > 0 else 0.0)

    lines: list[str] = [
        "# Intraday NP Backtest Report",
        "",
        f"Period  : {all_days['entry_ts'].min().strftime('%Y-%m-%d')} → "
        f"{all_days['entry_ts'].max().strftime('%Y-%m-%d')}",
        "Strategy: Buy 07:00 UTC (EU open) · Exit {11,12,13}:00 UTC",
        f"Gate 1  : Daily direction — yhat24 vs entry (LONG if yhat24 > entry)",
        f"Gate 2  : Duration consistency — ≥{_MIN_EXIT_AGREEMENT}/3 exit yhats agree with daily direction",
        f"Gate 3  : Entry calibration — |yhat8 − actual| / actual ≤ {_MAX_ENTRY_DEV*100:.0f}%",
        f"Model   : Full NP pipeline · n_lags=168 · epochs=60 · HMM regressors · "
        f"quantiles=[0.1,0.9] · seed={_SEED} · {_N_SAMPLES} random folds (rng={_SAMPLE_SEED})",
        "Cutoff  : previous day 23:00 UTC",
        "Eve     : SOL return 21:00 (prev day) → 07:00 entry (actual prices)",
        "TC      : 0.01/leg  (0.02 round-trip)  ·  Invest 1.0/trade",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Samples evaluated | {n_all} |",
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
        lines += ["## Exit-Hour Distribution", "",
                  "| Exit UTC | Trades | % | Mean net | Hit rate |",
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
        t2 = traded.copy()
        t2["month"] = t2["entry_ts"].dt.to_period("M")
        for month, grp in t2.groupby("month"):
            lines.append(
                f"| {month} | {len(grp)} "
                f"| {100*(grp['net_return_pct']>0).mean():.0f}% "
                f"| {grp['net_return_pct'].sum():+.2f}% |"
            )
        lines.append("")

        lines += ["## Trade Log", "",
                  "| Date | Dir | Exit | Eve 21→07 | DayDir | Agree | Dev | Exp | Actual | Net |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in traded.sort_values("entry_ts").iterrows():
            eve     = r.get("prev_eve_pct", np.nan)
            eve_str = f"{float(eve):+.2f}%" if not np.isnan(float(eve)) else "n/a"
            ddir    = r.get("daily_dir_pct", np.nan)
            ddir_str= f"{float(ddir):+.1f}%" if not np.isnan(float(ddir)) else "n/a"
            agree   = r.get("exit_agreement", np.nan)
            agree_str = f"{int(agree)}/3" if not np.isnan(float(agree)) else "n/a"
            dev     = r.get("entry_dev_pct", np.nan)
            dev_str = f"{float(dev):.1f}%" if not np.isnan(float(dev)) else "n/a"
            lines.append(
                f"| {r['entry_ts'].strftime('%Y-%m-%d')} "
                f"| {r.get('direction', '?').upper()} "
                f"| {int(r['exit_hour_utc']):02d}:00 "
                f"| {eve_str} "
                f"| {ddir_str} "
                f"| {agree_str} "
                f"| {dev_str} "
                f"| {r['exp_return_pct']:+.1f}% "
                f"| {r['actual_return_pct']:+.1f}% "
                f"| {r['net_return_pct']:+.2f}% |"
            )
        lines.append("")

        if "prev_eve_pct" in traded.columns:
            eve_vals = traded["prev_eve_pct"].dropna()
            if len(eve_vals) > 1:
                pos_eve = traded[traded["prev_eve_pct"] > 0]
                neg_eve = traded[traded["prev_eve_pct"] <= 0]
                eve_corr = float(
                    traded[["prev_eve_pct", "actual_return_pct"]].corr().iloc[0, 1]
                )
                lines += [
                    "## Evening Indicator Analysis (21:00 prev day → 07:00 entry)",
                    "",
                    "| Subset | Trades | Hit rate | Mean net P&L |",
                    "|---|---|---|---|",
                    f"| Eve positive (SOL rose 21→07) | {len(pos_eve)} "
                    f"| {100*(pos_eve['net_return_pct']>0).mean():.0f}% "
                    f"| {pos_eve['net_return_pct'].mean():+.3f}% |"
                    if len(pos_eve) else "| Eve positive | 0 | n/a | n/a |",
                    f"| Eve negative (SOL fell 21→07) | {len(neg_eve)} "
                    f"| {100*(neg_eve['net_return_pct']>0).mean():.0f}% "
                    f"| {neg_eve['net_return_pct'].mean():+.3f}% |"
                    if len(neg_eve) else "| Eve negative | 0 | n/a | n/a |",
                    f"| Pearson corr(eve_pct, actual_ret) | {eve_corr:.3f} | | |",
                    "",
                ]

        if "exit_agreement" in traded.columns:
            lines += ["## Duration Consistency Breakdown", "",
                      "| Exit agreements | Trades | Hit rate | Mean net |",
                      "|---|---|---|---|"]
            for n_agree in sorted(traded["exit_agreement"].dropna().unique()):
                sub = traded[traded["exit_agreement"] == n_agree]
                lines.append(
                    f"| {int(n_agree)}/3 | {len(sub)} "
                    f"| {100*(sub['net_return_pct']>0).mean():.0f}% "
                    f"| {sub['net_return_pct'].mean():+.3f}% |"
                )
            lines.append("")

        if "direction" in traded.columns:
            longs  = traded[traded["direction"] == "long"]
            shorts = traded[traded["direction"] == "short"]
            lines += ["## Direction Breakdown", "",
                      "| Direction | Trades | Hit rate | Mean net | Cum P&L |",
                      "|---|---|---|---|---|"]
            for label, grp in [("LONG", longs), ("SHORT", shorts)]:
                if grp.empty:
                    continue
                lines.append(
                    f"| {label} | {len(grp)} "
                    f"| {100*(grp['net_return_pct']>0).mean():.0f}% "
                    f"| {grp['net_return_pct'].mean():+.3f}% "
                    f"| {grp['net_return_pct'].sum():+.2f}% |"
                )
            lines.append("")

        exp_mean = float(traded["exp_return_pct"].mean())
        corr_val = float(traded[["exp_return_pct", "actual_return_pct"]].corr().iloc[0, 1])
        lines += [
            "## Signal Quality",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Mean NP expected return (selected exit) | {exp_mean:+.2f}% |",
            f"| Pearson corr(expected, actual) | {corr_val:.3f} |",
            "",
        ]

    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def run(config: dict) -> pd.DataFrame:
    """Run NeuralProphet intraday walk-forward backtest. Returns trade-log DataFrame."""
    from neuralprophet import set_random_seed  # noqa: PLC0415

    best = load_best_features(config)
    if best is None:
        raise FileNotFoundError(
            "best_features.json missing — run 'python main.py hmm' first"
        )
    feature_subset: list[str] = best["feature_subset"]

    df_common = load_common_dataframe(config)
    X_df_full = build_feature_matrix(df_common.copy(), feature_subset)
    sol_close = df_common["SOL_close"].reindex(X_df_full.index)

    end_ts   = X_df_full.index[-1]
    start_ts = end_ts - pd.DateOffset(years=_LOOKBACK_YEARS)

    all_07h = [
        ts for ts in sol_close.index
        if ts >= start_ts and ts <= end_ts and ts.hour == _ENTRY_HOUR
    ]

    rng = np.random.default_rng(_SAMPLE_SEED)
    n_avail = len(all_07h)
    size = min(_N_SAMPLES, n_avail)
    idx = sorted(rng.choice(n_avail, size=size, replace=False), reverse=True)
    entry_timestamps = [all_07h[i] for i in idx]

    logger.info(
        "Intraday NP backtest (EU session): %d samples (%s → %s)",
        len(entry_timestamps),
        entry_timestamps[-1].strftime("%Y-%m-%d") if entry_timestamps else "n/a",
        entry_timestamps[0].strftime("%Y-%m-%d") if entry_timestamps else "n/a",
    )

    set_random_seed(_SEED)
    records: list[dict] = []

    for fold_i, entry_ts in enumerate(entry_timestamps):
        cutoff_ts = entry_ts.normalize() - pd.Timedelta(hours=1)

        X_cap   = X_df_full.loc[X_df_full.index <= cutoff_ts].copy()
        sol_cap = sol_close.loc[sol_close.index <= cutoff_ts]

        if len(X_cap) < 200:
            records.append({
                "entry_ts": entry_ts, "traded": False,
                "skip_reason": "insufficient_data",
            })
            continue

        regime_cols       = _add_regime_proba_to_df(X_cap, config, best)
        np_feature_subset = feature_subset + regime_cols

        np_df = _build_np_df(X_cap, sol_cap, np_feature_subset)

        logger.info(
            "  [%d/%d] cutoff=%s  rows=%d  regressors=%d",
            fold_i + 1, len(entry_timestamps),
            cutoff_ts.strftime("%Y-%m-%d %H:%M"),
            len(np_df),
            len(np_feature_subset),
        )

        try:
            model = _train_model(np_df, np_feature_subset, config)
        except Exception as exc:
            logger.warning("NP training failed at %s: %s", cutoff_ts, exc)
            continue

        result = _get_forecasts(model, np_df, np_feature_subset)
        if result is None:
            continue
        exp, lo, hi, yhat_entry_price, yhat_daily_price = result

        if entry_ts not in sol_close.index:
            continue
        entry_price = float(sol_close.loc[entry_ts])
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        # Gate 3: entry calibration
        if not np.isnan(yhat_entry_price) and yhat_entry_price > 0:
            entry_dev = abs(yhat_entry_price - entry_price) / entry_price
            if entry_dev > _MAX_ENTRY_DEV:
                logger.debug(
                    "  %s  yhat8=$%.2f vs actual=$%.2f (dev=%.1f%%) — skip calibration",
                    entry_ts.date(), yhat_entry_price, entry_price, entry_dev * 100,
                )
                records.append({
                    "entry_ts": entry_ts, "traded": False,
                    "skip_reason": "entry_calibration",
                    "entry_price": entry_price,
                    "yhat_entry_price": yhat_entry_price,
                    "entry_dev_pct": entry_dev * 100,
                })
                continue
        else:
            entry_dev = np.nan

        # Gate 1: daily direction from yhat24
        if np.isnan(yhat_daily_price) or yhat_daily_price <= 0:
            records.append({"entry_ts": entry_ts, "traded": False,
                            "skip_reason": "no_daily_anchor"})
            continue
        daily_dir_pct = (yhat_daily_price / entry_price - 1.0) * 100
        direction = "long" if yhat_daily_price > entry_price else "short"

        # Gate 2: duration consistency — count exit yhats agreeing with direction
        if direction == "long":
            n_consistent = int(np.sum(
                (~np.isnan(exp)) & (exp > entry_price)
            ))
        else:
            n_consistent = int(np.sum(
                (~np.isnan(exp)) & (exp > 0) & (exp < entry_price)
            ))

        if n_consistent < _MIN_EXIT_AGREEMENT:
            records.append({
                "entry_ts": entry_ts, "traded": False,
                "skip_reason": "duration_inconsistent",
                "direction": direction,
                "daily_dir_pct": daily_dir_pct,
                "exit_agreement": n_consistent,
            })
            continue

        # Evening indicator: 21:00 prev day → 07:00 entry
        ts_21h = entry_ts - pd.Timedelta(hours=10)
        if ts_21h in sol_close.index:
            p21 = float(sol_close.loc[ts_21h])
            prev_eve_pct = (entry_price / p21 - 1.0) * 100 if p21 > 0 else np.nan
        else:
            prev_eve_pct = np.nan

        sel = _select_exit(entry_price, exp, lo, hi, direction)
        if sel is None:
            records.append({
                "entry_ts": entry_ts, "traded": False,
                "skip_reason": "no_consistent_exit",
                "entry_price": entry_price,
                "prev_eve_pct": prev_eve_pct,
                "direction": direction,
                "daily_dir_pct": daily_dir_pct,
            })
            continue

        exit_hour, exp_ret, ci_rel, exit_agreement = sel

        exit_ts = entry_ts.normalize() + pd.Timedelta(hours=exit_hour)
        if exit_ts in sol_close.index:
            exit_price = float(sol_close.loc[exit_ts])
        else:
            future_prices = sol_close.loc[sol_close.index >= exit_ts]
            if future_prices.empty:
                continue
            exit_price = float(future_prices.iloc[0])
            exit_ts    = future_prices.index[0]

        if np.isnan(exit_price) or exit_price <= 0:
            continue

        raw_ret    = float(exit_price / entry_price) - 1.0
        actual_ret = raw_ret if direction == "long" else -raw_ret
        net_ret    = actual_ret - _TC_PER_LEG * 2

        records.append({
            "entry_ts":          entry_ts,
            "traded":            True,
            "direction":         direction,
            "entry_price":       entry_price,
            "yhat_entry_price":  yhat_entry_price,
            "entry_dev_pct":     entry_dev * 100 if not np.isnan(entry_dev) else np.nan,
            "daily_dir_pct":     daily_dir_pct,
            "exit_agreement":    exit_agreement,
            "exit_ts":           exit_ts,
            "exit_price":        exit_price,
            "exit_hour_utc":     exit_hour,
            "prev_eve_pct":      prev_eve_pct,
            "exp_return_pct":    exp_ret * 100,
            "ci_rel_width":      ci_rel,
            "actual_return_pct": actual_ret * 100,
            "net_return_pct":    net_ret * 100,
            "pnl":               net_ret,
        })
        eve_str  = f"{prev_eve_pct:+.2f}%" if not np.isnan(prev_eve_pct) else "n/a"
        dev_str  = f"{entry_dev*100:.1f}%" if not np.isnan(entry_dev) else "n/a"
        logger.info(
            "  %s  %s  exit=%02d:00  eve=%s  dev=%s  day=%+.1f%%  agree=%d/3  exp=%+.1f%%  ci=%.2f  actual=%+.1f%%  net=%+.1f%%",
            entry_ts.strftime("%Y-%m-%d"), direction.upper(),
            exit_hour, eve_str, dev_str,
            daily_dir_pct, exit_agreement,
            exp_ret * 100, ci_rel,
            actual_ret * 100, net_ret * 100,
        )

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("Intraday NP: no records generated")
        return df

    traded = df[df["traded"].fillna(False)]
    n_all  = len(df)
    n_tr   = len(traded)
    logger.info(
        "Intraday NP complete: %d samples  %d trades (%.0f%% trigger)",
        n_all, n_tr, 100 * n_tr / max(n_all, 1),
    )
    if n_tr:
        logger.info(
            "  Cum P&L=%+.2f%%  hit=%.1f%%  mean/trade=%+.3f%%",
            traded["net_return_pct"].sum(),
            100 * (traded["net_return_pct"] > 0).mean(),
            traded["net_return_pct"].mean(),
        )

    out_parquet = processed_dir(config) / "intraday_np_backtest.parquet"
    df.to_parquet(out_parquet)
    logger.info("NP intraday results → %s", out_parquet)

    out_report = processed_dir(config) / "INTRADAY_NP_REPORT.md"
    report_txt = _make_report(traded, df, out_report)
    logger.info("NP intraday report → %s", out_report)
    print("\n" + report_txt)

    return df
