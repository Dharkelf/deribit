"""Prophet Day-Selection Backtest.

Design
------
Template Method: run() is the fixed pipeline skeleton.

Strategy design period : ≤ 2023  (realized moves profile "big move days",
                                   analogous to HMM feature selection on history)
Backtest period        : 2024 onwards  (frozen feature rule + NP walk-forward)

Pipeline
--------
1. Offline analysis (≤ 2023, no NP training)
   a. Realized SOL moves: 23:00 UTC day D → 15/16/17/18 UTC day D+1
   b. Correlation decay: features at t−{0,1,2,3}h vs |move|; best-lag map frozen
   c. Sell-hour baseline: mean return per sell hour on big-move days (≤ 2023)

2. Candidate selection (2024+)
   Feature rule applied at 23:00 UTC only — no realized future prices

3. Adaptive sampling
   Two NP calibration folds → avg_fold_sec
   n_folds = floor(3600 × 0.85 / avg_fold_sec)
   Stratified by year × HMM regime; within strata sorted by feature score

4. Walk-forward backtest (per sampled day D)
   NP trained on data ≤ 23:00 UTC day D (strict cutoff)
   sell_hour = argmin(CI_width) for h ∈ {15,16,17,18} UTC  ← no leakage
   Stop-loss: hourly SOL price check from 00:00 D+1; −10 % from entry → exit
   Gate: NP predicted_move > 1 % AND direction = Long

5. Strategy comparison + PROPHET_DAY_REPORT.md
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from src.hmm.features import build_feature_matrix, load_common_dataframe
from src.hmm.model import GaussianHMMModel
from src.hmm.optimizer import load_best_features
from src.hmm.predict_prophet import predict_backtest_fold
from src.utils.paths import models_dir, processed_dir

logger = logging.getLogger(__name__)

_DESIGN_END = pd.Timestamp("2023-12-31 23:00:00", tz="UTC")
_BT_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
_SELL_HOURS = [15, 16, 17, 18]
_STOP_LOSS = 0.10
_MIN_MOVE_PCT = 0.01
_BUDGET_SEC = 3600.0
_OVERHEAD = 0.85
_N_CAL = 2
_DECAY_LAGS = [0, 1, 2, 3]


# ── Phase 1: design-period analysis ──────────────────────────────────────────


def build_day_returns(
    sol: pd.Series,
    sell_hours: list[int] = _SELL_HOURS,
) -> pd.DataFrame:
    """Realized SOL log-returns: 23:00 UTC day D → each sell_hour UTC day D+1.

    Returns DataFrame indexed by date (day D), columns:
      entry_ts, sol_entry, move_h{h} for each h, move_best (max |move|)
    """
    entry_ts_index = sol.index[sol.index.hour == 23]
    sol_dict: dict[pd.Timestamp, float] = sol.to_dict()
    records: list[dict[str, Any]] = []
    for ts in entry_ts_index:
        entry_price = sol_dict.get(ts, np.nan)
        if np.isnan(entry_price) or entry_price <= 0:
            continue
        row: dict[str, Any] = {"entry_ts": ts, "sol_entry": float(entry_price)}
        for h in sell_hours:
            exit_ts = ts + pd.Timedelta(hours=h + 1)
            ep = sol_dict.get(exit_ts, np.nan)
            row[f"move_h{h}"] = (
                float(np.log(ep / entry_price))
                if not np.isnan(ep) and ep > 0
                else np.nan
            )
        records.append(row)

    df = pd.DataFrame(records)
    df.index = pd.Index([r["entry_ts"].date() for r in records])
    move_cols = [f"move_h{h}" for h in sell_hours]
    df["move_best"] = df[move_cols].abs().max(axis=1)
    return df


def analyze_predictor_decay(
    feat_df: pd.DataFrame,
    day_returns: pd.DataFrame,
    design_end: pd.Timestamp = _DESIGN_END,
    lags: list[int] = _DECAY_LAGS,
) -> pd.DataFrame:
    """Correlation of each feature at lags 0–3 h before 23:00 with |move_best|.

    Returns DataFrame sorted by abs_corr descending.
    """
    feat = feat_df.loc[feat_df.index <= design_end]
    entry_mask = feat.index.hour == 23
    entry_times = feat.index[entry_mask]
    entry_dates = pd.Index([t.date() for t in entry_times])
    move = day_returns.loc[day_returns.index <= design_end.date(), "move_best"].dropna()

    records: list[dict[str, Any]] = []
    for feat_name in feat.columns:
        for lag in lags:
            lagged = feat[feat_name].shift(lag)
            vals = lagged.reindex(entry_times).values
            feat_series = pd.Series(vals, index=entry_dates)
            common = feat_series.index.intersection(move.index)
            if len(common) < 20:
                continue
            corr = float(feat_series.loc[common].corr(move.loc[common]))
            if not np.isnan(corr):
                records.append(
                    {
                        "feature": feat_name,
                        "lag_h": lag,
                        "corr": corr,
                        "abs_corr": abs(corr),
                    }
                )

    if not records:
        return pd.DataFrame(columns=["feature", "lag_h", "corr", "abs_corr"])
    return (
        pd.DataFrame(records)
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )


def _sell_hour_baseline(
    day_returns: pd.DataFrame,
    design_end: pd.Timestamp = _DESIGN_END,
    top_quantile: float = 0.25,
    sell_hours: list[int] = _SELL_HOURS,
) -> pd.Series:
    """Mean return per sell_hour on big-move days in the design period."""
    dr = day_returns.loc[day_returns.index <= design_end.date()].dropna(
        subset=["move_best"]
    )
    threshold = dr["move_best"].quantile(1 - top_quantile)
    big_move = dr[dr["move_best"] >= threshold]
    return pd.Series(
        {h: float(big_move[f"move_h{h}"].mean()) for h in sell_hours},
        name="mean_return_on_big_move_days",
    )


def _get_semantic_labels(
    hmm_model: GaussianHMMModel,
    X_df: pd.DataFrame,
    raw_labels: np.ndarray,
) -> pd.Series:
    sol_idx = list(X_df.columns).index("SOL_log_return")
    means = hmm_model._model.means_[:, sol_idx]
    k = hmm_model.n_components
    order = np.argsort(means)
    _LABEL_MAP: dict[int, dict[int, str]] = {
        2: {1: "Bearish", 2: "Bullish"},
        3: {1: "Bearish", 2: "Neutral", 3: "Bullish"},
        4: {
            1: "Strong Bearish",
            2: "Bearish",
            3: "Bullish",
            4: "Strong Bullish",
        },
        5: {
            1: "Strong Bearish",
            2: "Bearish",
            3: "Neutral",
            4: "Bullish",
            5: "Strong Bullish",
        },
    }
    names = _LABEL_MAP.get(k, {r: f"Regime {r}" for r in range(1, k + 1)})
    id_to_label = {
        int(state): names.get(rank, f"Regime {rank}")
        for rank, state in enumerate(order, start=1)
    }
    return pd.Series(
        [id_to_label.get(int(s), "Neutral") for s in raw_labels],
        index=X_df.index,
    )


# ── Phase 2: candidate selection ─────────────────────────────────────────────


def select_candidates(
    feat_df: pd.DataFrame,
    label_series: pd.Series,
    vol_median: float,
) -> list[pd.Timestamp]:
    """Apply frozen feature rule (design-period) to 2024+ data.

    Returns list of 23:00 UTC timestamps eligible as backtest entry points.
    Filters: weekday, SOL_vol_168h > design-period median, BTC/VIX momentum
    signal, HMM regime ∈ {Bullish, Strong Bullish}.
    """
    mask = (feat_df.index >= _BT_START) & (feat_df.index.hour == 23)
    candidates: list[pd.Timestamp] = []
    for ts in feat_df.index[mask]:
        if ts.dayofweek >= 5:
            continue
        row = feat_df.loc[ts]
        label = str(label_series.get(ts, "Neutral"))
        vol_ok = float(row.get("SOL_vol_168h", 0.0)) > vol_median
        momentum_ok = (
            float(row.get("BTC_momentum", 0.0)) > 0.0
            or float(row.get("VIX_zscore", 0.0)) > 0.0
        )
        regime_ok = label in ("Bullish", "Strong Bullish")
        if vol_ok and momentum_ok and regime_ok:
            candidates.append(ts)

    logger.info(
        "Candidate days 2024+: %d  (vol_median=%.6f)", len(candidates), vol_median
    )
    return candidates


def _feature_score(
    feat_df: pd.DataFrame,
    candidates: list[pd.Timestamp],
) -> pd.Series:
    """Composite score from feature quantile ranks (no future data)."""
    sub = feat_df.reindex(candidates)
    score = pd.Series(0.0, index=sub.index)
    for col in ("SOL_vol_168h", "VIX_zscore"):
        if col in sub.columns:
            score += sub[col].rank(pct=True).fillna(0.5)
    if "BTC_momentum" in sub.columns:
        score += sub["BTC_momentum"].abs().rank(pct=True).fillna(0.5)
    return score


def sample_days(
    candidates: list[pd.Timestamp],
    label_series: pd.Series,
    feat_df: pd.DataFrame,
    n: int,
    seed: int = 42,
) -> list[pd.Timestamp]:
    """Stratified sample by year × HMM regime; within strata by feature score."""
    if not candidates:
        return []
    score = _feature_score(feat_df, candidates)
    df = pd.DataFrame(
        {
            "ts": candidates,
            "year": [t.year for t in candidates],
            "regime": [str(label_series.get(t, "Neutral")) for t in candidates],
            "score": [float(score.get(t, 0.0)) for t in candidates],
        }
    )
    df = df.sort_values("score", ascending=False)

    strata = df.groupby(["year", "regime"])
    n_strata = max(1, len(strata))
    base = max(1, n // n_strata)

    sampled: list[pd.Timestamp] = []
    for _, grp in strata:
        k = min(base, len(grp))
        sampled.extend(grp.head(k)["ts"].tolist())

    # fill up to n if needed
    used = set(sampled)
    remaining = [t for t in df["ts"].tolist() if t not in used]
    if len(sampled) < n and remaining:
        extra = remaining[: n - len(sampled)]
        sampled.extend(extra)

    sampled = sorted(set(sampled))[:n]
    logger.info("Sampled %d/%d candidates for NP backtest", len(sampled), n)
    return sampled


# ── Phase 3: calibration ─────────────────────────────────────────────────────


def calibrate_fold_time(
    config: dict,
    feature_subset: list[str],
    df_common: pd.DataFrame,
    candidates: list[pd.Timestamp],
    n_cal: int = _N_CAL,
) -> float:
    """Time n_cal NP folds on the first candidates; return avg seconds/fold."""
    times: list[float] = []
    for cutoff_ts in candidates[:n_cal]:
        t0 = time.perf_counter()
        try:
            predict_backtest_fold(config, cutoff_ts, feature_subset, df_common)
        except Exception as exc:
            logger.warning("Calibration fold %s failed: %s", cutoff_ts.date(), exc)
            continue
        times.append(time.perf_counter() - t0)

    if not times:
        fallback = 90.0
        logger.warning("All calibration folds failed; using fallback %.0f s", fallback)
        return fallback

    avg = float(np.mean(times))
    n_budget = int(_BUDGET_SEC * _OVERHEAD / avg)
    logger.info(
        "Calibration: %d folds, avg=%.1f s → budget allows ~%d folds",
        len(times),
        avg,
        n_budget,
    )
    return avg


# ── Phase 4: fold helpers ─────────────────────────────────────────────────────


def find_sell_hour_by_ci(
    np_exp_48: np.ndarray,
    np_lo_48: np.ndarray,
    np_hi_48: np.ndarray,
    sell_hours: list[int] = _SELL_HOURS,
) -> tuple[int, float]:
    """Return (sell_hour, ci_width_pct) with tightest CI among sell_hours.

    Step index mapping: sell_hour h UTC (D+1) → np_exp_48[h]
    (step 0 = 00:00 D+1, step h = h:00 D+1 when cutoff = 23:00 D).
    No leakage: uses only NP forecast produced at entry time.
    """
    best_h, best_ci = sell_hours[0], float("inf")
    for h in sell_hours:
        if h >= len(np_exp_48):
            continue
        exp = float(np_exp_48[h])
        if np.isnan(exp) or exp <= 0:
            continue
        width = float(np_hi_48[h] - np_lo_48[h])
        ci = width / exp
        if ci < best_ci:
            best_ci = ci
            best_h = h
    return best_h, best_ci


def apply_stop_loss(
    sol_dict: dict[pd.Timestamp, float],
    entry_ts: pd.Timestamp,
    sell_ts: pd.Timestamp,
    stop_pct: float = _STOP_LOSS,
) -> tuple[pd.Timestamp, float, bool, int | None]:
    """Walk hourly prices entry+1h … sell_ts; exit if price drops > stop_pct.

    Returns (exit_ts, exit_price, was_stopped, stop_hour_utc).
    Real-time execution semantics — not leakage.
    """
    entry_price = sol_dict.get(entry_ts, np.nan)
    if np.isnan(entry_price) or entry_price <= 0:
        return entry_ts, np.nan, False, None

    check_times = pd.date_range(
        start=entry_ts + pd.Timedelta(hours=1),
        end=sell_ts,
        freq="1h",
    )
    for ts in check_times:
        price = sol_dict.get(ts, np.nan)
        if np.isnan(price) or price <= 0:
            continue
        if price / entry_price - 1.0 < -stop_pct:
            return ts, float(price), True, int(ts.hour)

    exit_price = sol_dict.get(sell_ts, np.nan)
    return sell_ts, float(exit_price), False, None


# ── Phase 4: main backtest loop ───────────────────────────────────────────────


def backtest_prophet_days(
    config: dict,
    sampled: list[pd.Timestamp],
    feature_subset: list[str],
    df_common: pd.DataFrame,
    feat_df: pd.DataFrame,
    label_series: pd.Series,
    sell_hours: list[int] = _SELL_HOURS,
    stop_loss_pct: float = _STOP_LOSS,
    min_move_pct: float = _MIN_MOVE_PCT,
) -> pd.DataFrame:
    """Walk-forward NP backtest; returns one row per fold."""
    sol = df_common["SOL_close"]
    if sol.index.tz is None:
        sol.index = sol.index.tz_localize("UTC")
    sol_dict: dict[pd.Timestamp, float] = {
        ts: float(v) for ts, v in sol.items() if not np.isnan(v)
    }

    records: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for i, cutoff_ts in enumerate(sampled):
        elapsed = time.perf_counter() - t_start
        if elapsed > _BUDGET_SEC:
            logger.warning(
                "Budget exhausted after %d/%d folds (%.0f s)", i, len(sampled), elapsed
            )
            break

        logger.info(
            "[%d/%d] NP fold %s  (elapsed %.0f s)",
            i + 1,
            len(sampled),
            cutoff_ts.date(),
            elapsed,
        )

        try:
            fold = predict_backtest_fold(config, cutoff_ts, feature_subset, df_common)
        except Exception as exc:
            logger.warning("Fold %s failed: %s", cutoff_ts.date(), exc)
            continue

        sol_last = float(fold["sol_last"])
        np_exp_48: np.ndarray = fold["np_exp_48"]
        np_lo_48: np.ndarray = fold["np_lo_48"]
        np_hi_48: np.ndarray = fold["np_hi_48"]

        sell_hour, ci_width_pct = find_sell_hour_by_ci(
            np_exp_48, np_lo_48, np_hi_48, sell_hours
        )
        sell_ts = cutoff_ts + pd.Timedelta(hours=sell_hour + 1)

        predicted_price = (
            float(np_exp_48[sell_hour]) if sell_hour < len(np_exp_48) else np.nan
        )
        np_move_pct = (
            (predicted_price - sol_last) / sol_last
            if sol_last > 0 and not np.isnan(predicted_price)
            else np.nan
        )
        gated_in = not np.isnan(np_move_pct) and np_move_pct > min_move_pct

        exit_ts, exit_price, was_stopped, stop_hour = apply_stop_loss(
            sol_dict, cutoff_ts, sell_ts, stop_loss_pct
        )

        sell_price_bnh = sol_dict.get(sell_ts, np.nan)
        bnh_return = (
            float(np.log(sell_price_bnh / sol_last))
            if sol_last > 0 and not np.isnan(sell_price_bnh) and sell_price_bnh > 0
            else np.nan
        )
        actual_return = (
            float(np.log(exit_price / sol_last))
            if sol_last > 0 and not np.isnan(exit_price) and exit_price > 0
            else np.nan
        )

        feat_snap = feat_df.loc[cutoff_ts] if cutoff_ts in feat_df.index else None

        def _fv(col: str) -> float:
            return (
                float(feat_snap[col])
                if feat_snap is not None and col in feat_df.columns
                else np.nan
            )

        records.append(
            {
                "date": cutoff_ts.date(),
                "cutoff_ts": cutoff_ts,
                "year": int(cutoff_ts.year),
                "weekday": cutoff_ts.day_name(),
                "regime": str(label_series.get(cutoff_ts, "Neutral")),
                "sell_hour": int(sell_hour),
                "sell_ts": sell_ts,
                "ci_width_pct": float(ci_width_pct),
                "np_predicted": float(predicted_price),
                "np_move_pct": float(np_move_pct)
                if not np.isnan(np_move_pct)
                else np.nan,
                "gated_in": bool(gated_in),
                "exit_ts": exit_ts,
                "exit_price": float(exit_price),
                "actual_return": float(actual_return),
                "bnh_return": float(bnh_return),
                "was_stopped": bool(was_stopped),
                "stop_hour": int(stop_hour) if stop_hour is not None else None,
                "sol_last": float(sol_last),
                "in_data_rmse": float(fold["in_data_rmse"]),
                "in_data_adj_r2": float(fold["in_data_adj_r2"]),
                "SOL_vol_168h": _fv("SOL_vol_168h"),
                "BTC_momentum": _fv("BTC_momentum"),
                "VIX_zscore": _fv("VIX_zscore"),
            }
        )

    total = time.perf_counter() - t_start
    logger.info(
        "Backtest done: %d folds in %.1f s (%.1f s/fold avg)",
        len(records),
        total,
        total / max(len(records), 1),
    )
    return pd.DataFrame(records)


# ── Phase 5: comparison + report ─────────────────────────────────────────────


def _strategy_metrics(df: pd.DataFrame, ret_col: str, label: str) -> dict[str, Any]:
    r = df[ret_col].dropna()
    if len(r) == 0:
        return {
            "strategy": label,
            "n": 0,
            "mean_ret": np.nan,
            "cum_ret": np.nan,
            "hit_rate": np.nan,
            "sharpe": np.nan,
        }
    cum = float(np.exp(r.sum()) - 1.0)
    hit = float((r > 0).mean())
    sh = float(r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan
    return {
        "strategy": label,
        "n": int(len(r)),
        "mean_ret": float(r.mean()),
        "cum_ret": cum,
        "hit_rate": hit,
        "sharpe": sh,
    }


def compare_strategies(results: pd.DataFrame) -> pd.DataFrame:
    """Four strategies compared overall and per year."""
    gated = results[results["gated_in"]]
    rows = [
        _strategy_metrics(gated, "actual_return", "NP-Gated + Stop-Loss"),
        _strategy_metrics(gated, "bnh_return", "NP-Gated (no stop)"),
        _strategy_metrics(results, "bnh_return", "All candidates (no gate)"),
    ]
    return pd.DataFrame(rows).set_index("strategy")


def _write_report(
    results: pd.DataFrame,
    comparison: pd.DataFrame,
    decay_df: pd.DataFrame,
    sell_hour_baseline: pd.Series,
    n_candidates: int,
    out_dir: "Path",  # noqa: F821
) -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        f"# PROPHET_DAY_REPORT — {now}",
        "",
        "## 1. Design-Period Analysis (≤ 2023)",
        "",
        "### Top-10 Features by Correlation with |Move| (all lags)",
        "",
        "| Feature | Lag h | Corr | |Corr| |",
        "|---|---|---|---|",
    ]
    for _, row in decay_df.head(10).iterrows():
        lines.append(
            f"| {row['feature']} | {int(row['lag_h'])} | {row['corr']:.4f} | {row['abs_corr']:.4f} |"
        )

    lines += [
        "",
        "### Mean Return per Sell Hour on Big-Move Days (≤ 2023, top 25%)",
        "",
        "| Sell Hour UTC | Mean Log-Return |",
        "|---|---|",
    ]
    for h, v in sell_hour_baseline.items():
        lines.append(f"| {h}:00 | {v:.6f} |")

    lines += [
        "",
        "## 2. Candidate Selection (2024+)",
        "",
        f"- Total candidates: {n_candidates}",
        f"- Sampled for NP backtest: {len(results)}",
        "",
        "### Regime breakdown of sampled days",
        "",
        "| Regime | Count |",
        "|---|---|",
    ]
    for regime, cnt in results["regime"].value_counts().items():
        lines.append(f"| {regime} | {cnt} |")

    lines += [
        "",
        "### Sell-hour distribution (dynamic CI selection)",
        "",
        "| Sell Hour UTC | Count | % |",
        "|---|---|---|",
    ]
    for h, cnt in results["sell_hour"].value_counts().sort_index().items():
        pct = 100.0 * cnt / max(len(results), 1)
        lines.append(f"| {h}:00 | {cnt} | {pct:.1f}% |")

    lines += [
        "",
        f"- Stop-loss triggered: {results['was_stopped'].sum()} / {len(results)} folds",
        f"- Gated-in trades: {results['gated_in'].sum()} / {len(results)} folds",
        "",
        "## 3. Strategy Comparison",
        "",
        "| Strategy | N | Mean Return | Cum Return | Hit Rate | Sharpe |",
        "|---|---|---|---|---|---|",
    ]
    for strat, row in comparison.iterrows():
        lines.append(
            f"| {strat} | {int(row['n'])} "
            f"| {row['mean_ret']:.4f} "
            f"| {row['cum_ret']:.2%} "
            f"| {row['hit_rate']:.1%} "
            f"| {row['sharpe']:.2f} |"
        )

    lines += ["", "### Per-Year Breakdown (NP-Gated + Stop-Loss)", ""]
    gated = results[results["gated_in"]].copy()
    if not gated.empty:
        lines += [
            "| Year | N | Mean Return | Cum Return | Hit Rate | Stop-Loss % |",
            "|---|---|---|---|---|---|",
        ]
        for year, grp in gated.groupby("year"):
            r = grp["actual_return"].dropna()
            hit = float((r > 0).mean()) if len(r) else np.nan
            stop_pct = float(grp["was_stopped"].mean()) * 100
            cum = float(np.exp(r.sum()) - 1.0) if len(r) else np.nan
            lines.append(
                f"| {year} | {len(r)} | {r.mean():.4f} | {cum:.2%} | {hit:.1%} | {stop_pct:.1f}% |"
            )

    lines += [
        "",
        "## 4. Per-Fold Detail (first 30 rows)",
        "",
        "| Date | Regime | Sell h | CI% | NP Move% | Gated | Return | Stopped |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in results.head(30).iterrows():
        lines.append(
            f"| {row['date']} | {row['regime']} | {row['sell_hour']}:00"
            f" | {row['ci_width_pct']:.1%}"
            f" | {row['np_move_pct']:.1%}"
            f" | {'✓' if row['gated_in'] else '✗'}"
            f" | {row['actual_return']:.3f}"
            f" | {'✓' if row['was_stopped'] else '✗'} |"
        )

    report_path = out_dir / "PROPHET_DAY_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report → %s", report_path)

    parquet_path = out_dir / "prophet_day_results.parquet"
    save = results.copy()
    save.index = pd.RangeIndex(len(save))
    for col in ("cutoff_ts", "sell_ts", "exit_ts"):
        if col in save.columns:
            save[col] = save[col].astype("datetime64[us, UTC]")
    save.to_parquet(parquet_path, engine="pyarrow")
    logger.info("Results → %s", parquet_path)


# ── Entry point ───────────────────────────────────────────────────────────────


def run(config: dict) -> None:
    """Full Prophet Day-Selection Backtest pipeline."""
    logger.info("=== Prophet Day-Selection Backtest ===")
    out_dir = processed_dir(config)

    # ── Load artefacts ────────────────────────────────────────────────────────
    best = load_best_features(config)
    if best is None:
        raise FileNotFoundError(
            "best_features.json missing; run 'python main.py hmm' first"
        )
    feature_subset: list[str] = best["feature_subset"]
    n_comp: int = int(best["n_components"])

    model_path = models_dir(config) / f"best_hmm_k{n_comp}.pkl"
    hmm_model = GaussianHMMModel.load(model_path)

    logger.info("Loading full dataset …")
    df_common = load_common_dataframe(config)
    feat_df = build_feature_matrix(df_common.copy(), feature_subset)

    raw_labels = hmm_model.predict(feat_df.values)
    label_series = _get_semantic_labels(hmm_model, feat_df, raw_labels)

    sol = df_common["SOL_close"]
    if sol.index.tz is None:
        sol.index = sol.index.tz_localize("UTC")

    # ── Phase 1: design-period analysis ──────────────────────────────────────
    logger.info("Phase 1: design-period analysis (≤ 2023) …")
    day_returns = build_day_returns(sol)

    decay_df = analyze_predictor_decay(feat_df, day_returns)
    logger.info(
        "Top predictor: %s", decay_df.iloc[0].to_dict() if not decay_df.empty else "n/a"
    )

    sell_hour_base = _sell_hour_baseline(day_returns)
    logger.info("Sell-hour baseline (big-move days): %s", sell_hour_base.to_dict())

    # Vol threshold: design-period median at 23:00 entries
    design_vols = feat_df.loc[
        (feat_df.index <= _DESIGN_END) & (feat_df.index.hour == 23), "SOL_vol_168h"
    ].dropna()
    vol_median = float(design_vols.median()) if not design_vols.empty else 0.0

    # ── Phase 2: candidates ────────────────────────────────────────────────────
    logger.info("Phase 2: selecting candidates (2024+) …")
    candidates = select_candidates(feat_df, label_series, vol_median)
    if not candidates:
        logger.error("No candidates found — check data coverage for 2024+")
        return

    # ── Phase 3: calibrate + sample ───────────────────────────────────────────
    logger.info("Phase 3: calibrating NP fold time …")
    avg_fold_sec = calibrate_fold_time(config, feature_subset, df_common, candidates)
    n_folds = max(3, int(_BUDGET_SEC * _OVERHEAD / avg_fold_sec))
    logger.info("Budget: %d folds", n_folds)

    # Exclude calibration folds from sampling to avoid repeated training
    sampled = sample_days(candidates[_N_CAL:], label_series, feat_df, n_folds)
    if not sampled:
        logger.error("No days sampled — not enough candidates after calibration skip")
        return

    # ── Phase 4: backtest ─────────────────────────────────────────────────────
    logger.info("Phase 4: NP walk-forward backtest (%d folds) …", len(sampled))
    results = backtest_prophet_days(
        config, sampled, feature_subset, df_common, feat_df, label_series
    )
    if results.empty:
        logger.error("No fold results — check data and model")
        return

    # ── Phase 5: report ───────────────────────────────────────────────────────
    comparison = compare_strategies(results)
    logger.info("\n%s", comparison.to_string())

    _write_report(
        results, comparison, decay_df, sell_hour_base, len(candidates), out_dir
    )
    logger.info("=== Prophet Day Backtest complete ===")
