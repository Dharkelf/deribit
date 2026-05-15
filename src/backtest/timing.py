"""Timing analysis for the hourly regime strategy.

Design
------
Standalone analysis module.  Recomputes the hourly regime strategy (no XGB)
and answers: *when* does the edge come from?

Four analyses:
  1. Heatmap UTC hour × weekday — mean strategy_lr per cell (all regimes +
     Bullish/Strong-Bullish only)
  2. 3-hour block contribution — mean cumulative log-return per 3-hour block
     of the day, showing which blocks carry the most alpha
  3. Weekend long-hold simulation — buy Fri 19:00 UTC, sell Mon 07:00 UTC;
     compare equity curve vs rest-of-week
  4. Year-by-year robustness — top-5 UTC hours ranked by mean strategy_lr;
     shows whether timing is stable across market cycles

Outputs (data/processed/):
  timing_analysis.png   — 4-panel figure
  TIMING_REPORT.md      — markdown summary with tables
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.hmm.features import build_feature_matrix, load_common_dataframe
from src.hmm.model import GaussianHMMModel
from src.hmm.optimizer import load_best_features
from src.hmm.visualize import _assign_regime_colors_and_labels
from src.utils.paths import models_dir, processed_dir

from .strategy import RegimeStrategy

logger = logging.getLogger(__name__)

_BG = "#0f0f0f"
_AX_BG = "#0f0f0f"
_TICK_CLR = "#bbbbbb"
_GRID_CLR = "#444444"
_SPINE = "#333333"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors=_TICK_CLR, labelsize=8)
    ax.grid(axis="y", lw=0.3, alpha=0.4, color=_GRID_CLR)
    ax.grid(axis="x", lw=0.2, alpha=0.2, color=_GRID_CLR)
    ax.spines[["top", "right", "left", "bottom"]].set_color(_SPINE)


# ── Build hourly strategy ─────────────────────────────────────────────────────


def _build_hourly_strategy(config: dict) -> pd.DataFrame:
    """Load HMM model, assign regimes, apply hourly strategy (no stop, no window)."""
    best = load_best_features(config)
    if best is None:
        raise FileNotFoundError(
            "best_features.json missing — run 'python main.py hmm' first"
        )

    df_common = load_common_dataframe(config)
    X_hmm = build_feature_matrix(df_common.copy(), best["feature_subset"])

    cutoff = config.get(
        "_cutoff", pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(hours=1)
    )
    X_hmm = X_hmm.loc[X_hmm.index <= cutoff]

    mpath = models_dir(config) / f"best_hmm_k{best['n_components']}.pkl"
    model = GaussianHMMModel.load(mpath)
    logger.info("HMM model loaded ← %s", mpath)

    labels = model.predict(X_hmm.values)
    regime_info = _assign_regime_colors_and_labels(model, X_hmm, labels)
    label_series = pd.Series(
        [regime_info[int(lbl)]["label"] for lbl in labels],
        index=X_hmm.index,
        name="regime",
    )

    sol_close = df_common["SOL_close"].reindex(X_hmm.index)
    sol_lr = np.log(sol_close / sol_close.shift(1)).dropna()

    strategy_df = RegimeStrategy().apply(sol_lr, label_series)
    logger.info("Hourly strategy computed: %d rows", len(strategy_df))
    return strategy_df


# ── Analysis helpers ──────────────────────────────────────────────────────────


def _hour_weekday_heatmap(
    strategy_df: pd.DataFrame,
    bullish_only: bool = False,
) -> pd.DataFrame:
    """Mean strategy_lr by (weekday, UTC hour).

    Returns DataFrame[7 rows × 24 columns] — weekday 0=Mon, 6=Sun.
    """
    df = strategy_df.copy()
    if bullish_only:
        df = df[df["regime"].isin(["Bullish", "Strong Bullish"])]
    df = df[df["position"] != 0]  # exclude flat hours from mean

    df = df.assign(
        hour=df.index.hour,
        weekday=df.index.weekday,
    )
    pivot = df.pivot_table(
        values="strategy_lr",
        index="weekday",
        columns="hour",
        aggfunc="mean",
        fill_value=np.nan,
    )
    # Ensure full 7×24 grid
    pivot = pivot.reindex(index=range(7), columns=range(24), fill_value=np.nan)
    return pivot


def _block_contribution(strategy_df: pd.DataFrame, block_h: int = 3) -> pd.DataFrame:
    """Mean cumulative strategy log-return per N-hour block of the day.

    Returns DataFrame with columns: block_start, mean_lr, n_hours.
    """
    df = strategy_df.assign(hour=strategy_df.index.hour)
    df["block"] = (df["hour"] // block_h) * block_h

    agg = (
        df.groupby("block")["strategy_lr"]
        .agg(mean_lr="mean", n_hours="count")
        .reset_index()
        .rename(columns={"block": "block_start"})
    )
    return agg


def _weekend_hold_stats(strategy_df: pd.DataFrame) -> dict:
    """Compare Fri 19:00 – Mon 07:00 UTC vs rest-of-week in the hourly strategy.

    Returns dict with keys: weekend_lr, weekday_lr, weekend_equity, weekday_equity,
    weekend_sharpe, weekday_sharpe, n_weekend_h, n_weekday_h.
    """
    df = strategy_df.copy()
    # Weekend window: Friday hour ≥ 19 OR Saturday OR Sunday OR Monday hour < 8
    wd = df.index.weekday  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    hr = df.index.hour
    is_weekend = (
        ((wd == 4) & (hr >= 19))  # Fri 19:00+
        | (wd == 5)  # All day Sat
        | (wd == 6)  # All day Sun
        | ((wd == 0) & (hr < 8))  # Mon 00:00–07:59
    )

    wk_lr = df.loc[is_weekend, "strategy_lr"].values
    wd_lr = df.loc[~is_weekend, "strategy_lr"].values

    def _sharpe(lr: np.ndarray) -> float:
        s = float(np.std(lr, ddof=1))
        if len(lr) < 2 or s == 0:
            return float("nan")
        return float(lr.mean() / s * np.sqrt(8760))

    return {
        "weekend_lr": wk_lr,
        "weekday_lr": wd_lr,
        "weekend_equity": float(np.exp(wk_lr.sum()) - 1),
        "weekday_equity": float(np.exp(wd_lr.sum()) - 1),
        "weekend_sharpe": _sharpe(wk_lr),
        "weekday_sharpe": _sharpe(wd_lr),
        "n_weekend_h": int(is_weekend.sum()),
        "n_weekday_h": int((~is_weekend).sum()),
    }


def _year_robustness(strategy_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Top-N UTC hours by mean strategy_lr per year.

    Returns DataFrame[n_years × top_n] with UTC hour values.
    """
    df = strategy_df.assign(hour=strategy_df.index.hour, year=strategy_df.index.year)
    df = df[df["position"] != 0]

    records = []
    for year, grp in df.groupby("year"):
        ranked = (
            grp.groupby("hour")["strategy_lr"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )
        row: dict = {"year": year}
        for rank, (hour, val) in enumerate(ranked.items(), 1):
            row[f"rank{rank}_hour"] = int(hour)
            row[f"rank{rank}_lr"] = float(val)
        records.append(row)

    return pd.DataFrame(records).set_index("year")


# ── Figure ────────────────────────────────────────────────────────────────────


def _plot(
    strategy_df: pd.DataFrame,
    heatmap_all: pd.DataFrame,
    heatmap_bull: pd.DataFrame,
    block_df: pd.DataFrame,
    weekend_stats: dict,
    robustness_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(20, 16), facecolor=_BG)
    fig.suptitle(
        "Timing Analysis — UTC Hour × Weekday Edge",
        color=_TICK_CLR,
        fontsize=14,
        y=0.98,
    )

    gs = fig.add_gridspec(
        2, 2, hspace=0.45, wspace=0.35, left=0.07, right=0.97, top=0.94, bottom=0.06
    )

    ax_hm_all = fig.add_subplot(gs[0, 0])
    ax_hm_bull = fig.add_subplot(gs[0, 1])
    ax_block = fig.add_subplot(gs[1, 0])
    ax_wknd = fig.add_subplot(gs[1, 1])

    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def _heatmap(ax: plt.Axes, pivot: pd.DataFrame, title: str) -> None:
        data = pivot.values.astype(float)
        vmax = (
            float(np.nanpercentile(np.abs(data[~np.isnan(data)]), 95))
            if not np.all(np.isnan(data))
            else 1e-4
        )
        vmax = max(vmax, 1e-6)
        im = ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            cmap="RdYlGn",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(title, color=_TICK_CLR, fontsize=9)
        ax.set_xlabel("UTC Hour", color=_TICK_CLR, fontsize=8)
        ax.set_ylabel("Weekday", color=_TICK_CLR, fontsize=8)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([str(h) for h in range(0, 24, 2)], color=_TICK_CLR, fontsize=7)
        ax.set_yticks(range(7))
        ax.set_yticklabels(weekday_labels, color=_TICK_CLR, fontsize=7)
        ax.spines[["top", "right", "left", "bottom"]].set_color(_SPINE)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(colors=_TICK_CLR, labelsize=7)
        cbar.set_label("Mean log-return", color=_TICK_CLR, fontsize=7)

    _heatmap(ax_hm_all, heatmap_all, "Hourly edge — all active positions")
    _heatmap(ax_hm_bull, heatmap_bull, "Hourly edge — Bullish/Strong-Bullish only")

    # ── Panel 3: 3h block contribution ───────────────────────────────────────
    _style(ax_block)
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in block_df["mean_lr"]]
    ax_block.bar(
        block_df["block_start"], block_df["mean_lr"] * 1e4, color=colors, width=2.5
    )
    ax_block.set_title(
        "Mean strategy log-return by 3-hour UTC block (×10⁻⁴)",
        color=_TICK_CLR,
        fontsize=9,
    )
    ax_block.set_xlabel("UTC block start hour", color=_TICK_CLR, fontsize=8)
    ax_block.set_ylabel("Mean log-return ×10⁻⁴", color=_TICK_CLR, fontsize=8)
    ax_block.set_xticks(block_df["block_start"])
    ax_block.set_xticklabels(
        [f"{h:02d}:00" for h in block_df["block_start"]],
        color=_TICK_CLR,
        fontsize=7,
        rotation=30,
    )
    ax_block.axhline(0, color=_TICK_CLR, lw=0.5, alpha=0.5)

    # ── Panel 4: Weekend vs weekday cumulative equity ─────────────────────────
    _style(ax_wknd)
    wk_eq = np.exp(np.cumsum(weekend_stats["weekend_lr"]))
    wd_eq = np.exp(np.cumsum(weekend_stats["weekday_lr"]))
    ax_wknd.plot(
        wk_eq,
        color="#9b59b6",
        lw=1.2,
        label=f"Weekend (Fri 19–Mon 07) n={weekend_stats['n_weekend_h']}h",
    )
    ax_wknd.plot(
        wd_eq,
        color="#3498db",
        lw=1.2,
        label=f"Weekday (rest) n={weekend_stats['n_weekday_h']}h",
    )
    ax_wknd.set_title(
        "Cumulative equity: weekend vs weekday hours", color=_TICK_CLR, fontsize=9
    )
    ax_wknd.set_xlabel("Hour index", color=_TICK_CLR, fontsize=8)
    ax_wknd.set_ylabel("Equity (start=1)", color=_TICK_CLR, fontsize=8)
    ax_wknd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}×"))
    ax_wknd.legend(
        fontsize=7, facecolor="#1a1a1a", labelcolor=_TICK_CLR, framealpha=0.7
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    logger.info("Timing figure saved → %s", out_path)


# ── Markdown report ───────────────────────────────────────────────────────────


def _markdown_report(
    strategy_df: pd.DataFrame,
    block_df: pd.DataFrame,
    weekend_stats: dict,
    robustness_df: pd.DataFrame,
    heatmap_all: pd.DataFrame,
    out_path: Path,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dr = strategy_df.index
    date_range = f"{dr.min().strftime('%Y-%m-%d')} → {dr.max().strftime('%Y-%m-%d')}"

    lines: list[str] = [
        f"# Timing Analysis Report — {now}",
        "",
        f"Period: {date_range}  |  Total hours: {len(strategy_df):,}",
        "",
        "---",
        "",
        "## 1. Best UTC Hours (all active positions)",
        "",
    ]

    # Top hours by mean strategy_lr
    df_pos = strategy_df[strategy_df["position"] != 0].copy()
    df_pos = df_pos.assign(hour=df_pos.index.hour)
    hour_stats = (
        df_pos.groupby("hour")["strategy_lr"]
        .agg(mean_lr="mean", n="count")
        .sort_values("mean_lr", ascending=False)
        .head(10)
    )
    lines.append("| UTC Hour | Mean log-return | N |")
    lines.append("|---|---|---|")
    for hour, row in hour_stats.iterrows():
        lines.append(
            f"| {int(hour):02d}:00 | {row['mean_lr']:.6f} | {int(row['n']):,} |"
        )
    lines.append("")

    lines += [
        "## 2. 3-Hour Block Analysis",
        "",
        "| Block (UTC) | Mean log-return (×10⁻⁴) | N hours |",
        "|---|---|---|",
    ]
    for _, row in block_df.iterrows():
        lines.append(
            f"| {int(row['block_start']):02d}:00–{int(row['block_start']) + 3:02d}:00"
            f" | {row['mean_lr'] * 1e4:.3f}"
            f" | {int(row['n_hours']):,} |"
        )
    lines.append("")

    lines += [
        "## 3. Weekend vs Weekday",
        "",
        "> Weekend = Fri 19:00 – Mon 07:59 UTC",
        "",
        "| Period | Cum. return | Ann. Sharpe | Hours |",
        "|---|---|---|---|",
        f"| Weekend | {weekend_stats['weekend_equity']:+.1%} | {weekend_stats['weekend_sharpe']:.2f} | {weekend_stats['n_weekend_h']:,} |",
        f"| Weekday | {weekend_stats['weekday_equity']:+.1%} | {weekend_stats['weekday_sharpe']:.2f} | {weekend_stats['n_weekday_h']:,} |",
        "",
    ]

    lines += [
        "## 4. Year-by-Year Robustness (Top-5 UTC Hours)",
        "",
        "| Year | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |",
        "|---|---|---|---|---|---|",
    ]
    for year, row in robustness_df.iterrows():
        cols = []
        for rank in range(1, 6):
            h = row.get(f"rank{rank}_hour", float("nan"))
            lr = row.get(f"rank{rank}_lr", float("nan"))
            if np.isnan(h):
                cols.append("—")
            else:
                cols.append(f"{int(h):02d}:00 ({lr:.4f})")
        lines.append(f"| {year} | {' | '.join(cols)} |")
    lines.append("")

    # Best hour × weekday cell
    flat = heatmap_all.stack().dropna()
    if len(flat) > 0:
        best_idx = flat.idxmax()
        best_wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][best_idx[0]]
        lines += [
            "## 5. Best Single Cell (Weekday × Hour)",
            "",
            f"Best edge: **{best_wd} {best_idx[1]:02d}:00 UTC**  "
            f"mean log-return = `{flat.max():.6f}`",
            "",
        ]

    lines += [
        "---",
        f"*Generated: {now}*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Timing report saved → %s", out_path)


# ── Public entry point ────────────────────────────────────────────────────────


def run(config: dict) -> None:
    """Run timing analysis and write outputs to data/processed/."""
    logger.info("Timing analysis — start")

    strategy_df = _build_hourly_strategy(config)
    heatmap_all = _hour_weekday_heatmap(strategy_df, bullish_only=False)
    heatmap_bull = _hour_weekday_heatmap(strategy_df, bullish_only=True)
    block_df = _block_contribution(strategy_df, block_h=3)
    weekend_stats = _weekend_hold_stats(strategy_df)
    robustness_df = _year_robustness(strategy_df, top_n=5)

    logger.info(
        "Weekend equity=%.1f%%  sharpe=%.2f | Weekday equity=%.1f%%  sharpe=%.2f",
        weekend_stats["weekend_equity"] * 100,
        weekend_stats["weekend_sharpe"],
        weekend_stats["weekday_equity"] * 100,
        weekend_stats["weekday_sharpe"],
    )

    out_dir = processed_dir(config)
    _plot(
        strategy_df,
        heatmap_all,
        heatmap_bull,
        block_df,
        weekend_stats,
        robustness_df,
        out_dir / "timing_analysis.png",
    )
    _markdown_report(
        strategy_df,
        block_df,
        weekend_stats,
        robustness_df,
        heatmap_all,
        out_dir / "TIMING_REPORT.md",
    )
    logger.info("Timing analysis — done")
