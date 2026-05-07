"""BacktestReport — Parquet + 4-panel PNG + BACKTEST_REPORT.md.

Design
------
Template Method: generate() is the fixed output skeleton.

Outputs (all to data/processed/):
  backtest_results.parquet  — raw fold × hour predictions
  backtest_report.png       — 4-panel dashboard
  BACKTEST_REPORT.md        — structured metrics + auto-generated improvement ideas
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .metrics import (
    annualized_return,
    directional_accuracy,
    mae,
    max_drawdown,
    rmse,
    sharpe,
)
from .strategy import RegimeStrategy

logger = logging.getLogger(__name__)

_BG       = "#0f0f0f"
_AX_BG    = "#0f0f0f"
_TICK_CLR = "#bbbbbb"
_GRID_CLR = "#444444"
_SPINE    = "#333333"

_REGIME_COLORS: dict[str, str] = {
    "Strong Bearish": "#922b21",
    "Bearish":        "#e74c3c",
    "Neutral":        "#f39c12",
    "Bullish":        "#2ecc71",
    "Strong Bullish": "#1e8449",
}

# Canonical ordering for consistent bar-chart layout
_REGIME_ORDER = ["Strong Bearish", "Bearish", "Neutral", "Bullish", "Strong Bullish"]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors=_TICK_CLR, labelsize=8)
    ax.grid(axis="y", lw=0.3, alpha=0.4, color=_GRID_CLR)
    ax.grid(axis="x", lw=0.2, alpha=0.2, color=_GRID_CLR)
    ax.spines[["top", "right", "left", "bottom"]].set_color(_SPINE)


def _improvement_ideas(fold_df: pd.DataFrame, strategy_df: pd.DataFrame) -> list[str]:
    ideas: list[str] = []
    h1 = fold_df[fold_df["horizon_h"] == 1]

    # Per-regime directional accuracy below chance
    for regime, grp in h1.groupby("regime"):
        if len(grp) < 5:
            continue
        da = directional_accuracy(grp["actual"].values, grp["xgb_pred"].values)
        if not np.isnan(da) and da < 0.50:
            ideas.append(
                f"XGB Directional Accuracy in '{regime}': {da:.0%} < 50 % "
                f"→ per-Regime-Training oder regime-spezifisches Feature-Set prüfen"
            )

    # Sharpe: strategy vs buy-and-hold
    sp_strat = sharpe(strategy_df["strategy_lr"].values)
    sp_bnh   = sharpe(strategy_df["bnh_lr"].values)
    if not (np.isnan(sp_strat) or np.isnan(sp_bnh)):
        if sp_strat < sp_bnh:
            ideas.append(
                f"Regime-Strategie Sharpe {sp_strat:.2f} < Buy-and-Hold {sp_bnh:.2f} "
                f"→ Positionsgrößen neu kalibrieren (z. B. Neutral → Long statt Cash)"
            )
        if sp_strat < 0:
            ideas.append(
                f"Strategie Sharpe negativ ({sp_strat:.2f}) "
                f"→ Short-Positionen deaktivieren und rein Long / Cash testen"
            )

    # Max drawdown exceeds 20 %
    mdd = max_drawdown(strategy_df["equity_strategy"].values)
    if not np.isnan(mdd) and mdd < -0.20:
        ideas.append(
            f"Max Drawdown {mdd:.0%} → Stop-Loss-Regel (z. B. −15 % Trail) "
            f"für Strong Bearish-Phasen implementieren"
        )

    # RMSE spike detection
    fold_rmse = h1.groupby("fold_id").apply(
        lambda g: rmse(g["actual"].values, g["xgb_pred"].values)
    )
    if len(fold_rmse) > 5:
        mu, sig   = fold_rmse.mean(), fold_rmse.std()
        n_spikes  = int((fold_rmse > mu + 2 * sig).sum())
        if n_spikes > 0:
            ideas.append(
                f"{n_spikes} Folds mit RMSE > μ+2σ (>${mu + 2 * sig:.2f}) "
                f"→ Volatility-Regime als Conditioning-Variable oder Volatility-Scaling prüfen"
            )

    # NP reminder
    ideas.append(
        "NeuralProphet vom Walk-Forward ausgeschlossen (NP+ Shape-Bug + ~55 s/Fold). "
        "Nach Bugfix NP-Folds separat einbeziehen und gegen XGB vergleichen."
    )

    # 3-year backfill recommendation if history < 700 days
    days = (fold_df.index.max() - fold_df.index.min()).days
    n_folds = fold_df["fold_id"].nunique()
    if days < 700:
        ideas.append(
            f"Aktuell nur {days} Tage History ({n_folds} Folds). "
            f"3-Jahr-Backfill (history_days: 1095) für statistisch belastbare Metriken "
            f"empfohlen (~130 Folds)."
        )

    return ideas


def generate(
    fold_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
    config: dict,
) -> None:
    """Save Parquet + 4-panel PNG + BACKTEST_REPORT.md to data/processed/."""
    out_dir = Path(config.get("storage", {}).get("processed_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Parquet ────────────────────────────────────────────────────────────────
    parquet_path = out_dir / "backtest_results.parquet"
    fold_save    = fold_df.copy()
    fold_save.index = fold_save.index.as_unit("us")
    fold_save.to_parquet(parquet_path, engine="pyarrow")
    logger.info("Backtest results → %s", parquet_path)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    h1           = fold_df[fold_df["horizon_h"] == 1]
    overall_rmse = rmse(h1["actual"].values,  h1["xgb_pred"].values)
    overall_mae  = mae(h1["actual"].values,   h1["xgb_pred"].values)
    overall_da   = directional_accuracy(h1["actual"].values, h1["xgb_pred"].values)

    sp_strat = sharpe(strategy_df["strategy_lr"].values)
    sp_bnh   = sharpe(strategy_df["bnh_lr"].values)
    mdd      = max_drawdown(strategy_df["equity_strategy"].values)
    ann_ret  = annualized_return(strategy_df["strategy_lr"].values)
    ann_bnh  = annualized_return(strategy_df["bnh_lr"].values)

    regime_metrics = (
        h1.groupby("regime")
        .apply(
            lambda g: pd.Series(
                {
                    "folds":   int(g["fold_id"].nunique()),
                    "rmse":    rmse(g["actual"].values, g["xgb_pred"].values),
                    "mae":     mae(g["actual"].values, g["xgb_pred"].values),
                    "dir_acc": directional_accuracy(g["actual"].values, g["xgb_pred"].values),
                }
            )
        )
        .reindex([r for r in _REGIME_ORDER if r in h1["regime"].unique()])
    )

    fold_rmse_by_fold = h1.groupby("fold_id").apply(
        lambda g: pd.Series(
            {
                "date": g.index.min(),
                "rmse": rmse(g["actual"].values, g["xgb_pred"].values),
            }
        )
    ).reset_index(drop=True)

    # ── 4-panel PNG ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    fig.patch.set_facecolor(_BG)
    (ax1, ax2), (ax3, ax4) = axes

    # Panel 1 — Cumulative equity: strategy vs buy-and-hold
    _style(ax1)
    ax1.plot(
        strategy_df.index, strategy_df["equity_strategy"],
        color="#3498db", lw=1.5,
        label=f"Regime Strategy  Sharpe={sp_strat:.2f}  Ann.={ann_ret:.1%}",
    )
    ax1.plot(
        strategy_df.index, strategy_df["equity_bnh"],
        color="#9945ff", lw=1.0, alpha=0.7,
        label=f"Buy & Hold  Sharpe={sp_bnh:.2f}  Ann.={ann_bnh:.1%}",
    )
    ax1.set_title("Kumulativer Return — Option B (Regime-Strategie)", color="white",
                  fontsize=10, loc="left")
    ax1.set_ylabel("Equity (start = 1.0)", color=_TICK_CLR, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.legend(fontsize=8, framealpha=0.15, labelcolor="white",
               facecolor="#1a1a1a", edgecolor=_SPINE)

    # Panel 2 — XGB RMSE per fold (time series)
    _style(ax2)
    if not fold_rmse_by_fold.empty:
        ax2.bar(
            fold_rmse_by_fold["date"], fold_rmse_by_fold["rmse"],
            width=pd.Timedelta(days=5), color="#e67e22", alpha=0.8,
        )
    ax2.axhline(overall_rmse, color="#f1c40f", lw=1.0, linestyle="--",
                label=f"Gesamt RMSE ${overall_rmse:.2f}")
    ax2.set_title("XGB RMSE pro Fold — Option A (Walk-Forward)", color="white",
                  fontsize=10, loc="left")
    ax2.set_ylabel("RMSE (USD)", color=_TICK_CLR, fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax2.legend(fontsize=8, framealpha=0.15, labelcolor="white",
               facecolor="#1a1a1a", edgecolor=_SPINE)

    # Panel 3 — Directional accuracy per regime (bar chart)
    _style(ax3)
    if not regime_metrics.empty and "dir_acc" in regime_metrics.columns:
        regimes = list(regime_metrics.index)
        da_vals = regime_metrics["dir_acc"].fillna(0).values
        colors  = [_REGIME_COLORS.get(r, "#888888") for r in regimes]
        ax3.bar(regimes, da_vals, color=colors, alpha=0.85)
        ax3.axhline(0.5, color="#f1c40f", lw=1.0, linestyle="--",
                    label="50 % (Zufallsniveau)")
        ax3.set_ylim(0, 1)
        ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax3.set_title("Directional Accuracy per Regime — Option A", color="white",
                  fontsize=10, loc="left")
    ax3.set_ylabel("Directional Accuracy", color=_TICK_CLR, fontsize=9)
    ax3.tick_params(axis="x", rotation=20)
    ax3.legend(fontsize=8, framealpha=0.15, labelcolor="white",
               facecolor="#1a1a1a", edgecolor=_SPINE)

    # Panel 4 — RMSE / MAE per regime (grouped bar chart)
    _style(ax4)
    if not regime_metrics.empty:
        x   = np.arange(len(regime_metrics))
        w   = 0.35
        clr = [_REGIME_COLORS.get(r, "#888888") for r in regime_metrics.index]
        ax4.bar(x - w / 2, regime_metrics["rmse"].fillna(0), width=w,
                color=clr, alpha=0.85, label="RMSE")
        ax4.bar(x + w / 2, regime_metrics["mae"].fillna(0), width=w,
                color=clr, alpha=0.45, label="MAE")
        ax4.set_xticks(x)
        ax4.set_xticklabels(list(regime_metrics.index), rotation=20,
                            fontsize=8, color=_TICK_CLR)
    ax4.set_title("RMSE / MAE per Regime — Option A", color="white", fontsize=10, loc="left")
    ax4.set_ylabel("USD", color=_TICK_CLR, fontsize=9)
    ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax4.legend(fontsize=8, framealpha=0.15, labelcolor="white",
               facecolor="#1a1a1a", edgecolor=_SPINE)

    fig.tight_layout(pad=2.0)
    png_path = out_dir / "backtest_report.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Backtest plot → %s", png_path)

    # ── BACKTEST_REPORT.md ────────────────────────────────────────────────────
    ideas    = _improvement_ideas(fold_df, strategy_df)
    start_ts = fold_df.index.min().strftime("%Y-%m-%d")
    end_ts   = fold_df.index.max().strftime("%Y-%m-%d")
    n_folds  = int(fold_df["fold_id"].nunique())
    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    bt_cfg   = config.get("backtest", {})

    regime_dist = h1.groupby("regime").size()

    lines: list[str] = [
        f"# Backtest Report — {now_str}",
        "",
        "## 1. Datenbasis",
        "",
        "| Merkmal | Wert |",
        "|---|---|",
        f"| Zeitraum | {start_ts} → {end_ts} |",
        f"| Folds | {n_folds} (Step {bt_cfg.get('step_days', 7)}d, Horizont {bt_cfg.get('horizon_hours', 24)}h) |",
        f"| Mindest-Trainingsfenster | {bt_cfg.get('min_train_days', 30)} Tage |",
        f"| Modelle | XGB walk-forward (A) + HMM Regime-Strategie (B) |",
        f"| NeuralProphet | ausgeschlossen — NP+ Shape-Bug + ~55 s/Fold |",
        "",
        "### Regime-Verteilung an Fold-Startpunkten",
        "",
        "| Regime | Folds |",
        "|---|---|",
    ]
    for regime in _REGIME_ORDER:
        if regime in regime_dist:
            lines.append(f"| {regime} | {int(regime_dist[regime])} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Forecast-Accuracy — XGB Walk-Forward (Option A)",
        "",
        "> Oracle-Evaluation (echte Features an jedem Schritt, keine Rekursion).",
        "> Produktions-Forecast (rekursiv) ist schlechter — dies ist die obere Schranke.",
        "",
        "| Metrik | Gesamt |",
        "|---|---|",
        f"| RMSE | ${overall_rmse:.2f} |",
        f"| MAE | ${overall_mae:.2f} |",
        f"| Directional Accuracy | {overall_da:.1%} |",
        "",
        "### Per-Regime",
        "",
        "| Regime | RMSE | MAE | Dir. Accuracy | Folds |",
        "|---|---|---|---|---|",
    ]
    for regime, row in regime_metrics.iterrows():
        da_str = f"{row['dir_acc']:.1%}" if not np.isnan(row["dir_acc"]) else "n/a"
        lines.append(
            f"| {regime} | ${row['rmse']:.2f} | ${row['mae']:.2f} | {da_str} | {int(row['folds'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Regime-Strategie-Performance (Option B)",
        "",
        "> **Hinweis:** HMM-Modell auf dem gesamten Datensatz trainiert → leichter",
        "> Look-ahead-Bias bei Regime-Labels. Echte Performance wird schlechter sein.",
        "> Positionsmap: Strong Bullish=+1.0, Bullish=+0.5, Neutral=0, Bearish=−0.5, Strong Bearish=−1.0.",
        "> Keine Transaktionskosten modelliert.",
        "",
        "| Metrik | Strategie | Buy-and-Hold |",
        "|---|---|---|",
        f"| Ann. Return | {ann_ret:.1%} | {ann_bnh:.1%} |",
        f"| Sharpe | {sp_strat:.2f} | {sp_bnh:.2f} |",
        f"| Max Drawdown | {mdd:.1%} | — |",
        "",
        "---",
        "",
        "## 4. Verbesserungsideen",
        "",
    ]
    for i, idea in enumerate(ideas, 1):
        lines.append(f"{i}. {idea}")

    lines += [
        "",
        "---",
        f"*Generiert: {now_str} | Daten: {start_ts} → {end_ts} | Folds: {n_folds}*",
    ]

    md_path = out_dir / "BACKTEST_REPORT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Backtest report → %s", md_path)
