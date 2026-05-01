"""Visual inspection of HMM regime models on SOL price — last 365 days.

Standalone script — not imported by any other module.

Shows the top 3–5 Optuna-selected model configurations, each in its own
subplot, with the SOL/USD close price over the last year and a coloured
regime background (axvspan) to make regime transitions visible.

Usage:
    python -m src.collector.inspect_opt_regime
    python src/collector/inspect_opt_regime.py
"""

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "config" / "settings.yaml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# Deferred imports so the module stays importable even without hmmlearn installed
from src.hmm.features import build_feature_matrix, load_common_dataframe  # noqa: E402
from src.hmm.model import build_model  # noqa: E402
from src.hmm.optimizer import run_optimization, top_n_results  # noqa: E402

_REGIME_PALETTE = [
    "#4c72b0",  # blue     — state 0
    "#dd8452",  # orange   — state 1
    "#55a868",  # green    — state 2
    "#c44e52",  # red      — state 3
    "#8172b2",  # purple   — state 4
]

_LOOKBACK_DAYS = 365


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _shade_regimes(
    ax: plt.Axes,
    timestamps: pd.DatetimeIndex,
    labels: np.ndarray,
    n_components: int,
    alpha: float = 0.20,
) -> list:
    """Fill contiguous regime blocks. Returns list of Patch handles for legend."""
    from matplotlib.patches import Patch  # local import to keep top-level lean

    handles = []
    for r in range(n_components):
        color = _REGIME_PALETTE[r % len(_REGIME_PALETTE)]
        handles.append(Patch(facecolor=color, alpha=alpha + 0.1, label=f"Regime {r}"))

    i = 0
    while i < len(labels):
        r = labels[i]
        j = i + 1
        while j < len(labels) and labels[j] == r:
            j += 1
        t_start = timestamps[i]
        t_end = timestamps[min(j, len(timestamps) - 1)]
        ax.axvspan(
            t_start, t_end,
            alpha=alpha,
            color=_REGIME_PALETTE[r % len(_REGIME_PALETTE)],
            linewidth=0,
        )
        i = j

    return handles


def _fmt_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    logger.info("Loading / running Optuna study …")
    study = run_optimization(config)
    candidates = top_n_results(study, n=5)

    if not candidates:
        logger.error("No completed Optuna trials found.")
        return

    logger.info("Fitting %d candidate models on full dataset …", len(candidates))
    df_common = load_common_dataframe(config)

    # One-year window for display
    cutoff = df_common.index.max() - pd.Timedelta(days=_LOOKBACK_DAYS)

    n_panels = len(candidates)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(15, 4 * n_panels),
        sharex=False,
    )
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"SOL/USD — Top {n_panels} HMM Regime Models (last {_LOOKBACK_DAYS}d)",
        fontsize=12, y=1.001,
    )

    for ax, result in zip(axes, candidates):
        n_k = result["n_components"]
        subset = result["feature_subset"]
        obj = result["objective"]
        trial_no = result["trial_number"]

        try:
            X_df = build_feature_matrix(df_common.copy(), subset)
        except ValueError as e:
            logger.warning("Skipping trial %d — feature build failed: %s", trial_no, e)
            ax.text(
                0.5, 0.5, f"Trial {trial_no}: feature error\n{e}",
                ha="center", va="center", transform=ax.transAxes, color="grey",
            )
            continue

        model = build_model(config, n_components=n_k)
        model.fit(X_df.values)
        labels = model.predict(X_df.values)

        # SOL close aligned to feature matrix index
        sol = df_common["SOL_close"].reindex(X_df.index)

        # Restrict to lookback window
        mask = sol.index >= cutoff
        sol_year = sol[mask]
        idx_offset = int(mask.values.argmax())  # first True index in full array
        lab_year = labels[idx_offset: idx_offset + len(sol_year)]

        # Regime stats for subtitle
        stats = model.regime_stats(X_df.values)
        freq_str = "  ".join(
            f"R{r}:{v['frequency']:.0%}" for r, v in sorted(stats.items())
        )

        # Plot
        regime_handles = _shade_regimes(ax, sol_year.index, lab_year, n_k)
        ax.plot(sol_year.index, sol_year.values, color="#9945ff", linewidth=1.0, zorder=3)
        ax.set_title(
            f"k={n_k}  |  {len(subset)+1} features  |  CV obj={obj:.4f}  "
            f"(trial #{trial_no})   [{freq_str}]",
            fontsize=8.5, loc="left",
        )
        ax.set_ylabel("SOL (USD)", fontsize=8)
        ax.legend(handles=regime_handles, fontsize=7, loc="upper left", ncol=n_k)
        _fmt_axis(ax)

        logger.info(
            "Trial %d: n_components=%d  n_features=%d  obj=%.4f",
            trial_no, n_k, len(subset) + 1, obj,
        )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
