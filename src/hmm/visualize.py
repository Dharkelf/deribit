"""SOL/USD regime visualization with 1-week forecast.

Standalone script — not imported by any other module.

Shows:
  - SOL/USD close price over the last 365 days
  - Regime background shading (contiguous coloured bands under the curve)
  - 1-week hourly forecast line with ±2σ confidence band
  - Regime legend: colour, label (Bullish/Neutral/Bearish by mean return),
    frequency, and annualised volatility per regime

Usage:
    python -m src.hmm.visualize
    python src/hmm/visualize.py
"""

import logging
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "config" / "settings.yaml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_LOOKBACK_DAYS = 365
_FORECAST_HOURS = 7 * 24  # 168 steps

# Regime colours — sorted by mean SOL return: bearish → neutral → bullish
_BEARISH_COLOR  = "#e74c3c"   # red
_NEUTRAL_COLOR  = "#f39c12"   # amber
_BULLISH_COLOR  = "#2ecc71"   # green
_EXTRA_COLORS   = ["#8172b2", "#3498db"]   # for k > 3

_SOL_COLOR      = "#9945ff"   # SOL purple
_FORECAST_COLOR = "#f1c40f"   # yellow


# ─────────────────────────────────────────────────────────────────────────────
# Regime helpers
# ─────────────────────────────────────────────────────────────────────────────


def _assign_regime_colors_and_labels(
    model,
    X_df: pd.DataFrame,
    labels: np.ndarray,
) -> dict[int, dict]:
    """Characterise each regime by mean SOL log-return and volatility.

    Regimes are ranked by mean return:
      lowest  → Bearish  (red)
      highest → Bullish  (green)
      middle  → Neutral  (amber)   [only for k=3+]
    Extra regimes (k>3) get purple / blue.

    Returns dict[regime_id] → {color, label, frequency, ann_vol_pct}
    """
    sol_idx = list(X_df.columns).index("SOL_log_return")
    means    = model._model.means_[:, sol_idx]          # (k,)
    covars   = model._model.covars_                     # shape depends on type
    k        = model.n_components

    if model.covariance_type == "full":
        vols = np.array([covars[i][sol_idx, sol_idx] for i in range(k)])
    elif model.covariance_type == "diag":
        vols = covars[:, sol_idx]
    elif model.covariance_type == "tied":
        vols = np.full(k, covars[sol_idx, sol_idx])
    else:
        vols = covars

    ann_vols = np.sqrt(np.maximum(vols, 0) * 24 * 365) * 100  # annualised %

    # Rank by mean return
    order = np.argsort(means)        # ascending: lowest → highest
    base_labels = {
        1: "Bearish",
        k: "Bullish",
    }
    if k == 2:
        base_colors = [_BEARISH_COLOR, _BULLISH_COLOR]
    elif k == 3:
        base_labels[2] = "Neutral"
        base_colors = [_BEARISH_COLOR, _NEUTRAL_COLOR, _BULLISH_COLOR]
    else:
        mid_colors = _EXTRA_COLORS[: k - 2]
        base_colors = [_BEARISH_COLOR] + mid_colors + [_BULLISH_COLOR]
        for rank in range(2, k):
            base_labels[rank] = f"Regime {rank}"

    info: dict[int, dict] = {}
    for rank, regime_id in enumerate(order, start=1):
        mask = labels == regime_id
        info[regime_id] = {
            "color":       base_colors[rank - 1],
            "label":       base_labels.get(rank, f"Regime {rank}"),
            "frequency":   float(mask.mean()),
            "ann_vol_pct": float(ann_vols[regime_id]),
            "mean_ret":    float(means[regime_id]),
        }
    return info


def _shade_regimes(
    ax: plt.Axes,
    timestamps: pd.DatetimeIndex,
    labels: np.ndarray,
    regime_info: dict[int, dict],
    alpha: float = 0.22,
) -> None:
    """Fill contiguous regime blocks with semi-transparent bands."""
    i = 0
    while i < len(labels):
        r = labels[i]
        j = i + 1
        while j < len(labels) and labels[j] == r:
            j += 1
        t_end = timestamps[min(j, len(timestamps) - 1)]
        ax.axvspan(
            timestamps[i], t_end,
            alpha=alpha,
            color=regime_info[r]["color"],
            linewidth=0,
        )
        i = j


# ─────────────────────────────────────────────────────────────────────────────
# Forecast (reused from predict.py logic, self-contained here)
# ─────────────────────────────────────────────────────────────────────────────


def _kstep_forecast(model, X_df: pd.DataFrame, k: int, sol_last: float):
    hmm       = model._model
    A         = hmm.transmat_
    means     = hmm.means_
    covars    = hmm.covars_
    sol_idx   = list(X_df.columns).index("SOL_log_return")
    state_mean = means[:, sol_idx]

    if model.covariance_type == "full":
        state_var = np.array([covars[i][sol_idx, sol_idx] for i in range(model.n_components)])
    elif model.covariance_type == "diag":
        state_var = covars[:, sol_idx]
    elif model.covariance_type == "tied":
        state_var = np.full(model.n_components, covars[sol_idx, sol_idx])
    else:
        state_var = covars

    pi_j = model.predict_proba(X_df.values)[-1]
    cum_lr, cum_var = 0.0, 0.0
    cum_lrs  = np.zeros(k)
    cum_vars = np.zeros(k)

    for j in range(k):
        mu_j   = float(pi_j @ state_mean)
        ex2_j  = float(pi_j @ (state_mean ** 2 + state_var))
        var_j  = ex2_j - mu_j ** 2
        cum_lr  += mu_j
        cum_var += max(var_j, 0.0)
        cum_lrs[j]  = cum_lr
        cum_vars[j] = cum_var
        pi_j = pi_j @ A

    sigma     = np.sqrt(cum_vars)
    exp_price = sol_last * np.exp(cum_lrs)
    lo_price  = sol_last * np.exp(cum_lrs - 2.0 * sigma)
    hi_price  = sol_last * np.exp(cum_lrs + 2.0 * sigma)

    future_ts = pd.date_range(
        start=X_df.index[-1] + pd.Timedelta(hours=1),
        periods=k, freq="1h", tz="UTC",
    )
    return future_ts, exp_price, lo_price, hi_price


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # ── Load optimizer results ────────────────────────────────────────────────
    from src.hmm.optimizer import run_optimization, top_n_results
    from src.hmm.features import build_feature_matrix, load_common_dataframe
    from src.hmm.model import GaussianHMMModel, build_model
    from src.utils.paths import models_dir

    study    = run_optimization(config)
    best     = top_n_results(study, n=1)[0]
    n_k      = best["n_components"]
    subset   = best["feature_subset"]

    # ── Load or fit final model ───────────────────────────────────────────────
    model_path = models_dir(config) / f"best_hmm_k{n_k}.pkl"
    if model_path.exists():
        logger.info("Loading saved model ← %s", model_path)
        model = GaussianHMMModel.load(model_path)
    else:
        logger.info("Fitting model (n_components=%d) on full dataset …", n_k)
        df_common = load_common_dataframe(config)
        X_df      = build_feature_matrix(df_common.copy(), subset)
        model     = build_model(config, n_components=n_k)
        model.fit(X_df.values)
        model.save(model_path)

    # ── Build feature matrix and align SOL price ──────────────────────────────
    df_common = load_common_dataframe(config)
    X_df      = build_feature_matrix(df_common.copy(), subset)
    sol_close = df_common["SOL_close"].reindex(X_df.index)

    # ── Predict regimes ───────────────────────────────────────────────────────
    labels     = model.predict(X_df.values)
    regime_info = _assign_regime_colors_and_labels(model, X_df, labels)

    # ── Restrict history to lookback window ───────────────────────────────────
    cutoff   = sol_close.index[-1] - pd.Timedelta(days=_LOOKBACK_DAYS)
    mask     = sol_close.index >= cutoff
    sol_year = sol_close[mask]

    idx_start   = int(np.searchsorted(X_df.index, sol_year.index[0]))
    lab_year    = labels[idx_start: idx_start + len(sol_year)]
    ts_year     = sol_year.index

    # ── 7-day forecast ────────────────────────────────────────────────────────
    sol_last = float(sol_close.iloc[-1])
    future_ts, exp_price, lo_price, hi_price = _kstep_forecast(
        model, X_df, _FORECAST_HOURS, sol_last
    )

    logger.info(
        "SOL last=$%.2f  E[+7d]=$%.2f  CI=[$%.2f, $%.2f]",
        sol_last, exp_price[-1], lo_price[-1], hi_price[-1],
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(17, 6))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    # Regime shading
    _shade_regimes(ax, ts_year, lab_year, regime_info, alpha=0.25)

    # Historical SOL price
    ax.plot(
        sol_year.index, sol_year.values,
        color=_SOL_COLOR, linewidth=1.2, zorder=4, label="SOL/USD close",
    )

    # Vertical "today" divider
    today = sol_close.index[-1]
    ax.axvline(today, color="#555555", linewidth=0.8, linestyle=":", zorder=3)

    # Forecast
    ax.plot(
        future_ts, exp_price,
        color=_FORECAST_COLOR, linewidth=2.2, linestyle="--", zorder=5,
        label=f"E[+7d]  ${exp_price[-1]:.2f}",
    )
    ax.fill_between(
        future_ts, lo_price, hi_price,
        color=_FORECAST_COLOR, alpha=0.15, zorder=2,
        label=f"±2σ  [${lo_price[-1]:.0f} – ${hi_price[-1]:.0f}]",
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    regime_handles = []
    for rid, info in sorted(regime_info.items()):
        lbl = (
            f"{info['label']}  "
            f"({info['frequency']:.0%} | "
            f"μ={info['mean_ret']*100:+.3f}%/h | "
            f"σ={info['ann_vol_pct']:.0f}% ann.)"
        )
        regime_handles.append(
            Patch(facecolor=info["color"], alpha=0.55, label=lbl)
        )

    all_handles, all_labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles=regime_handles + all_handles,
        labels=[h.get_label() for h in regime_handles] + all_labels,
        fontsize=8.5,
        loc="upper left",
        framealpha=0.15,
        labelcolor="white",
        facecolor="#1a1a1a",
        edgecolor="#333333",
    )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax.tick_params(axis="both", labelsize=8.5, colors="#bbbbbb")
    ax.grid(axis="y", linewidth=0.3, alpha=0.4, color="#444444")
    ax.grid(axis="x", linewidth=0.2, alpha=0.2, color="#444444")
    ax.spines[["top", "right", "left", "bottom"]].set_color("#333333")
    ax.set_ylabel("USD", color="#bbbbbb", fontsize=9)

    last_date  = sol_close.index[-1].strftime("%Y-%m-%d")
    fcast_date = future_ts[-1].strftime("%Y-%m-%d")
    ax.set_title(
        f"SOL/USD — {n_k}-Regime HMM  |  Last: ${sol_last:.2f} ({last_date})  "
        f"→  E[+7d]: ${exp_price[-1]:.2f} ({fcast_date})",
        color="white", fontsize=11, loc="left", pad=10,
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
