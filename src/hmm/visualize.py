"""SOL/USD combined regime + forecast visualization — three panels.

Panel 1 — HMM Regime (full year)
  SOL/USD close price over the last 365 days; regime background shading;
  1-week HMM k-step Markov forecast ±2σ; regime legend with frequency,
  mean hourly return, and annualised volatility.

Panel 2 — XGBoost (2-week window)
  Last 7 days actual SOL + in-data XGB fitted prices (quality check);
  next 7 days XGB recursive forecast with q10/q90 CI band;
  optional XGB+ overlay with top-3 non-selected features.

Panel 3 — NeuralProphet (2-week window)
  Last 7 days actual SOL + NeuralProphet fitted values;
  next 7 days NeuralProphet direct multi-step forecast with q10/q90 CI.

Semantic regime labels (k=5, sorted by mean SOL log-return)
  Rank 1 → Strong Bearish (dark red)
  Rank 2 → Bearish        (red)
  Rank 3 → Neutral        (amber)
  Rank 4 → Bullish        (green)
  Rank 5 → Strong Bullish (dark green)

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

from src.utils.paths import raw_dir

warnings.filterwarnings("ignore")

_ROOT        = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "config" / "settings.yaml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_LOOKBACK_DAYS  = 365
_FORECAST_HOURS = 7 * 24   # 168 steps

# ── Regime colours (sorted by mean SOL return: lowest → highest rank) ─────────
_STRONG_BEARISH_COLOR = "#922b21"   # dark red
_BEARISH_COLOR        = "#e74c3c"   # red
_NEUTRAL_COLOR        = "#f39c12"   # amber
_BULLISH_COLOR        = "#2ecc71"   # green
_STRONG_BULLISH_COLOR = "#1e8449"   # dark green
_EXTRA_COLORS         = ["#8172b2", "#3498db", "#9b59b6", "#1abc9c"]  # k > 5

# ── Forecast / asset colours ──────────────────────────────────────────────────
_SOL_COLOR      = "#9945ff"   # SOL purple
_HMM_COLOR      = "#f1c40f"   # yellow
_XGB_COLOR      = "#3498db"   # blue
_XGB_PLUS_COLOR = "#1abc9c"   # teal
_NP_COLOR       = "#e67e22"   # orange
_INDATA_ALPHA   = 0.55        # in-data prediction line opacity
_BG             = "#0f0f0f"
_AX_BG          = "#0f0f0f"
_TICK_COLOR     = "#bbbbbb"
_GRID_COLOR     = "#444444"
_SPINE_COLOR    = "#333333"


# ─────────────────────────────────────────────────────────────────────────────
# Regime helpers
# ─────────────────────────────────────────────────────────────────────────────


def _assign_regime_colors_and_labels(
    model: object,
    X_df: pd.DataFrame,
    labels: np.ndarray,
) -> dict[int, dict]:
    """Assign semantic colour + label per regime, sorted by mean SOL log-return.

    Ranks: 1 = lowest mean return (most bearish) … k = highest (most bullish).
    Handles k = 2 … 5 explicitly; k > 5 gets generic labels.
    """
    sol_idx  = list(X_df.columns).index("SOL_log_return")
    means    = model._model.means_[:, sol_idx]          # type: ignore[attr-defined]
    covars   = model._model.covars_                     # type: ignore[attr-defined]
    k        = model.n_components                       # type: ignore[attr-defined]

    if model.covariance_type == "full":                 # type: ignore[attr-defined]
        vols = np.array([covars[i][sol_idx, sol_idx] for i in range(k)])
    elif model.covariance_type == "diag":               # type: ignore[attr-defined]
        vols = covars[:, sol_idx]
    elif model.covariance_type == "tied":               # type: ignore[attr-defined]
        vols = np.full(k, covars[sol_idx, sol_idx])
    else:
        vols = covars

    ann_vols = np.sqrt(np.maximum(vols, 0) * 24 * 365) * 100

    order = np.argsort(means)   # ascending rank: rank 1 = lowest mean return

    if k == 2:
        base_labels = {1: "Bearish", 2: "Bullish"}
        base_colors = [_BEARISH_COLOR, _BULLISH_COLOR]
    elif k == 3:
        base_labels = {1: "Bearish", 2: "Neutral", 3: "Bullish"}
        base_colors = [_BEARISH_COLOR, _NEUTRAL_COLOR, _BULLISH_COLOR]
    elif k == 4:
        base_labels = {1: "Strong Bearish", 2: "Bearish", 3: "Bullish", 4: "Strong Bullish"}
        base_colors = [_STRONG_BEARISH_COLOR, _BEARISH_COLOR, _BULLISH_COLOR, _STRONG_BULLISH_COLOR]
    elif k == 5:
        base_labels = {
            1: "Strong Bearish", 2: "Bearish",
            3: "Neutral",
            4: "Bullish",       5: "Strong Bullish",
        }
        base_colors = [
            _STRONG_BEARISH_COLOR, _BEARISH_COLOR, _NEUTRAL_COLOR,
            _BULLISH_COLOR, _STRONG_BULLISH_COLOR,
        ]
    else:
        mid_colors  = _EXTRA_COLORS[: k - 2]
        base_colors = [_STRONG_BEARISH_COLOR] + mid_colors + [_STRONG_BULLISH_COLOR]
        base_labels = {1: "Strong Bearish", k: "Strong Bullish"}
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
    """Fill contiguous regime blocks with semi-transparent colour bands."""
    i = 0
    while i < len(labels):
        r = labels[i]
        j = i + 1
        while j < len(labels) and labels[j] == r:
            j += 1
        t_end = timestamps[min(j, len(timestamps) - 1)]
        ax.axvspan(timestamps[i], t_end, alpha=alpha,
                   color=regime_info[r]["color"], linewidth=0)
        i = j


# ─────────────────────────────────────────────────────────────────────────────
# HMM k-step forecast (self-contained copy from predict.py)
# ─────────────────────────────────────────────────────────────────────────────


def _kstep_forecast(
    model: object,
    X_df: pd.DataFrame,
    k: int,
    sol_last: float,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    hmm       = model._model                            # type: ignore[attr-defined]
    A         = hmm.transmat_
    means     = hmm.means_
    covars    = hmm.covars_
    sol_idx   = list(X_df.columns).index("SOL_log_return")
    state_mean = means[:, sol_idx]
    n_states  = model.n_components                      # type: ignore[attr-defined]

    if model.covariance_type == "full":                 # type: ignore[attr-defined]
        state_var = np.array([covars[i][sol_idx, sol_idx] for i in range(n_states)])
    elif model.covariance_type == "diag":               # type: ignore[attr-defined]
        state_var = covars[:, sol_idx]
    elif model.covariance_type == "tied":               # type: ignore[attr-defined]
        state_var = np.full(n_states, covars[sol_idx, sol_idx])
    else:
        state_var = covars

    pi_j     = model.predict_proba(X_df.values)[-1]    # type: ignore[attr-defined]
    cum_lrs  = np.zeros(k)
    cum_vars = np.zeros(k)
    cum_lr = cum_var = 0.0

    for j in range(k):
        mu_j  = float(pi_j @ state_mean)
        ex2_j = float(pi_j @ (state_mean ** 2 + state_var))
        var_j = ex2_j - mu_j ** 2
        cum_lr  += mu_j
        cum_var += max(var_j, 0.0)
        cum_lrs[j]  = cum_lr
        cum_vars[j] = cum_var
        pi_j = pi_j @ A

    sigma    = np.sqrt(cum_vars)
    exp_price = sol_last * np.exp(cum_lrs)
    lo_price  = sol_last * np.exp(cum_lrs - 2.0 * sigma)
    hi_price  = sol_last * np.exp(cum_lrs + 2.0 * sigma)

    future_ts = pd.date_range(
        start=X_df.index[-1] + pd.Timedelta(hours=1),
        periods=k, freq="1h", tz="UTC",
    )
    return future_ts, exp_price, lo_price, hi_price


# ─────────────────────────────────────────────────────────────────────────────
# Panel helpers
# ─────────────────────────────────────────────────────────────────────────────


def _style_ax(ax: plt.Axes, hourly: bool = False) -> None:
    ax.set_facecolor(_AX_BG)
    ax.tick_params(axis="both", labelsize=8.5, colors=_TICK_COLOR)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4, color=_GRID_COLOR)
    ax.grid(axis="x", linewidth=0.2, alpha=0.2, color=_GRID_COLOR)
    ax.spines[["top", "right", "left", "bottom"]].set_color(_SPINE_COLOR)
    ax.set_ylabel("USD", color=_TICK_COLOR, fontsize=9)
    if hourly:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:00 UTC"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))


def _draw_two_week_panel(
    ax: plt.Axes,
    in_data_ts: pd.DatetimeIndex,
    in_data_actual: np.ndarray,
    in_data_pred: np.ndarray,
    in_data_rmse: float,
    future_ts: pd.DatetimeIndex,
    exp_price: np.ndarray,
    lo_price: np.ndarray,
    hi_price: np.ndarray,
    forecast_color: str,
    in_data_label: str,
    forecast_label: str,
    current_regime: str | None = None,
    feature_names: list[str] | None = None,
    today_actual_ts: pd.DatetimeIndex | None = None,
    today_actual: np.ndarray | None = None,
    plus_exp: np.ndarray | None = None,
    plus_in_pred: np.ndarray | None = None,
    plus_in_ts: pd.DatetimeIndex | None = None,
    plus_rmse: float | None = None,
    plus_features: list[str] | None = None,
    today_midnight: pd.Timestamp | None = None,
) -> None:
    """Draw the 3-day history + today 00:00–23:00 forecast panel."""
    today = in_data_ts[-1]
    _today_midnight = today_midnight if today_midnight is not None else pd.Timestamp.now(tz="UTC").normalize()

    # Actual prices: prefer SOL.parquet series (continuous, no VIX-join gaps)
    if today_actual is not None and today_actual_ts is not None and len(today_actual) > 0:
        last_t = today_actual_ts[-1].strftime("%d %b %H:%M")
        ax.plot(today_actual_ts, today_actual,
                color=_SOL_COLOR, linewidth=1.2, zorder=4,
                label=f"SOL/USD actual (bis {last_t} UTC)")
    else:
        ax.plot(in_data_ts, in_data_actual,
                color=_SOL_COLOR, linewidth=1.2, zorder=4, label="SOL/USD actual")

    # In-data model predictions
    ax.plot(in_data_ts, in_data_pred,
            color=forecast_color, linewidth=1.0, alpha=_INDATA_ALPHA, linestyle="-",
            zorder=3, label=f"{in_data_label} in-data  RMSE=${in_data_rmse:.2f}")

    # XGB+ in-data (if available)
    if plus_in_pred is not None and plus_in_ts is not None:
        ts_for_plus = plus_in_ts if len(plus_in_ts) == len(plus_in_pred) else in_data_ts
        ax.plot(ts_for_plus, plus_in_pred,
                color=_XGB_PLUS_COLOR, linewidth=1.0, alpha=_INDATA_ALPHA, linestyle="-",
                zorder=3, label=f"{in_data_label}+ in-data  RMSE=${plus_rmse:.2f}")

    # Today divider
    ax.axvline(today, color="#555555", linewidth=0.8, linestyle=":", zorder=3)

    # Forecast
    ax.fill_between(future_ts, lo_price, hi_price,
                    color=forecast_color, alpha=0.15, zorder=2,
                    label=f"CI 80%  [${lo_price[-1]:.0f}–${hi_price[-1]:.0f}]")
    ax.plot(future_ts, exp_price,
            color=forecast_color, linewidth=2.2, linestyle="--", zorder=5,
            label=f"{forecast_label}  E[+24h]=${exp_price[-1]:.2f}")

    # XGB+ forecast
    if plus_exp is not None:
        feat_str = ", ".join((plus_features or [])[:2])
        ax.plot(future_ts, plus_exp,
                color=_XGB_PLUS_COLOR, linewidth=2.0, linestyle="-", zorder=5,
                label=f"{in_data_label}+ E[+24h]=${plus_exp[-1]:.2f}  (+{feat_str}…)")

    # Regime + feature annotation (top-right)
    if current_regime is not None:
        feat_line = ("Features: " + ", ".join(feature_names)) if feature_names else ""
        annotation = f"Active regime: {current_regime}\n{feat_line}"
        ax.text(
            0.99, 0.97, annotation,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=7, color="#cccccc",
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      edgecolor="#444444", alpha=0.7),
        )

    # Fix x-axis: 3 full calendar days back (midnight) → today 23:00 UTC
    ax.set_xlim(
        _today_midnight - pd.Timedelta(days=3),
        _today_midnight + pd.Timedelta(hours=23),
    )

    # Legend
    ax.legend(fontsize=8, loc="upper left", framealpha=0.15,
              labelcolor="white", facecolor="#1a1a1a", edgecolor="#333333")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def run(config: dict) -> None:
    """Full visualization pipeline. Called from main.py with injected time values."""
    # ── HMM pipeline ──────────────────────────────────────────────────────────
    from src.hmm.optimizer import run_optimization, top_n_results
    from src.hmm.features import build_feature_matrix, load_common_dataframe
    from src.hmm.model import GaussianHMMModel, build_model
    from src.utils.paths import models_dir

    study  = run_optimization(config)
    best   = top_n_results(study, n=1)[0]
    n_k    = best["n_components"]
    subset = best["feature_subset"]

    model_path = models_dir(config) / f"best_hmm_k{n_k}.pkl"
    if model_path.exists():
        model = GaussianHMMModel.load(model_path)
    else:
        df_common = load_common_dataframe(config)
        X_df      = build_feature_matrix(df_common.copy(), subset)
        model     = build_model(config, n_components=n_k)
        model.fit(X_df.values)
        model.save(model_path)

    df_common = load_common_dataframe(config)
    X_df      = build_feature_matrix(df_common.copy(), subset)
    sol_close = df_common["SOL_close"].reindex(X_df.index)

    # Align to yesterday 23:00 UTC — HMM forecast starts at today 00:00 UTC
    _cutoff = config.get("_cutoff", pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(hours=1))
    if X_df.index[-1] > _cutoff:
        X_df      = X_df.loc[X_df.index <= _cutoff]
        sol_close = sol_close.loc[sol_close.index <= _cutoff]

    labels      = model.predict(X_df.values)
    regime_info = _assign_regime_colors_and_labels(model, X_df, labels)

    current_regime_label = regime_info[int(labels[-1])]["label"]

    cutoff   = sol_close.index[-1] - pd.Timedelta(days=_LOOKBACK_DAYS)
    mask     = sol_close.index >= cutoff
    sol_year = sol_close[mask]
    idx_start  = int(np.searchsorted(X_df.index, sol_year.index[0]))
    lab_year   = labels[idx_start: idx_start + len(sol_year)]
    ts_year    = sol_year.index

    sol_last = float(sol_close.iloc[-1])
    future_ts_hmm, exp_price_hmm, lo_price_hmm, hi_price_hmm = _kstep_forecast(
        model, X_df, _FORECAST_HOURS, sol_last
    )
    logger.info(
        "HMM: last=$%.2f  E[+7d]=$%.2f  CI=[$%.2f, $%.2f]",
        sol_last, exp_price_hmm[-1], lo_price_hmm[-1], hi_price_hmm[-1],
    )

    # ── Recent SOL actual prices from SOL.parquet (last 96 h to now) ─────────
    # SOL.parquet has intraday data including today and covers any gaps that
    # the common DataFrame (inner-joined with VIX) might have.
    _sol_par = raw_dir(config) / "SOL.parquet"
    _today_actual_ts: pd.DatetimeIndex | None = None
    _today_actual: np.ndarray | None = None
    if _sol_par.exists():
        _sol_full = pd.read_parquet(_sol_par)
        if _sol_full.index.tz is None:
            _sol_full.index = _sol_full.index.tz_localize("UTC")
        _now_today    = config.get("_today", pd.Timestamp.now(tz="UTC").normalize())
        _last_hour    = config.get("_last_hour", pd.Timestamp.now(tz="UTC").floor("h"))
        _window_start = _now_today - pd.Timedelta(days=3)
        _recent_sol   = _sol_full.loc[
            (_sol_full.index >= _window_start) & (_sol_full.index <= _last_hour),
            "close",
        ]
        if not _recent_sol.empty:
            _today_actual_ts = _recent_sol.index
            _today_actual    = _recent_sol.values.astype(float)
            logger.info(
                "Recent actual SOL: %d rows (%s – %s UTC)",
                len(_today_actual),
                _today_actual_ts[0].strftime("%Y-%m-%d %H:%M"),
                _today_actual_ts[-1].strftime("%Y-%m-%d %H:%M"),
            )

    # ── XGBoost pipeline ──────────────────────────────────────────────────────
    from src.hmm.predict_xgb import run as run_xgb
    xgb_results = run_xgb(config)

    # ── NeuralProphet pipeline ────────────────────────────────────────────────
    from src.hmm.predict_prophet import run as run_np
    np_results = run_np(config)

    # ── Build figure (3 panels) ───────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(17, 18))
    fig.patch.set_facecolor(_BG)

    # ── Panel 1: HMM (1-year view) ────────────────────────────────────────────
    ax1.set_facecolor(_AX_BG)
    _shade_regimes(ax1, ts_year, lab_year, regime_info, alpha=0.25)
    ax1.plot(sol_year.index, sol_year.values,
             color=_SOL_COLOR, linewidth=1.2, zorder=4, label="SOL/USD close")
    ax1.axvline(sol_close.index[-1], color="#555555", linewidth=0.8, linestyle=":", zorder=3)
    ax1.plot(future_ts_hmm, exp_price_hmm,
             color=_HMM_COLOR, linewidth=2.2, linestyle="--", zorder=5,
             label=f"HMM E[+7d]  ${exp_price_hmm[-1]:.2f}")
    ax1.fill_between(future_ts_hmm, lo_price_hmm, hi_price_hmm,
                     color=_HMM_COLOR, alpha=0.15, zorder=2,
                     label=f"±2σ  [${lo_price_hmm[-1]:.0f}–${hi_price_hmm[-1]:.0f}]")

    regime_handles = []
    for rid, info in sorted(regime_info.items()):
        lbl = (f"{info['label']}  "
               f"({info['frequency']:.0%} | "
               f"μ={info['mean_ret']*100:+.3f}%/h | "
               f"σ={info['ann_vol_pct']:.0f}% ann.)")
        regime_handles.append(Patch(facecolor=info["color"], alpha=0.55, label=lbl))

    all_h, all_l = ax1.get_legend_handles_labels()
    ax1.legend(
        handles=regime_handles + all_h,
        labels=[h.get_label() for h in regime_handles] + all_l,
        fontsize=8.5, loc="upper left", framealpha=0.15,
        labelcolor="white", facecolor="#1a1a1a", edgecolor="#333333",
    )

    ax1.tick_params(axis="both", labelsize=8.5, colors=_TICK_COLOR)
    ax1.grid(axis="y", linewidth=0.3, alpha=0.4, color=_GRID_COLOR)
    ax1.grid(axis="x", linewidth=0.2, alpha=0.2, color=_GRID_COLOR)
    ax1.spines[["top", "right", "left", "bottom"]].set_color(_SPINE_COLOR)
    ax1.set_ylabel("USD", color=_TICK_COLOR, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    last_date  = sol_close.index[-1].strftime("%Y-%m-%d")
    fcast_date = future_ts_hmm[-1].strftime("%Y-%m-%d")
    ax1.set_title(
        f"SOL/USD — {n_k}-Regime HMM  |  Last: ${sol_last:.2f} ({last_date})"
        f"  →  E[+7d]: ${exp_price_hmm[-1]:.2f} ({fcast_date})",
        color="white", fontsize=11, loc="left", pad=10,
    )

    # ── Panel 2: XGBoost ──────────────────────────────────────────────────────
    _today_midnight = config.get("_today", pd.Timestamp.now(tz="UTC").normalize())
    _style_ax(ax2, hourly=True)
    xgb_features = xgb_results.get("feature_names", subset)
    _draw_two_week_panel(
        ax2,
        in_data_ts      = xgb_results["in_data_ts"],
        in_data_actual  = xgb_results["in_data_actual"],
        in_data_pred    = xgb_results["in_data_pred"],
        in_data_rmse    = xgb_results["in_data_rmse"],
        future_ts       = xgb_results["future_ts"],
        exp_price       = xgb_results["xgb_exp"],
        lo_price        = xgb_results["xgb_lo"],
        hi_price        = xgb_results["xgb_hi"],
        forecast_color  = _XGB_COLOR,
        in_data_label   = "XGB",
        forecast_label  = "XGBoost",
        current_regime  = current_regime_label,
        feature_names   = xgb_features,
        today_actual_ts = _today_actual_ts,
        today_actual    = _today_actual,
        plus_exp        = xgb_results.get("xgb_plus_exp"),
        plus_in_pred    = xgb_results.get("xgb_plus_in_pred"),
        plus_in_ts      = xgb_results.get("xgb_plus_in_ts"),
        plus_rmse       = xgb_results.get("xgb_plus_rmse"),
        plus_features   = xgb_results.get("plus_features"),
        today_midnight  = _today_midnight,
    )
    ax2.set_title(
        f"XGBoost recursive forecast  |  E[+24h]: ${xgb_results['xgb_exp'][-1]:.2f}"
        + (f"  XGB+: ${xgb_results['xgb_plus_exp'][-1]:.2f}"
           if xgb_results.get("xgb_plus_exp") is not None else ""),
        color="white", fontsize=10, loc="left", pad=8,
    )

    # ── Panel 3: NeuralProphet ────────────────────────────────────────────────
    _style_ax(ax3, hourly=True)
    np_r = np_results
    _draw_two_week_panel(
        ax3,
        in_data_ts      = np_r["in_data_ts"],
        in_data_actual  = np_r["in_data_actual"],
        in_data_pred    = np_r["in_data_pred"],
        in_data_rmse    = np_r["in_data_rmse"],
        future_ts       = np_r["future_ts"],
        exp_price       = np_r["np_exp"],
        lo_price        = np_r["np_lo"],
        hi_price        = np_r["np_hi"],
        forecast_color  = _NP_COLOR,
        in_data_label   = "NeuralProphet",
        forecast_label  = "NeuralProphet",
        current_regime  = current_regime_label,
        feature_names   = subset,
        today_actual_ts = _today_actual_ts,
        today_actual    = _today_actual,
        today_midnight  = _today_midnight,
    )
    ax3.set_title(
        f"NeuralProphet direct forecast  |  E[+24h]: ${np_r['np_exp'][-1]:.2f}"
        if not np.isnan(np_r["np_exp"][-1]) else "NeuralProphet direct forecast",
        color="white", fontsize=10, loc="left", pad=8,
    )

    fig.tight_layout(pad=2.0)

    out_dir = Path(config.get("paths", {}).get("processed_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sol_forecast.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info("Plot saved → %s", out_path)

    plt.show()


def main() -> None:
    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    now_utc = pd.Timestamp.now(tz="UTC")
    config["_now_utc"]   = now_utc
    config["_today"]     = now_utc.normalize()
    config["_cutoff"]    = now_utc.normalize() - pd.Timedelta(hours=1)
    config["_last_hour"] = now_utc.floor("h")

    run(config)


if __name__ == "__main__":
    main()
