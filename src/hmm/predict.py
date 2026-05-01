"""1-week forward SOL price forecast (168 hourly steps).

Design
------
Template Method: run() is the fixed pipeline skeleton —
  1. Load/run optimizer (cached Optuna study)
  2. Select best feature subset + n_components
  3. Fit final GaussianHMM on the complete historical dataset
  4. k-step Markov transition: P(regime t+k) = π₀ · Aᵏ
  5. Expected SOL log-return and price; ±2σ confidence band
  6. Plot: historical SOL close with regime background + forecast horizon

Forecast maths
--------------
  π₀  = posterior state distribution at last observed timestamp (forward_proba[-1])
  πⱼ  = π₀ · Aʲ                     (j = 1, …, k  where k = 7*24 = 168)
  μ_j = Σᵢ πⱼ[i] · mean_i[sol_idx]  (expected log-return at step j)
  σ²_j= Σᵢ πⱼ[i] · (mean_i² + diag_cov_i)[sol_idx] − μ_j²
  cum_lr(k) = Σⱼ₌₁ᵏ μ_j             (cumulative expected log-return)
  cum_var(k) = Σⱼ₌₁ᵏ σ²_j           (additive; assumes step independence)
  E[SOL(t+k)] = SOL_last · exp(cum_lr(k))
  CI = SOL_last · exp(cum_lr ± 2·√cum_var)
"""

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.hmm.features import build_feature_matrix, load_common_dataframe
from src.hmm.model import GaussianHMMModel, build_model
from src.hmm.optimizer import run_optimization, top_n_results
from src.utils.paths import models_dir, project_root

logger = logging.getLogger(__name__)

_FORECAST_HOURS = 7 * 24  # 1 week

_REGIME_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]

_ROOT = project_root()
_CONFIG_PATH = _ROOT / "config" / "settings.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Forecast helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sol_col_index(X_df: pd.DataFrame) -> int:
    """Column index of SOL_log_return in the feature matrix."""
    cols = list(X_df.columns)
    return cols.index("SOL_log_return")


def _kstep_forecast(
    model: GaussianHMMModel,
    X_df: pd.DataFrame,
    k: int,
    sol_last_price: float,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Compute k-step ahead expected SOL price and ±2σ band.

    Returns
    -------
    future_ts : DatetimeIndex  hourly timestamps for steps 1..k
    exp_price : ndarray        expected SOL price at each step
    lo_price  : ndarray        lower 2σ bound
    hi_price  : ndarray        upper 2σ bound
    """
    hmm = model._model
    A = hmm.transmat_  # (n_states, n_states)
    means = hmm.means_  # (n_states, n_features)
    covars = hmm.covars_  # depends on covariance_type

    sol_idx = _sol_col_index(X_df)

    # Current state distribution — posterior at last observation
    pi = model.predict_proba(X_df.values)[-1]  # (n_states,)

    # Pre-extract per-state mean and variance for SOL log-return
    state_mean = means[:, sol_idx]  # (n_states,)

    if model.covariance_type == "full":
        state_var = np.array([covars[i][sol_idx, sol_idx] for i in range(model.n_components)])
    elif model.covariance_type == "diag":
        state_var = covars[:, sol_idx]
    elif model.covariance_type == "tied":
        state_var = np.full(model.n_components, covars[sol_idx, sol_idx])
    else:  # spherical
        state_var = covars  # scalar per state

    cum_lr = 0.0
    cum_var = 0.0
    cum_lrs = np.zeros(k)
    cum_vars = np.zeros(k)

    pi_j = pi.copy()
    for j in range(k):
        mu_j = float(pi_j @ state_mean)
        # E[X²] − (E[X])² for a mixture
        ex2_j = float(pi_j @ (state_mean**2 + state_var))
        var_j = ex2_j - mu_j**2
        cum_lr += mu_j
        cum_var += max(var_j, 0.0)
        cum_lrs[j] = cum_lr
        cum_vars[j] = cum_var
        pi_j = pi_j @ A

    exp_price = sol_last_price * np.exp(cum_lrs)
    sigma = np.sqrt(cum_vars)
    lo_price = sol_last_price * np.exp(cum_lrs - 2.0 * sigma)
    hi_price = sol_last_price * np.exp(cum_lrs + 2.0 * sigma)

    last_ts = X_df.index[-1]
    future_ts = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=k,
        freq="1h",
        tz="UTC",
    )

    return future_ts, exp_price, lo_price, hi_price


# ─────────────────────────────────────────────────────────────────────────────
# Regime background shading helper
# ─────────────────────────────────────────────────────────────────────────────


def _shade_regimes(
    ax: plt.Axes,
    timestamps: pd.DatetimeIndex,
    labels: np.ndarray,
    n_components: int,
    alpha: float = 0.18,
) -> None:
    """Fill contiguous regime blocks with semi-transparent colour bands."""
    colors = _REGIME_COLORS[:n_components]
    i = 0
    while i < len(labels):
        regime = labels[i]
        j = i + 1
        while j < len(labels) and labels[j] == regime:
            j += 1
        ax.axvspan(
            timestamps[i], timestamps[min(j, len(timestamps) - 1)],
            alpha=alpha, color=colors[regime % len(colors)], linewidth=0,
        )
        i = j


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────


def run(config: dict) -> None:
    """Full pipeline: optimise → fit → forecast → plot."""
    logger.info("=== HMM predict: 1-week SOL forecast ===")

    # 1. Optimise (cached)
    study = run_optimization(config)
    best_results = top_n_results(study, n=1)
    if not best_results:
        logger.error("No completed Optuna trials — cannot predict.")
        return
    best = best_results[0]

    n_components = best["n_components"]
    feature_subset = best["feature_subset"]
    logger.info(
        "Best config: n_components=%d  n_optional_features=%d  obj=%.4f",
        n_components, len(feature_subset), best["objective"],
    )

    # 2. Build full feature matrix
    df_common = load_common_dataframe(config)
    X_df = build_feature_matrix(df_common.copy(), feature_subset)
    logger.info("Feature matrix: %d rows × %d cols", *X_df.shape)

    # 3. Fit final model
    model = build_model(config, n_components=n_components)
    model.fit(X_df.values)

    # 4. Predict regimes on full history
    labels = model.predict(X_df.values)

    # SOL price aligned to feature matrix index
    sol_close = df_common["SOL_close"].reindex(X_df.index)
    sol_last = float(sol_close.iloc[-1])

    # 5. k-step forecast
    future_ts, exp_price, lo_price, hi_price = _kstep_forecast(
        model, X_df, _FORECAST_HOURS, sol_last
    )

    logger.info(
        "Forecast: SOL last=%.2f  E[+7d]=%.2f  CI=[%.2f, %.2f]",
        sol_last, exp_price[-1], lo_price[-1], hi_price[-1],
    )

    # 6. Save model
    mdir = models_dir(config)
    model_path = mdir / f"best_hmm_k{n_components}.pkl"
    model.save(model_path)

    # 7. Plot
    _plot_forecast(
        sol_close, labels, n_components, future_ts, exp_price, lo_price, hi_price
    )


def _plot_forecast(
    sol_close: pd.Series,
    labels: np.ndarray,
    n_components: int,
    future_ts: pd.DatetimeIndex,
    exp_price: np.ndarray,
    lo_price: np.ndarray,
    hi_price: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))

    # Regime shading — last year of history only (for readability)
    cutoff = sol_close.index[-1] - pd.Timedelta(days=365)
    mask = sol_close.index >= cutoff
    ts_year = sol_close.index[mask]
    lab_year = labels[np.searchsorted(sol_close.index, ts_year[0]):]
    lab_year = lab_year[: len(ts_year)]
    _shade_regimes(ax, ts_year, lab_year, n_components)

    # Historical SOL price
    ax.plot(sol_close.index, sol_close.values, color="#9945ff", linewidth=1.2,
            label="SOL close (USD)")

    # Forecast
    ax.plot(future_ts, exp_price, color="#f39c12", linewidth=2.0,
            linestyle="--", label="Expected SOL (+7d)")
    ax.fill_between(future_ts, lo_price, hi_price, color="#f39c12", alpha=0.18,
                    label="±2σ confidence")

    # Marker at today
    ax.axvline(sol_close.index[-1], color="grey", linewidth=0.8, linestyle=":")

    ax.set_title(
        f"SOL/USD — HMM {n_components}-regime forecast  "
        f"(last {sol_close.index[-1].date()}, E[+7d]=${exp_price[-1]:.2f})",
        fontsize=11, loc="left",
    )
    ax.set_ylabel("USD")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry (python -m src.hmm.predict)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    with open(_CONFIG_PATH) as f:
        _config = yaml.safe_load(f)

    run(_config)
