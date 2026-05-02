"""NeuralProphet 1-week SOL/USD forecast using HMM-selected features as regressors.

Design
------
Template Method: run() is the fixed pipeline skeleton.

Why NeuralProphet over classic Prophet
----------------------------------------
NeuralProphet (v0.8) adds an AR-Net autoregressive component and supports
lagged external regressors, making it suitable for hourly financial data.
Classic Prophet lacks autoregression and would be dominated by trend/seasonality.

Architecture choices
--------------------
  n_lags      = 168  : one week of SOL price history as AR input window
  n_forecasts = 168  : direct multi-step — ALL 168 steps predicted in one shot
                       from the current window; no recursive error accumulation
  Seasonality : daily (24 h) and weekly (168 h) — NeuralProphet exploits
                intraday and day-of-week patterns that XGBoost ignores
  Regressors  : HMM-selected features added as add_lagged_regressor(n_lags=1)
                At each time t the model sees regressor[t-1] as input.
                For the forecast window the regressor is forward-filled with
                its last known value (valid for slow-changing signals like Fed
                rate, VIX, fear-greed indices, rolling volatility).
  Quantiles   : [0.1, 0.9] for 80 % CI bands
  Normalise   : "soft" — NeuralProphet rescales y by its mean; robust for
                financial price levels with trend
  Target      : SOL close price (levels, not log-returns) — NeuralProphet
                handles trend natively; log-returns waste the trend component

In-data check
-------------
Run inference on the last 168 h of training data (real features) and compare
the model's fitted values against actual prices.  RMSE shown in the legend.

No NeuralProphet+ variant — lagged regressor set is small and NeuralProphet
already searches the best AR lag weights internally.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.hmm.features import build_feature_matrix, load_common_dataframe
from src.hmm.optimizer import load_best_features

logger = logging.getLogger(__name__)

_FORECAST_HOURS   = 48        # 2-day buffer; today's 24h extracted by UTC mask
_INDATA_HOURS     = 72        # in-data window shown in plot (last 3 days)
_NP_TRAIN_WINDOW  = 4000      # ~167 days; safe on M5 16 GB with n_forecasts=48


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────


def _build_np_df(
    X_df: pd.DataFrame,
    sol_close: pd.Series,
    feature_subset: list[str],
) -> pd.DataFrame:
    """Build NeuralProphet input DataFrame.

    NeuralProphet requires timezone-naive 'ds' and 'y' columns.
    Regressor columns are added alongside.
    """
    aligned = sol_close.reindex(X_df.index).dropna()
    idx     = aligned.index

    df = pd.DataFrame({
        "ds": idx.tz_localize(None),   # NeuralProphet does not support tz-aware
        "y":  aligned.values,
    })
    for feat in feature_subset:
        if feat in X_df.columns:
            df[feat] = X_df[feat].reindex(idx).values
    return df.reset_index(drop=True)


def _build_future_df(
    np_df: pd.DataFrame,
    model: Any,
    feature_subset: list[str],
    n_steps: int = _FORECAST_HOURS,
) -> pd.DataFrame:
    """Create the future dataframe with forward-filled regressor values."""
    future = model.make_future_dataframe(
        np_df,
        periods=n_steps,
        n_historic_predictions=True,
    )
    last_known = {feat: np_df[feat].iloc[-1] for feat in feature_subset if feat in np_df.columns}
    for feat, val in last_known.items():
        mask = future["ds"] > np_df["ds"].iloc[-1]
        future.loc[mask, feat] = val
    return future


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────


def _train_model(
    np_df: pd.DataFrame,
    feature_subset: list[str],
    config: dict,
) -> Any:
    """Fit a NeuralProphet model and return it."""
    from neuralprophet import NeuralProphet, set_log_level  # noqa: PLC0415

    set_log_level("ERROR")

    # n_lags=168 (1 week): safe now that n_forecasts=24 (not 168) keeps memory low
    n_lags_ar = min(168, len(np_df) // 4)

    model = NeuralProphet(
        n_forecasts=_FORECAST_HOURS,
        n_lags=n_lags_ar,
        n_changepoints=10,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        quantiles=[0.1, 0.9],
        normalize="soft",
        learning_rate=0.001,  # fixed LR avoids PyTorch 2.6 LR-finder checkpoint bug
        trainer_config={"accelerator": "cpu", "max_epochs": 40},
    )

    for feat in feature_subset:
        if feat in np_df.columns:
            model.add_lagged_regressor(feat, n_lags=1)

    model.fit(np_df, freq="h", progress=None)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Forecast extraction
# ─────────────────────────────────────────────────────────────────────────────


def _extract_forecast(
    forecast: pd.DataFrame,
    np_df: pd.DataFrame,
    sol_close: pd.Series,
    tz: str = "UTC",
    today_midnight: pd.Timestamp | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray,
           pd.DatetimeIndex, np.ndarray, np.ndarray, float]:
    """Split forecast into in-data (historic) and future portions.

    Returns
    -------
    in_data_ts, in_data_pred, in_data_actual, in_data_rmse,
    future_ts, np_exp, np_lo, np_hi
    """
    last_known_ds = np_df["ds"].iloc[-1]

    # Historic in-data window (last 72 h before last_known_ds)
    hist = forecast[forecast["ds"] <= last_known_ds].tail(_INDATA_HOURS).copy()

    def _to_utc(ds_series: pd.Series) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.to_datetime(ds_series)).tz_localize(tz)

    in_data_ts = _to_utc(hist["ds"])

    yhat_cols = [c for c in forecast.columns if c.startswith("yhat") and c[4:].isdigit()]
    yhat_cols.sort(key=lambda c: int(c[4:]))

    lo_cols = [c for c in forecast.columns if c.endswith(" 10.0%")]
    hi_cols = [c for c in forecast.columns if c.endswith(" 90.0%")]
    lo_cols.sort(key=lambda c: int(c.split("yhat")[1].split(" ")[0]))
    hi_cols.sort(key=lambda c: int(c.split("yhat")[1].split(" ")[0]))

    # In-data: use yhat1 from each historic row
    in_data_pred = hist["yhat1"].ffill().values.astype(float)

    # Future: yhat_i from the last in-data origin → last_known_ds + i*1h.
    # Build step timestamps; filter to today's full UTC calendar day.
    last_hist_row = forecast[forecast["ds"] == last_known_ds]
    n_steps = len(yhat_cols)
    all_step_ts = pd.date_range(
        start=pd.Timestamp(last_known_ds) + pd.Timedelta(hours=1),
        periods=n_steps, freq="1h",
    )  # tz-naive, matching NP convention

    _tm = today_midnight if today_midnight is not None else pd.Timestamp.now(tz="UTC").normalize()
    today_midnight_naive = _tm.tz_localize(None) if _tm.tzinfo is not None else _tm
    today_end_naive      = today_midnight_naive + pd.Timedelta(hours=23)
    today_mask           = (all_step_ts >= today_midnight_naive) & (all_step_ts <= today_end_naive)
    today_idx            = np.where(today_mask)[0]

    if len(today_idx) == 0 or last_hist_row.empty or not yhat_cols:
        np_exp    = np.full(24, np.nan)
        np_lo     = np_exp.copy()
        np_hi     = np_exp.copy()
        future_ts = pd.DatetimeIndex([], tz=tz)
    else:
        row       = last_hist_row.iloc[0]
        np_exp    = np.array([row.get(yhat_cols[i], np.nan) for i in today_idx], dtype=float)
        np_lo     = np.array([row.get(lo_cols[i],   np.nan) for i in today_idx], dtype=float) if lo_cols else np_exp.copy()
        np_hi     = np.array([row.get(hi_cols[i],   np.nan) for i in today_idx], dtype=float) if hi_cols else np_exp.copy()
        future_ts = pd.DatetimeIndex(all_step_ts[today_mask]).tz_localize(tz)

    # Actual prices for RMSE (align to in_data_ts)
    actual = sol_close.reindex(in_data_ts)
    valid  = ~(np.isnan(in_data_pred) | actual.isna().values)
    rmse   = float(np.sqrt(np.mean((in_data_pred[valid] - actual.values[valid]) ** 2)))

    return in_data_ts, in_data_pred, actual.values, rmse, future_ts, np_exp, np_lo, np_hi


# ─────────────────────────────────────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def run(config: dict) -> dict:
    """Full NeuralProphet pipeline.  Returns results dict for visualize.py.

    Dict keys
    ---------
    in_data_ts     : DatetimeIndex  — last 168 h timestamps
    in_data_actual : np.ndarray     — actual SOL close
    in_data_pred   : np.ndarray     — NP fitted values
    in_data_rmse   : float
    future_ts      : DatetimeIndex  — next 168 h timestamps
    np_exp         : np.ndarray     — mean forecast
    np_lo          : np.ndarray     — q10 lower bound
    np_hi          : np.ndarray     — q90 upper bound
    sol_last       : float
    """
    logger.info("=== NeuralProphet predict: 1-week SOL forecast ===")

    best = load_best_features(config)
    if best is None:
        raise FileNotFoundError("best_features.json missing; run HMM pipeline first")

    feature_subset: list[str] = best["feature_subset"]
    logger.info("Using %d HMM-selected features as lagged regressors", len(feature_subset))

    df_common = load_common_dataframe(config)
    X_df      = build_feature_matrix(df_common.copy(), feature_subset)
    sol_close = df_common["SOL_close"].reindex(X_df.index)

    # ── Align to yesterday 23:00 UTC so forecast starts at today 00:00 UTC ──
    today_midnight = config.get("_today", pd.Timestamp.now(tz="UTC").normalize())
    cutoff         = config.get("_cutoff", today_midnight - pd.Timedelta(hours=1))
    if X_df.index[-1] > cutoff:
        X_df      = X_df.loc[X_df.index <= cutoff]
        sol_close = sol_close.loc[sol_close.index <= cutoff]
        logger.info("Data capped at %s UTC; forecast covers today 00:00–23:00 UTC", cutoff)
    elif X_df.index[-1] < cutoff - pd.Timedelta(hours=23):
        logger.warning(
            "Data ends at %s — stale. Run 'python main.py collect' for today's forecast.",
            X_df.index[-1].date(),
        )

    np_df = _build_np_df(X_df, sol_close, feature_subset)
    np_df = np_df.iloc[-_NP_TRAIN_WINDOW:]  # cap to recent window to avoid OOM

    # NeuralProphet's Trainer is not safely serializable with PyTorch ≥2.6; always retrain.
    logger.info(
        "Training NeuralProphet (%d rows, AR n_lags=%d, n_forecasts=%d, %d regressors) …",
        len(np_df), min(168, len(np_df) // 4), _FORECAST_HOURS, len(feature_subset),
    )
    model = _train_model(np_df, feature_subset, config)

    future_df = _build_future_df(np_df, model, feature_subset)
    forecast  = model.predict(future_df)

    (
        in_data_ts, in_data_pred, in_data_actual, in_data_rmse,
        future_ts, np_exp, np_lo, np_hi,
    ) = _extract_forecast(forecast, np_df, sol_close, today_midnight=today_midnight)

    logger.info(
        "NP forecast: last=$%.2f  E[+7d]=$%.2f  CI=[$%.2f, $%.2f]  in-data RMSE=$%.2f",
        float(sol_close.iloc[-1]),
        float(np_exp[-1]) if not np.isnan(np_exp[-1]) else 0,
        float(np_lo[-1])  if not np.isnan(np_lo[-1])  else 0,
        float(np_hi[-1])  if not np.isnan(np_hi[-1])  else 0,
        in_data_rmse,
    )

    return {
        "in_data_ts":     in_data_ts,
        "in_data_actual": in_data_actual,
        "in_data_pred":   in_data_pred,
        "in_data_rmse":   in_data_rmse,
        "future_ts":      future_ts,
        "np_exp":         np_exp,
        "np_lo":          np_lo,
        "np_hi":          np_hi,
        "sol_last":       float(sol_close.iloc[-1]),
    }
