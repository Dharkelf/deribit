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
  n_forecasts = 48   : direct multi-step — 48 steps predicted in one shot;
                       no recursive error accumulation; today's 24h extracted by mask
  Seasonality : daily (24 h) and weekly (168 h)
  Regressors  : HMM-selected features + XGB+ additions (if they improve adj-R²)
                added as add_lagged_regressor(n_lags=1); each regressor value
                at t-1 is used to predict SOL close at t. For forecast horizon
                the regressor is forward-filled with its last known value.
  Quantiles   : [0.1, 0.9] for 80 % CI bands
  Normalise   : "soft" — rescales y by its mean; robust for price levels with trend
  Target      : SOL close price (levels, not log-returns) — NeuralProphet handles
                trend natively; log-returns waste the trend component

NP+ variant
-----------
After fitting the base model (HMM features), a second model is trained on
HMM features + XGB+ features. If in-data adj-R² improves, the NP+ model is
used and the added features are annotated in the plot.

Training window
---------------
All available data is used (no cap). On Apple M5 16 GB, 8826 rows × 24 features
consumes ~170 MB — well within budget. Training time is ~50–60 s per model.
The current dataset covers ~370 days; 3-year history is a planned extension.

In-data check
-------------
Run inference on the last 72 h of training data (real features, no recursion)
and compare fitted values against actual prices. Reports both RMSE and adj-R².
"""

import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd

from src.hmm.features import build_feature_matrix, load_common_dataframe
from src.hmm.optimizer import load_best_features
from src.utils.paths import models_dir

logger = logging.getLogger(__name__)

_FORECAST_HOURS = 48  # 2-day buffer; today's 24 h extracted by UTC mask
_INDATA_HOURS = 72  # in-data window shown in plot (last 3 days)
_MAX_EPOCHS = 60  # up from 40 — better convergence on full dataset


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _adj_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Adjusted R² — penalises each additional feature."""
    n = len(y_true)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12 or n <= n_features + 1:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))


def _load_xgb_plus_features(config: dict) -> list[str]:
    """Return XGB+ added features from cached model pkl, or [] if unavailable."""
    pkl = models_dir(config) / "xgb_plus_model.pkl"
    if not pkl.exists():
        return []
    try:
        with open(pkl, "rb") as f:
            _, _, _, plus_features = pickle.load(f)
        return list(plus_features or [])
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────


def _build_np_df(
    X_df: pd.DataFrame,
    sol_close: pd.Series,
    feature_subset: list[str],
) -> pd.DataFrame:
    """Build NeuralProphet input DataFrame (tz-naive ds + y + regressor columns)."""
    aligned = sol_close.reindex(X_df.index).dropna()
    idx = aligned.index
    df = pd.DataFrame(
        {
            "ds": idx.tz_localize(None),
            "y": aligned.values,
        }
    )
    for feat in feature_subset:
        if feat in X_df.columns:
            col = X_df[feat]
            # X_df[feat] returns a DataFrame when column names are duplicated;
            # add_lagged_regressor requires a 1D array.
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            df[feat] = col.reindex(idx).to_numpy()
    return df.reset_index(drop=True)


def _build_future_df(
    np_df: pd.DataFrame,
    model: Any,
    feature_subset: list[str],
    n_steps: int = _FORECAST_HOURS,
) -> pd.DataFrame:
    """Create the future dataframe with forward-filled regressor values."""
    future = model.make_future_dataframe(
        np_df, periods=n_steps, n_historic_predictions=True
    )
    last_known = {
        feat: np_df[feat].iloc[-1] for feat in feature_subset if feat in np_df.columns
    }
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
        learning_rate=0.001,  # fixed: avoids PyTorch ≥2.6 LR-finder checkpoint bug
        trainer_config={"accelerator": "cpu", "max_epochs": _MAX_EPOCHS},
    )

    for feat in feature_subset:
        if feat in np_df.columns:
            # Defensive: verify column is 1D before handing to NeuralProphet.
            # _build_np_df already guards against duplicate-column DataFrames;
            # this assertion catches any future regression at the call site.
            col_vals = np_df[feat].to_numpy()
            if col_vals.ndim != 1:
                logger.warning(
                    "Skipping regressor '%s': expected 1D, got shape %s",
                    feat,
                    col_vals.shape,
                )
                continue
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
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    float,
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Split forecast into in-data (historic) and future portions.

    Returns
    -------
    in_data_ts, in_data_pred, in_data_actual, in_data_rmse,
    future_ts, np_exp, np_lo, np_hi
    """
    last_known_ds = np_df["ds"].iloc[-1]

    hist = forecast[forecast["ds"] <= last_known_ds].tail(_INDATA_HOURS).copy()

    def _to_utc(ds_series: pd.Series) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.to_datetime(ds_series)).tz_localize(tz)

    in_data_ts = _to_utc(hist["ds"])

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

    in_data_pred = hist["yhat1"].ffill().values.astype(float)

    last_hist_row = forecast[forecast["ds"] == last_known_ds]
    n_steps = len(yhat_cols)
    all_step_ts = pd.date_range(
        start=pd.Timestamp(last_known_ds) + pd.Timedelta(hours=1),
        periods=n_steps,
        freq="1h",
    )

    _tm = (
        today_midnight
        if today_midnight is not None
        else pd.Timestamp.now(tz="UTC").normalize()
    )
    today_midnight_naive = _tm.tz_localize(None) if _tm.tzinfo is not None else _tm
    today_end_naive = today_midnight_naive + pd.Timedelta(hours=23)
    today_mask = (all_step_ts >= today_midnight_naive) & (
        all_step_ts <= today_end_naive
    )
    today_idx = np.where(today_mask)[0]

    if len(today_idx) == 0 or last_hist_row.empty or not yhat_cols:
        np_exp = np.full(24, np.nan)
        np_lo = np_exp.copy()
        np_hi = np_exp.copy()
        future_ts = pd.DatetimeIndex([], tz=tz)
    else:
        row = last_hist_row.iloc[0]
        np_exp = np.array(
            [row.get(yhat_cols[i], np.nan) for i in today_idx], dtype=float
        )
        np_lo = (
            np.array([row.get(lo_cols[i], np.nan) for i in today_idx], dtype=float)
            if lo_cols
            else np_exp.copy()
        )
        np_hi = (
            np.array([row.get(hi_cols[i], np.nan) for i in today_idx], dtype=float)
            if hi_cols
            else np_exp.copy()
        )
        future_ts = pd.DatetimeIndex(all_step_ts[today_mask]).tz_localize(tz)

    actual = sol_close.reindex(in_data_ts)
    valid = ~(np.isnan(in_data_pred) | actual.isna().values)
    rmse = float(np.sqrt(np.mean((in_data_pred[valid] - actual.values[valid]) ** 2)))

    return (
        in_data_ts,
        in_data_pred,
        actual.values,
        rmse,
        future_ts,
        np_exp,
        np_lo,
        np_hi,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def run(config: dict) -> dict:
    """Full NeuralProphet pipeline.  Returns results dict for visualize.py.

    Dict keys
    ---------
    in_data_ts        : DatetimeIndex  — last 72 h timestamps
    in_data_actual    : np.ndarray     — actual SOL close
    in_data_pred      : np.ndarray     — NP fitted values (yhat1)
    in_data_rmse      : float
    in_data_adj_r2    : float
    future_ts         : DatetimeIndex  — today 00:00–23:00 UTC
    np_exp            : np.ndarray     — mean forecast
    np_lo             : np.ndarray     — q10 lower bound
    np_hi             : np.ndarray     — q90 upper bound
    np_plus_features  : list[str]      — XGB+ features added if NP+ improved adj-R²
    sol_last          : float
    """
    logger.info("=== NeuralProphet predict: 1-week SOL forecast ===")

    best = load_best_features(config)
    if best is None:
        raise FileNotFoundError("best_features.json missing; run HMM pipeline first")

    feature_subset: list[str] = best["feature_subset"]
    logger.info(
        "Using %d HMM-selected features as lagged regressors", len(feature_subset)
    )

    df_common = load_common_dataframe(config)
    X_df = build_feature_matrix(df_common.copy(), feature_subset)
    sol_close = df_common["SOL_close"].reindex(X_df.index)

    today_midnight = config.get("_today", pd.Timestamp.now(tz="UTC").normalize())
    cutoff = config.get("_cutoff", today_midnight - pd.Timedelta(hours=1))
    if X_df.index[-1] > cutoff:
        X_df = X_df.loc[X_df.index <= cutoff]
        sol_close = sol_close.loc[sol_close.index <= cutoff]
        logger.info(
            "Data capped at %s UTC; forecast covers today 00:00–23:00 UTC", cutoff
        )
    elif X_df.index[-1] < cutoff - pd.Timedelta(hours=23):
        logger.warning(
            "Data ends at %s — stale. Run 'python main.py collect' first.",
            X_df.index[-1].date(),
        )

    # ── Base NP (all available data, no window cap) ───────────────────────────
    np_df = _build_np_df(X_df, sol_close, feature_subset)
    n_lags_ar = min(168, len(np_df) // 4)
    logger.info(
        "Training NP base (%d rows, AR n_lags=%d, n_forecasts=%d, %d regressors) …",
        len(np_df),
        n_lags_ar,
        _FORECAST_HOURS,
        len(feature_subset),
    )
    base_model = _train_model(np_df, feature_subset, config)
    base_future_df = _build_future_df(np_df, base_model, feature_subset)
    base_forecast = base_model.predict(base_future_df)
    (
        in_data_ts,
        in_data_pred,
        in_data_actual,
        in_data_rmse,
        future_ts,
        np_exp,
        np_lo,
        np_hi,
    ) = _extract_forecast(
        base_forecast, np_df, sol_close, today_midnight=today_midnight
    )

    valid = ~(np.isnan(in_data_pred) | np.isnan(in_data_actual))
    base_adj_r2 = _adj_r2(
        in_data_actual[valid], in_data_pred[valid], len(feature_subset)
    )
    in_data_adj_r2 = base_adj_r2
    np_plus_features: list[str] = []

    # ── NP+ (HMM features + XGB+ additions) ──────────────────────────────────
    plus_candidates = _load_xgb_plus_features(config)
    if plus_candidates:
        # Exclude features already in the HMM subset — NeuralProphet raises
        # "Name already used" if add_lagged_regressor is called twice for the
        # same feature name (e.g. ETH_vol_168h appearing in both sets).
        hmm_set = set(feature_subset)
        new_candidates = [f for f in plus_candidates if f not in hmm_set]
        plus_subset = feature_subset + new_candidates
        try:
            X_plus = build_feature_matrix(df_common.copy(), plus_subset)
            sol_plus = df_common["SOL_close"].reindex(X_plus.index)
            if X_plus.index[-1] > cutoff:
                X_plus = X_plus.loc[X_plus.index <= cutoff]
                sol_plus = sol_plus.loc[sol_plus.index <= cutoff]

            np_df_plus = _build_np_df(X_plus, sol_plus, plus_subset)
            logger.info(
                "Training NP+ (%d rows, %d regressors: %d HMM + %d new XGB+) …",
                len(np_df_plus),
                len(plus_subset),
                len(feature_subset),
                len(new_candidates),
            )
            plus_model = _train_model(np_df_plus, plus_subset, config)
            plus_future_df = _build_future_df(np_df_plus, plus_model, plus_subset)
            plus_forecast = plus_model.predict(plus_future_df)
            (
                p_in_ts,
                p_in_pred,
                p_in_actual,
                p_rmse,
                p_future_ts,
                p_exp,
                p_lo,
                p_hi,
            ) = _extract_forecast(
                plus_forecast, np_df_plus, sol_plus, today_midnight=today_midnight
            )

            p_valid = ~(np.isnan(p_in_pred) | np.isnan(p_in_actual))
            plus_adj_r2 = _adj_r2(
                p_in_actual[p_valid], p_in_pred[p_valid], len(plus_subset)
            )

            logger.info(
                "NP base adj-R²=%.4f  RMSE=$%.2f  |  NP+ adj-R²=%.4f  RMSE=$%.2f  → using %s",
                base_adj_r2,
                in_data_rmse,
                plus_adj_r2,
                p_rmse,
                "NP+" if plus_adj_r2 > base_adj_r2 else "base",
            )

            if plus_adj_r2 > base_adj_r2:
                in_data_ts, in_data_pred, in_data_actual, in_data_rmse = (
                    p_in_ts,
                    p_in_pred,
                    p_in_actual,
                    p_rmse,
                )
                future_ts, np_exp, np_lo, np_hi = p_future_ts, p_exp, p_lo, p_hi
                in_data_adj_r2 = plus_adj_r2
                np_plus_features = plus_candidates

        except Exception as exc:
            logger.warning("NP+ evaluation failed (%s) — using base model", exc)

    logger.info(
        "NP forecast: last=$%.2f  E[+1d]=$%.2f  CI=[$%.2f, $%.2f]"
        "  in-data RMSE=$%.2f  adj-R²=%.4f",
        float(sol_close.iloc[-1]),
        float(np_exp[-1]) if len(np_exp) and not np.isnan(np_exp[-1]) else 0.0,
        float(np_lo[-1]) if len(np_lo) and not np.isnan(np_lo[-1]) else 0.0,
        float(np_hi[-1]) if len(np_hi) and not np.isnan(np_hi[-1]) else 0.0,
        in_data_rmse,
        in_data_adj_r2,
    )

    return {
        "in_data_ts": in_data_ts,
        "in_data_actual": in_data_actual,
        "in_data_pred": in_data_pred,
        "in_data_rmse": in_data_rmse,
        "in_data_adj_r2": in_data_adj_r2,
        "future_ts": future_ts,
        "np_exp": np_exp,
        "np_lo": np_lo,
        "np_hi": np_hi,
        "np_plus_features": np_plus_features,
        "sol_last": float(sol_close.iloc[-1]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward backtest fold (full 48-step output, no today-mask)
# ─────────────────────────────────────────────────────────────────────────────


def predict_backtest_fold(
    config: dict,
    cutoff_ts: pd.Timestamp,
    feature_subset: list[str],
    df_common: pd.DataFrame,
) -> dict[str, object]:
    """Train NP on data ≤ cutoff_ts and return the full 48-step forecast.

    Designed for walk-forward backtesting.  Pass df_common pre-loaded once
    outside the fold loop to avoid repeated file I/O.

    Returns
    -------
    future_ts_48   : DatetimeIndex  cutoff_ts+1h … cutoff_ts+48h (UTC)
    np_exp_48      : ndarray(48,)   mean forecast
    np_lo_48       : ndarray(48,)   q10
    np_hi_48       : ndarray(48,)   q90
    sol_last       : float          SOL close at cutoff_ts
    in_data_rmse   : float
    in_data_adj_r2 : float
    """
    X_df = build_feature_matrix(df_common.copy(), feature_subset)
    sol_close = df_common["SOL_close"].reindex(X_df.index)

    X_df = X_df.loc[X_df.index <= cutoff_ts]
    sol_close = sol_close.loc[sol_close.index <= cutoff_ts]

    np_df = _build_np_df(X_df, sol_close, feature_subset)
    model = _train_model(np_df, feature_subset, config)
    future_df = _build_future_df(np_df, model, feature_subset)
    forecast = model.predict(future_df)

    last_known_ds = np_df["ds"].iloc[-1]
    sol_last = float(sol_close.iloc[-1])

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

    n_steps = len(yhat_cols)
    future_ts_48 = pd.DatetimeIndex(
        pd.date_range(
            start=pd.Timestamp(last_known_ds) + pd.Timedelta(hours=1),
            periods=n_steps,
            freq="1h",
        )
    ).tz_localize("UTC")

    last_row = forecast[forecast["ds"] == last_known_ds]
    if last_row.empty or not yhat_cols:
        nan_arr = np.full(max(n_steps, 48), np.nan)
        return {
            "future_ts_48": future_ts_48,
            "np_exp_48": nan_arr,
            "np_lo_48": nan_arr.copy(),
            "np_hi_48": nan_arr.copy(),
            "sol_last": sol_last,
            "in_data_rmse": np.nan,
            "in_data_adj_r2": np.nan,
        }

    row = last_row.iloc[0]
    np_exp_48 = np.array([row.get(c, np.nan) for c in yhat_cols], dtype=float)
    np_lo_48 = (
        np.array([row.get(c, np.nan) for c in lo_cols], dtype=float)
        if lo_cols
        else np_exp_48.copy()
    )
    np_hi_48 = (
        np.array([row.get(c, np.nan) for c in hi_cols], dtype=float)
        if hi_cols
        else np_exp_48.copy()
    )

    # In-data quality (last 72 h)
    hist = forecast[forecast["ds"] <= last_known_ds].tail(_INDATA_HOURS)
    in_data_ts_local = pd.DatetimeIndex(pd.to_datetime(hist["ds"])).tz_localize("UTC")
    in_data_pred_local = hist["yhat1"].ffill().values.astype(float)
    actual_vals = sol_close.reindex(in_data_ts_local).values
    valid = ~(np.isnan(in_data_pred_local) | np.isnan(actual_vals))
    if valid.any():
        in_data_rmse: float = float(
            np.sqrt(np.mean((in_data_pred_local[valid] - actual_vals[valid]) ** 2))
        )
        in_data_adj_r2: float = _adj_r2(
            actual_vals[valid], in_data_pred_local[valid], len(feature_subset)
        )
    else:
        in_data_rmse = np.nan
        in_data_adj_r2 = np.nan

    return {
        "future_ts_48": future_ts_48,
        "np_exp_48": np_exp_48,
        "np_lo_48": np_lo_48,
        "np_hi_48": np_hi_48,
        "sol_last": sol_last,
        "in_data_rmse": in_data_rmse,
        "in_data_adj_r2": in_data_adj_r2,
    }
