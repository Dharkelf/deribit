"""XGBoost 1-week SOL/USD forecast — recursive multi-step with rolling feature update.

Design
------
Template Method: run() is the fixed pipeline skeleton.

Training target  : SOL log-return 1 step ahead (X[t] → lr[t+1]).
Feature set      : HMM-selected subset from best_features.json.
Models           : three XGBoost regressors per feature set —
                     base  : reg:squarederror  (mean)
                     q10   : reg:quantileerror alpha=0.10  (lower CI)
                     q90   : reg:quantileerror alpha=0.90  (upper CI)

Recursive forecast (168 steps)
-------------------------------
At each step j the feature row is updated before the next prediction:
  SOL_log_return  ← predicted value from step j-1
  SOL_vol_24h     ← std(sol_buffer[-24:])   (real history + predicted)
  SOL_vol_168h    ← std(sol_buffer[-168:])  (real history + predicted)
  BTC_log_return_lag_Xh ← shifted from known BTC history; 0 for future steps
  All other features (ETH, VIX, Fed, F&G, correlations) ← frozen at t=0

XGBoost+
--------
Searches optional features not selected by the HMM optimiser.  Ranks each by
improvement in in-data RMSE (last 168 h).  Trains an enhanced model with the
top-3 improvement features and runs its own recursive forecast.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.hmm.features import build_feature_matrix, load_common_dataframe
from src.hmm.optimizer import _viable_optional_features, load_best_features
from src.utils.paths import models_dir

logger = logging.getLogger(__name__)

_FORECAST_HOURS = 7 * 24   # 168 steps
_LONG_WINDOW    = 168
_SHORT_WINDOW   = 24

_XGB_MODEL_FILENAME      = "xgb_model.pkl"
_XGB_PLUS_MODEL_FILENAME = "xgb_plus_model.pkl"

_XGB_BASE_PARAMS: dict[str, Any] = {
    "objective":        "reg:squarederror",
    "n_estimators":     800,
    "learning_rate":    0.03,
    "max_depth":        5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}
_XGB_Q_PARAMS: dict[str, Any] = {
    **{k: v for k, v in _XGB_BASE_PARAMS.items() if k not in ("objective",)},
    "objective":     "reg:quantileerror",
    "n_estimators":  400,
}


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def _build_train_data(
    X_df: pd.DataFrame,
    sol_close: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) where y = SOL log-return 1 step ahead.

    X[i] = feature row at time i; y[i] = log(sol_close[i+1]/sol_close[i]).
    Last row is dropped (no known future close).
    """
    aligned = sol_close.reindex(X_df.index)
    lr_next = np.log(aligned / aligned.shift(1)).shift(-1)
    valid   = lr_next.notna()
    return (
        X_df.loc[valid].values.astype(np.float32),
        lr_next.loc[valid].values.astype(np.float32),
    )


def _train_model(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float | None = None,
    n_estimators: int | None = None,
) -> xgb.XGBRegressor:
    params = dict(_XGB_BASE_PARAMS if quantile is None else _XGB_Q_PARAMS)
    if quantile is not None:
        params["quantile_alpha"] = quantile
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Recursive forecast engine
# ─────────────────────────────────────────────────────────────────────────────


def _init_state(X_df: pd.DataFrame) -> dict[str, Any]:
    """Extract rolling buffers needed for recursive feature updates."""
    sol_lr  = X_df["SOL_log_return"].values
    btc_lr  = (
        X_df["BTC_log_return"].values
        if "BTC_log_return" in X_df.columns
        else np.zeros(len(X_df))
    )
    return {
        "sol_buffer": list(sol_lr[-_LONG_WINDOW:]),
        "btc_history": btc_lr,
    }


def _update_row(
    row: dict[str, float],
    pred_lr: float,
    state: dict[str, Any],
    step: int,
    feature_cols: list[str],
) -> dict[str, float]:
    """Return updated feature row for the next recursive step (step 0-indexed)."""
    state["sol_buffer"].append(pred_lr)
    state["sol_buffer"] = state["sol_buffer"][-_LONG_WINDOW:]
    row = dict(row)

    row["SOL_log_return"] = pred_lr
    if "SOL_vol_24h" in feature_cols:
        buf = state["sol_buffer"]
        row["SOL_vol_24h"] = float(np.std(buf[-_SHORT_WINDOW:]) if len(buf) >= _SHORT_WINDOW else np.std(buf))
    if "SOL_vol_168h" in feature_cols:
        row["SOL_vol_168h"] = float(np.std(state["sol_buffer"]))

    # BTC lag features: shift known history forward; zero-pad future BTC steps.
    # At forecast step `step`, we need features FOR time t + step + 1.
    # BTC_lag_X at time t+step+1 = BTC_lr[t + step + 1 - X].
    # Index into btc_history (length n): n - X + step.
    # If index < n: in known history. If index >= n: future BTC → use 0.
    btc_hist = state["btc_history"]
    n_hist   = len(btc_hist)
    for lag in (1, 2, 3, 6, 12, 18, 24):
        feat = f"BTC_log_return_lag_{lag}h"
        if feat not in feature_cols:
            continue
        hist_idx = n_hist - lag + step
        row[feat] = float(btc_hist[hist_idx]) if 0 <= hist_idx < n_hist else 0.0

    return row


def _recursive_forecast(
    base_model: xgb.XGBRegressor,
    q10_model: xgb.XGBRegressor,
    q90_model: xgb.XGBRegressor,
    X_df: pd.DataFrame,
    sol_close: pd.Series,
    n_steps: int = _FORECAST_HOURS,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Run recursive n_steps forecast.

    Returns (future_ts, exp_price, lo_price, hi_price).
    """
    feature_cols = list(X_df.columns)
    state        = _init_state(X_df)
    current_row  = X_df.iloc[-1].to_dict()

    base_lrs: list[float] = []
    q10_lrs:  list[float] = []
    q90_lrs:  list[float] = []

    for step in range(n_steps):
        x = np.array([[current_row[c] for c in feature_cols]], dtype=np.float32)
        base_lrs.append(float(base_model.predict(x)[0]))
        q10_lrs.append(float(q10_model.predict(x)[0]))
        q90_lrs.append(float(q90_model.predict(x)[0]))
        current_row = _update_row(current_row, base_lrs[-1], state, step, feature_cols)

    sol_last  = float(sol_close.iloc[-1])
    exp_price = sol_last * np.exp(np.cumsum(base_lrs))
    lo_price  = sol_last * np.exp(np.cumsum(q10_lrs))
    hi_price  = sol_last * np.exp(np.cumsum(q90_lrs))

    future_ts = pd.date_range(
        start=X_df.index[-1] + pd.Timedelta(hours=1),
        periods=n_steps, freq="1h", tz="UTC",
    )
    return future_ts, exp_price, lo_price, hi_price


# ─────────────────────────────────────────────────────────────────────────────
# In-data prediction
# ─────────────────────────────────────────────────────────────────────────────


def _in_data_predict(
    model: xgb.XGBRegressor,
    X_df: pd.DataFrame,
    sol_close: pd.Series,
    window: int = _FORECAST_HOURS,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, float]:
    """Predict last `window` SOL prices using real features (no recursion).

    Returns (timestamps, predicted_prices, actual_prices, rmse).
    """
    if len(X_df) < window + 2:
        raise ValueError(f"X_df too short for in-data window of {window}")

    X_in        = X_df.iloc[-(window + 1):-1].values.astype(np.float32)
    pred_lrs    = model.predict(X_in)
    ts          = X_df.index[-window:]
    start_price = float(sol_close.iloc[-(window + 1)])
    pred_prices = start_price * np.exp(np.cumsum(pred_lrs))
    actual      = sol_close.reindex(ts).values
    rmse        = float(np.sqrt(np.mean((pred_prices - actual) ** 2)))
    return ts, pred_prices, actual, rmse


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost+ feature search
# ─────────────────────────────────────────────────────────────────────────────


def _find_plus_features(
    X_df_full: pd.DataFrame,
    sol_close_full: pd.Series,
    base_subset: list[str],
    base_rmse: float,
    n_extra: int = 3,
) -> list[str]:
    """Find top n_extra optional features not in base_subset by in-data RMSE gain.

    Uses a quick 100-tree model per candidate; only features with positive
    RMSE improvement are kept.
    """
    # Candidates: viable features not already in the base model
    candidates = [
        f for f in X_df_full.columns
        if f not in base_subset and f != "SOL_log_return"
    ]
    if not candidates:
        logger.info("XGB+: no candidate features available")
        return []

    logger.info("XGB+ searching %d candidate features (base RMSE=%.4f) …", len(candidates), base_rmse)
    improvements: list[tuple[float, str]] = []

    for feat in candidates:
        cols  = ["SOL_log_return"] + [f for f in base_subset if f in X_df_full.columns] + [feat]
        X_sub = X_df_full[cols].dropna()
        if len(X_sub) < 200:
            continue
        sol_sub = sol_close_full.reindex(X_sub.index)
        X_tr, y_tr = _build_train_data(X_sub, sol_sub)
        m = _train_model(X_tr, y_tr, n_estimators=100)
        _, _, _, rmse = _in_data_predict(m, X_sub, sol_sub)
        delta = base_rmse - rmse
        improvements.append((delta, feat))
        logger.debug("  + %-35s  Δ=%.4f", feat, delta)

    improvements.sort(reverse=True)
    top = [f for delta, f in improvements[:n_extra] if delta > 0]
    logger.info("XGB+ adds %d feature(s): %s", len(top), top)
    return top


# ─────────────────────────────────────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────────────────────────────────────


def _save_models(bundle: tuple, config: dict, filename: str) -> None:
    path = models_dir(config) / filename
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("XGB models saved → %s", path)


def _load_models(config: dict, filename: str) -> tuple | None:
    path = models_dir(config) / filename
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def run(config: dict, *, force: bool = False) -> dict:
    """Full XGBoost pipeline.  Returns results dict consumed by visualize.py.

    Dict keys
    ---------
    in_data_ts       : DatetimeIndex  — last 168 h timestamps
    in_data_actual   : np.ndarray     — actual SOL close (last 168 h)
    in_data_pred     : np.ndarray     — XGB predicted close (last 168 h)
    in_data_rmse     : float
    future_ts        : DatetimeIndex  — next 168 h timestamps
    xgb_exp          : np.ndarray     — mean forecast
    xgb_lo           : np.ndarray     — q10 lower bound
    xgb_hi           : np.ndarray     — q90 upper bound
    xgb_plus_exp     : np.ndarray | None
    xgb_plus_in_pred : np.ndarray | None
    xgb_plus_rmse    : float | None
    plus_features    : list[str]
    """
    logger.info("=== XGBoost predict: 1-week SOL forecast ===")

    best = load_best_features(config)
    if best is None:
        raise FileNotFoundError("best_features.json missing; run HMM pipeline first")

    feature_subset: list[str] = best["feature_subset"]
    logger.info("Using %d HMM-selected features", len(feature_subset))

    df_common = load_common_dataframe(config)
    X_df      = build_feature_matrix(df_common.copy(), feature_subset)
    sol_close = df_common["SOL_close"].reindex(X_df.index)

    # ── Base models ───────────────────────────────────────────────────────────
    cached_base = None if force else _load_models(config, _XGB_MODEL_FILENAME)
    if cached_base is not None:
        base_model, q10_model, q90_model = cached_base
        logger.info("XGB base models loaded from cache")
    else:
        logger.info("Training XGBoost base models (%d rows × %d features) …",
                    len(X_df), len(X_df.columns))
        X_tr, y_tr = _build_train_data(X_df, sol_close)
        base_model = _train_model(X_tr, y_tr)
        q10_model  = _train_model(X_tr, y_tr, quantile=0.10)
        q90_model  = _train_model(X_tr, y_tr, quantile=0.90)
        _save_models((base_model, q10_model, q90_model), config, _XGB_MODEL_FILENAME)

    # ── In-data prediction (real features, no recursion) ─────────────────────
    in_data_ts, in_data_pred, in_data_actual, in_data_rmse = _in_data_predict(
        base_model, X_df, sol_close
    )
    logger.info("In-data RMSE (last 168 h): $%.2f", in_data_rmse)

    # ── Recursive 7-day forecast ──────────────────────────────────────────────
    future_ts, xgb_exp, xgb_lo, xgb_hi = _recursive_forecast(
        base_model, q10_model, q90_model, X_df, sol_close
    )
    logger.info(
        "XGB forecast: last=$%.2f  E[+7d]=$%.2f  CI=[$%.2f, $%.2f]",
        float(sol_close.iloc[-1]), xgb_exp[-1], xgb_lo[-1], xgb_hi[-1],
    )

    # ── XGBoost+ ──────────────────────────────────────────────────────────────
    viable_all     = _viable_optional_features(df_common)
    X_df_full      = build_feature_matrix(df_common.copy(), viable_all)
    sol_close_full = df_common["SOL_close"].reindex(X_df_full.index)

    cached_plus = None if force else _load_models(config, _XGB_PLUS_MODEL_FILENAME)
    if cached_plus is not None:
        plus_model, plus_q10, plus_q90, plus_features = cached_plus
        logger.info("XGB+ models loaded from cache (features: %s)", plus_features)
    else:
        plus_features = _find_plus_features(
            X_df_full, sol_close_full, feature_subset, in_data_rmse
        )
        if plus_features:
            plus_subset = feature_subset + plus_features
            X_plus      = build_feature_matrix(df_common.copy(), plus_subset)
            sol_plus    = df_common["SOL_close"].reindex(X_plus.index)
            X_tr_p, y_tr_p = _build_train_data(X_plus, sol_plus)
            plus_model  = _train_model(X_tr_p, y_tr_p)
            plus_q10    = _train_model(X_tr_p, y_tr_p, quantile=0.10)
            plus_q90    = _train_model(X_tr_p, y_tr_p, quantile=0.90)
        else:
            plus_model = plus_q10 = plus_q90 = None
        _save_models(
            (plus_model, plus_q10, plus_q90, plus_features),
            config, _XGB_PLUS_MODEL_FILENAME,
        )

    xgb_plus_exp = xgb_plus_in_pred = xgb_plus_rmse = None
    xgb_plus_in_ts = None
    if plus_model is not None:
        plus_subset = feature_subset + plus_features
        X_plus      = build_feature_matrix(df_common.copy(), plus_subset)
        sol_plus    = df_common["SOL_close"].reindex(X_plus.index)

        _, xgb_plus_exp, _, _ = _recursive_forecast(
            plus_model, plus_q10, plus_q90, X_plus, sol_plus
        )
        xgb_plus_in_ts, xgb_plus_in_pred, _, xgb_plus_rmse = _in_data_predict(
            plus_model, X_plus, sol_plus
        )
        logger.info(
            "XGB+ E[+7d]=$%.2f  in-data RMSE=$%.2f  added: %s",
            xgb_plus_exp[-1], xgb_plus_rmse, plus_features,
        )

    return {
        "in_data_ts":       in_data_ts,
        "in_data_actual":   in_data_actual,
        "in_data_pred":     in_data_pred,
        "in_data_rmse":     in_data_rmse,
        "future_ts":        future_ts,
        "xgb_exp":          xgb_exp,
        "xgb_lo":           xgb_lo,
        "xgb_hi":           xgb_hi,
        "xgb_plus_exp":     xgb_plus_exp,
        "xgb_plus_in_pred": xgb_plus_in_pred,
        "xgb_plus_in_ts":   xgb_plus_in_ts,
        "xgb_plus_rmse":    xgb_plus_rmse,
        "plus_features":    plus_features,
        "sol_last":         float(sol_close.iloc[-1]),
    }
