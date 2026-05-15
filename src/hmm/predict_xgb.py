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
Searches optional features not selected by the HMM optimiser.  Each candidate
is evaluated independently (base + one feature) to avoid cross-feature row
collapse from sparse features (e.g. Max Pain).  Selection criterion: adjusted
R² improvement vs a 100-tree baseline — adj R² penalises each added feature,
providing a natural soft cap.  A dominance check then prunes added features
whose importances exceed the HMM-base mean.  All selected features are listed
in the plot annotation.
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
from src.utils.paths import models_dir, raw_dir

logger = logging.getLogger(__name__)

_FORECAST_HOURS  = 7 * 24   # 168 steps (full model horizon)
_DISPLAY_HOURS   = 24       # hours shown in plot forecast
_INDATA_HOURS    = 72       # hours shown in plot in-data window
_LONG_WINDOW     = 168
_SHORT_WINDOW   = 24

_XGB_MODEL_FILENAME      = "xgb_model.pkl"
_XGB_PLUS_MODEL_FILENAME = "xgb_plus_model.pkl"

_XGB_BASE_PARAMS: dict[str, Any] = {
    "objective":        "reg:squarederror",
    "n_estimators":     1500,
    "learning_rate":    0.015,
    "max_depth":        5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,    # reduces overfitting on financial time-series
    "reg_alpha":        0.1,  # L1 — shrinks noisy leaf weights
    "reg_lambda":       1.5,  # L2 — slightly stronger than default (1.0)
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
    "tree_method":      "hist",  # explicit; uses NEON on Apple Silicon
}
_XGB_Q_PARAMS: dict[str, Any] = {
    **{k: v for k, v in _XGB_BASE_PARAMS.items() if k not in ("objective",)},
    "objective":     "reg:quantileerror",
    "n_estimators":  800,  # more trees → tighter CI bands
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
    # sol_buffer is seeded with the last 168 real SOL log-returns so that
    # SOL_vol_24h / SOL_vol_168h at step 0 reflect actual recent volatility,
    # not a cold-start zero buffer.
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

    # BTC lag features: shift known history forward; persist last known value
    # for future BTC steps (forward-fill) rather than zero-padding, which
    # would create a discontinuous feature jump at the history boundary.
    # At forecast step `step`, we need features FOR time t + step + 1.
    # BTC_lag_X at time t+step+1 = BTC_lr[t + step + 1 - X].
    # Index into btc_history (length n): n - X + step.
    # If index < n: in known history. If index >= n: future BTC → persist last.
    btc_hist = state["btc_history"]
    n_hist   = len(btc_hist)
    for lag in (1, 2, 3, 6, 12, 18, 24):
        feat = f"BTC_log_return_lag_{lag}h"
        if feat not in feature_cols:
            continue
        hist_idx = n_hist - lag + step
        if 0 <= hist_idx < n_hist:
            row[feat] = float(btc_hist[hist_idx])
        else:
            row[feat] = float(btc_hist[-1])

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


def _adj_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Adjusted R² — penalises each additional feature so adding one only helps
    when the fit improvement exceeds the degrees-of-freedom cost."""
    n = len(y_true)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    # Guard: n <= n_features + 1 makes denominator (n - n_features - 1) ≤ 0,
    # which would produce a nonsensical negative infinity result.
    if ss_tot < 1e-12 or n <= n_features + 1:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))


def _in_data_predict(
    model: xgb.XGBRegressor,
    X_df: pd.DataFrame,
    sol_close: pd.Series,
    window: int = _FORECAST_HOURS,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, float, float]:
    """Predict last `window` SOL prices using real features (no recursion).

    Returns (timestamps, predicted_prices, actual_prices, rmse, adj_r2).
    """
    if len(X_df) < window + 2:
        raise ValueError(f"X_df too short for in-data window of {window}")

    X_in        = X_df.iloc[-(window + 1):-1].values.astype(np.float32)
    pred_lrs    = model.predict(X_in)
    ts          = X_df.index[-window:]
    start_price = float(sol_close.iloc[-(window + 1)])
    pred_prices = start_price * np.exp(np.cumsum(pred_lrs))
    actual      = sol_close.reindex(ts).values
    valid       = ~np.isnan(actual)
    rmse        = float(np.sqrt(np.mean((pred_prices[valid] - actual[valid]) ** 2)))
    ar2         = _adj_r2(actual[valid], pred_prices[valid], X_df.shape[1])
    return ts, pred_prices, actual, rmse, ar2


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost+ feature search
# ─────────────────────────────────────────────────────────────────────────────


def _base_dominates(
    model: xgb.XGBRegressor,
    col_names: list[str],
    base_subset: list[str],
) -> bool:
    """Return True if HMM-selected base features have ≥ mean importance vs added ones."""
    imp      = model.feature_importances_
    base_set = set(base_subset) | {"SOL_log_return"}
    base_imp = [imp[i] for i, f in enumerate(col_names) if f in base_set]
    plus_imp = [imp[i] for i, f in enumerate(col_names) if f not in base_set]
    if not plus_imp:
        return True
    return float(np.mean(base_imp)) >= float(np.mean(plus_imp))


def _find_plus_features(
    df_common: pd.DataFrame,
    sol_close: pd.Series,
    base_subset: list[str],
    viable_all: list[str],
    min_rows: int = 200,
) -> list[str]:
    """Find optional features beyond the HMM base subset that improve adj R².

    Each candidate is evaluated independently (base_subset + one candidate)
    so sparse features (Max Pain) don't collapse the window for other candidates.
    Selection criterion: adjusted R² improvement vs a 100-tree baseline model
    — adj R² penalises each additional feature, making n_extra a soft natural
    limit rather than a hard cap.

    After selection, a dominance check enforces that HMM-regime features retain
    higher mean XGBoost importance than the added features. Lowest-importance
    added features are pruned until the condition holds.
    """
    candidates = [f for f in viable_all if f not in base_subset and f != "SOL_log_return"]
    if not candidates:
        logger.info("XGB+: no candidate features available")
        return []

    # 100-tree baseline for fair comparison (full model uses 800 trees)
    try:
        X_base = build_feature_matrix(df_common.copy(), list(base_subset))
    except ValueError:
        logger.info("XGB+: could not build base matrix — skipping")
        return []
    sol_base = sol_close.reindex(X_base.index)
    X_tr_b, y_tr_b = _build_train_data(X_base, sol_base)
    m_base   = _train_model(X_tr_b, y_tr_b, n_estimators=100)
    _, _, _, _, quick_base_ar2 = _in_data_predict(m_base, X_base, sol_base)

    logger.info(
        "XGB+ searching %d candidate features (quick base adj-R²=%.4f) …",
        len(candidates), quick_base_ar2,
    )
    gains: list[tuple[float, str]] = []

    for feat in candidates:
        trial_subset = list(base_subset) + [feat]
        try:
            X_sub = build_feature_matrix(df_common.copy(), trial_subset)
        except ValueError:
            continue
        if len(X_sub) < min_rows:
            logger.debug("  skip %-35s — %d rows < %d", feat, len(X_sub), min_rows)
            continue
        sol_sub = sol_close.reindex(X_sub.index)
        X_tr, y_tr = _build_train_data(X_sub, sol_sub)
        m        = _train_model(X_tr, y_tr, n_estimators=100)
        _, _, _, _, ar2 = _in_data_predict(m, X_sub, sol_sub)
        delta = ar2 - quick_base_ar2
        gains.append((delta, feat))
        logger.debug("  + %-35s  Δadj-R²=%.4f  rows=%d", feat, delta, len(X_sub))

    # All candidates with positive adj R² gain (no hard cap — adj R² penalises count)
    gains.sort(reverse=True)
    selected = [f for delta, f in gains if delta > 0]
    logger.info("XGB+ adj-R² candidates: %s", selected)

    if not selected:
        return []

    # Dominance check: HMM-base feature importances must dominate the added ones.
    # Prune lowest-importance added features until dominance holds.
    check_subset = list(base_subset) + selected
    assert len(set(check_subset)) == len(check_subset), (
        f"Duplicate features in XGB+ subset: {check_subset}"
    )
    try:
        X_chk = build_feature_matrix(df_common.copy(), check_subset)
    except ValueError:
        return []
    sol_chk    = sol_close.reindex(X_chk.index)
    X_tr_c, y_tr_c = _build_train_data(X_chk, sol_chk)
    m_chk      = _train_model(X_tr_c, y_tr_c, n_estimators=100)
    col_names  = list(X_chk.columns)

    while selected and not _base_dominates(m_chk, col_names, base_subset):
        imp       = m_chk.feature_importances_
        base_set  = set(base_subset) | {"SOL_log_return"}
        added_imp = {f: imp[col_names.index(f)] for f in selected if f in col_names}
        drop      = min(added_imp, key=added_imp.get)
        logger.debug("  dominance pruning: dropping %s (imp=%.4f)", drop, added_imp[drop])
        selected.remove(drop)
        if not selected:
            break
        check_subset = list(base_subset) + selected
        try:
            X_chk = build_feature_matrix(df_common.copy(), check_subset)
        except ValueError:
            break
        sol_chk        = sol_close.reindex(X_chk.index)
        X_tr_c, y_tr_c = _build_train_data(X_chk, sol_chk)
        m_chk          = _train_model(X_tr_c, y_tr_c, n_estimators=100)
        col_names      = list(X_chk.columns)

    logger.info("XGB+ adds %d feature(s) after dominance check: %s", len(selected), selected)
    return selected


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


def _yesterday_end() -> pd.Timestamp:
    """Return yesterday 23:00 UTC — the last full hour before today's midnight."""
    return pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(hours=1)


def _cache_is_stale(config: dict, filename: str) -> bool:
    """True if the pkl is older than BTC.parquet, meaning data was refreshed."""
    pkl  = models_dir(config) / filename
    ref  = raw_dir(config) / "BTC.parquet"
    if not pkl.exists():
        return True
    if not ref.exists():
        return False
    return pkl.stat().st_mtime < ref.stat().st_mtime


def _filter_24h_features(feature_subset: list[str]) -> list[str]:
    """Drop features with >24h lookback (168h vol, correlations, momentum).

    These are useful for regime detection but irrelevant for a 24h-horizon
    recursive forecast where only recent context matters.
    """
    return [
        f for f in feature_subset
        if "_168h" not in f and "_momentum" not in f
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def run(config: dict, *, force: bool = False) -> dict:
    """Full XGBoost pipeline.  Returns results dict consumed by visualize.py.

    Dict keys
    ---------
    in_data_ts        : DatetimeIndex  — last 72 h timestamps
    in_data_actual    : np.ndarray     — actual SOL close (last 72 h)
    in_data_pred      : np.ndarray     — XGB predicted close (last 72 h)
    in_data_rmse      : float
    in_data_adj_r2    : float
    future_ts         : DatetimeIndex  — today 00:00–23:00 UTC
    xgb_exp           : np.ndarray     — mean forecast
    xgb_lo            : np.ndarray     — q10 lower bound
    xgb_hi            : np.ndarray     — q90 upper bound
    xgb_plus_exp      : np.ndarray | None
    xgb_plus_in_pred  : np.ndarray | None
    xgb_plus_rmse     : float | None
    xgb_plus_adj_r2   : float | None
    plus_features     : list[str]
    """
    logger.info("=== XGBoost predict: 1-week SOL forecast ===")

    best = load_best_features(config)
    if best is None:
        raise FileNotFoundError("best_features.json missing; run HMM pipeline first")

    feature_subset: list[str] = _filter_24h_features(best["feature_subset"])
    logger.info(
        "Using %d HMM-selected features (≤24h lookback, filtered from %d)",
        len(feature_subset), len(best["feature_subset"]),
    )

    df_common = load_common_dataframe(config)
    X_df      = build_feature_matrix(df_common.copy(), feature_subset)
    sol_close = df_common["SOL_close"].reindex(X_df.index)

    # ── Align to yesterday 23:00 UTC so forecast starts at today 00:00 UTC ──
    cutoff = config.get("_cutoff", _yesterday_end())
    if X_df.index[-1] > cutoff:
        X_df      = X_df.loc[X_df.index <= cutoff]
        sol_close = sol_close.loc[sol_close.index <= cutoff]
        logger.info("Data capped at %s UTC; forecast covers today 00:00–23:00 UTC", cutoff)
    elif X_df.index[-1] < cutoff - pd.Timedelta(hours=23):
        logger.warning(
            "Data ends at %s — stale. Run 'python main.py collect' for today's forecast.",
            X_df.index[-1].date(),
        )

    # ── Base models ───────────────────────────────────────────────────────────
    stale_base  = _cache_is_stale(config, _XGB_MODEL_FILENAME)
    cached_base = None if (force or stale_base) else _load_models(config, _XGB_MODEL_FILENAME)
    if stale_base and not force:
        logger.info("XGB base cache stale (parquet refreshed) — retraining")
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
    in_data_ts, in_data_pred, in_data_actual, in_data_rmse, in_data_adj_r2 = _in_data_predict(
        base_model, X_df, sol_close, window=_INDATA_HOURS
    )
    logger.info(
        "In-data RMSE (last %d h): $%.2f  adj-R²=%.4f",
        _INDATA_HOURS, in_data_rmse, in_data_adj_r2,
    )

    # ── Recursive forecast — full 168 steps, display = today 00:00–23:00 UTC ─
    future_ts_full, xgb_exp_full, xgb_lo_full, xgb_hi_full = _recursive_forecast(
        base_model, q10_model, q90_model, X_df, sol_close
    )
    _today_midnight = config.get("_today", pd.Timestamp.now(tz="UTC").normalize())
    _today_end      = _today_midnight + pd.Timedelta(hours=23)
    _day_mask = (future_ts_full >= _today_midnight) & (future_ts_full <= _today_end)
    if _day_mask.any():
        future_ts = future_ts_full[_day_mask]
        xgb_exp   = xgb_exp_full[_day_mask]
        xgb_lo    = xgb_lo_full[_day_mask]
        xgb_hi    = xgb_hi_full[_day_mask]
    else:
        future_ts = future_ts_full[:_DISPLAY_HOURS]
        xgb_exp   = xgb_exp_full[:_DISPLAY_HOURS]
        xgb_lo    = xgb_lo_full[:_DISPLAY_HOURS]
        xgb_hi    = xgb_hi_full[:_DISPLAY_HOURS]
    logger.info(
        "XGB forecast: last=$%.2f  E[today 23:00]=$%.2f  CI=[$%.2f, $%.2f]",
        float(sol_close.iloc[-1]), xgb_exp[-1], xgb_lo[-1], xgb_hi[-1],
    )

    # ── XGBoost+ ──────────────────────────────────────────────────────────────
    viable_all = _viable_optional_features(df_common)

    stale_plus  = _cache_is_stale(config, _XGB_PLUS_MODEL_FILENAME)
    cached_plus = None if (force or stale_plus) else _load_models(config, _XGB_PLUS_MODEL_FILENAME)
    if stale_plus and not force:
        logger.info("XGB+ cache stale — retraining")
    if cached_plus is not None:
        plus_model, plus_q10, plus_q90, plus_features = cached_plus
        logger.info("XGB+ models loaded from cache (features: %s)", plus_features)
    else:
        df_common_capped = df_common.loc[df_common.index <= cutoff]
        plus_features = _find_plus_features(
            df_common_capped, sol_close, feature_subset, viable_all
        )
        if plus_features:
            plus_subset = feature_subset + plus_features
            X_plus      = build_feature_matrix(df_common_capped.copy(), plus_subset)
            sol_plus    = df_common_capped["SOL_close"].reindex(X_plus.index)
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

    xgb_plus_exp = xgb_plus_in_pred = xgb_plus_rmse = xgb_plus_adj_r2 = None
    xgb_plus_in_ts = None
    if plus_model is not None:
        plus_subset = feature_subset + plus_features
        X_plus      = build_feature_matrix(df_common.copy(), plus_subset)
        sol_plus    = df_common["SOL_close"].reindex(X_plus.index)
        if X_plus.index[-1] > cutoff:
            X_plus   = X_plus.loc[X_plus.index <= cutoff]
            sol_plus = sol_plus.loc[sol_plus.index <= cutoff]

        _fts, _exp, _, _ = _recursive_forecast(
            plus_model, plus_q10, plus_q90, X_plus, sol_plus
        )
        _plus_mask   = (_fts >= _today_midnight) & (_fts <= _today_end)
        xgb_plus_exp = _exp[_plus_mask] if _plus_mask.any() else _exp[:_DISPLAY_HOURS]
        xgb_plus_in_ts, xgb_plus_in_pred, _, xgb_plus_rmse, xgb_plus_adj_r2 = _in_data_predict(
            plus_model, X_plus, sol_plus, window=_INDATA_HOURS
        )
        logger.info(
            "XGB+ E[+24h]=$%.2f  in-data RMSE=$%.2f  adj-R²=%.4f  added: %s",
            xgb_plus_exp[-1], xgb_plus_rmse, xgb_plus_adj_r2, plus_features,
        )

    return {
        "in_data_ts":        in_data_ts,
        "in_data_actual":    in_data_actual,
        "in_data_pred":      in_data_pred,
        "in_data_rmse":      in_data_rmse,
        "in_data_adj_r2":    in_data_adj_r2,
        "future_ts":         future_ts,
        "xgb_exp":           xgb_exp,
        "xgb_lo":            xgb_lo,
        "xgb_hi":            xgb_hi,
        "xgb_plus_exp":      xgb_plus_exp,
        "xgb_plus_in_pred":  xgb_plus_in_pred,
        "xgb_plus_in_ts":    xgb_plus_in_ts,
        "xgb_plus_rmse":     xgb_plus_rmse,
        "xgb_plus_adj_r2":   xgb_plus_adj_r2,
        "plus_features":     plus_features,
        "feature_names":     feature_subset,
        "sol_last":          float(sol_close.iloc[-1]),
    }
