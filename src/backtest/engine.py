"""Walk-forward backtest engine.

Design
------
Template Method: run() is the fixed pipeline skeleton.

Option A — XGB Walk-Forward Forecast Accuracy:
  For each fold at cutoff t (step = step_days × 24h):
    - Train XGB on X_df[0:t], sol_close[0:t]   (no cache — different window each fold)
    - Predict next horizon_hours using REAL features from X_df[t:t+H]  (oracle, no recursion)
    - Record RMSE, MAE, directional accuracy, horizon, regime label at t

  Oracle evaluation gives the upper bound on XGB accuracy; the recursive
  forecast used in production degrades further from error accumulation.

Option B — HMM Regime Strategy:
  Uses the pre-trained HMM model to assign regime labels to the full history.
  Mild look-ahead bias (model trained on all data) — documented in report.
  Position map: Strong Bullish=+1 … Strong Bearish=−1.
  Hourly P&L = position × actual SOL log-return.

NeuralProphet excluded: ~55 s/fold makes a 200-fold backtest impractical (~3 h).
The NP+ shape bug (2026-05-04) is fixed in predict_prophet._build_np_df and
_train_model; performance is the only remaining reason for exclusion.
"""

import logging

import numpy as np
import pandas as pd

from src.hmm.features import build_feature_matrix, load_common_dataframe
from src.hmm.model import GaussianHMMModel
from src.hmm.optimizer import load_best_features
from src.hmm.predict_xgb import (
    _build_train_data,
    _filter_24h_features,
    _train_model,
)
from src.hmm.visualize import _assign_regime_colors_and_labels
from src.utils.paths import models_dir

from .metrics import rmse
from .strategy import RegimeStrategy

logger = logging.getLogger(__name__)


def run(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward backtest (A) and regime strategy (B).

    Returns
    -------
    fold_df     : one row per (fold × forecast hour)
                  columns: fold_id, horizon_h, actual, xgb_pred, regime
    strategy_df : one row per hour over full history
                  columns: regime, position, strategy_lr, bnh_lr,
                           equity_strategy, equity_bnh
    """
    bt_cfg      = config.get("backtest", {})
    min_train_h = int(bt_cfg.get("min_train_days", 30)) * 24
    step_h      = int(bt_cfg.get("step_days",      7))  * 24
    horizon_h   = int(bt_cfg.get("horizon_hours",  24))
    trailing_stop_pct: float | None = bt_cfg.get("trailing_stop_pct") or None
    if trailing_stop_pct is not None:
        trailing_stop_pct = float(trailing_stop_pct)

    _th_raw = bt_cfg.get("trading_hours")
    trading_hours: tuple[int, int] | None = (
        (int(_th_raw[0]), int(_th_raw[1])) if _th_raw else None
    )

    # ── Data & model ──────────────────────────────────────────────────────────
    best = load_best_features(config)
    if best is None:
        raise FileNotFoundError("best_features.json missing — run 'python main.py hmm' first")

    # HMM uses the full feature subset; XGB uses the ≤24h-filtered subset.
    # Using the filtered subset for HMM.predict() causes a shape mismatch because
    # the saved model was trained on all features.
    hmm_features: list[str] = best["feature_subset"]
    xgb_features: list[str] = _filter_24h_features(best["feature_subset"])

    df_common  = load_common_dataframe(config)
    X_hmm      = build_feature_matrix(df_common.copy(), hmm_features)
    X_df       = build_feature_matrix(df_common.copy(), xgb_features)
    sol_close  = df_common["SOL_close"].reindex(X_df.index)

    cutoff = config.get("_cutoff", pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(hours=1))
    X_hmm     = X_hmm.loc[X_hmm.index <= cutoff]
    X_df      = X_df.loc[X_df.index <= cutoff]
    sol_close = sol_close.loc[sol_close.index <= cutoff]

    mpath = models_dir(config) / f"best_hmm_k{best['n_components']}.pkl"
    model = GaussianHMMModel.load(mpath)
    logger.info("HMM model loaded ← %s", mpath)

    # ── HMM regime labels — full history (Option B) ───────────────────────────
    labels      = model.predict(X_hmm.values)
    regime_info = _assign_regime_colors_and_labels(model, X_hmm, labels)
    label_series = pd.Series(
        [regime_info[int(lbl)]["label"] for lbl in labels],
        index=X_hmm.index,
        name="regime",
    )

    # ── Option B: Regime strategy ─────────────────────────────────────────────
    sol_lr = np.log(sol_close / sol_close.shift(1)).dropna()
    strategy_df = RegimeStrategy().apply(
        sol_lr, label_series,
        trailing_stop_pct=trailing_stop_pct,
        trading_hours=trading_hours,
    )
    if trading_hours is not None:
        n_off = int(strategy_df["off_hours"].sum())
        logger.info(
            "Trading hours %02d:00–%02d:00 UTC: %d hours filtered out (%.1f %%)",
            trading_hours[0], trading_hours[1],
            n_off, 100 * n_off / max(len(strategy_df), 1),
        )
    if trailing_stop_pct is not None:
        n_stopped = int(strategy_df["stopped"].sum())
        logger.info(
            "Trailing stop %.0f %%: %d hours stopped out (%.1f %% of all hours)",
            trailing_stop_pct, n_stopped, 100 * n_stopped / max(len(strategy_df), 1),
        )
    logger.info(
        "Regime strategy: %d hours  equity_strategy=%.4f  equity_bnh=%.4f",
        len(strategy_df),
        strategy_df["equity_strategy"].iloc[-1],
        strategy_df["equity_bnh"].iloc[-1],
    )

    # ── Option A: XGB walk-forward folds ──────────────────────────────────────
    N          = len(X_df)
    fold_idxs  = range(min_train_h, N - horizon_h, step_h)
    n_folds    = len(fold_idxs)
    logger.info(
        "Walk-forward: %d folds  min_train=%dd  step=%dd  horizon=%dh",
        n_folds, min_train_h // 24, step_h // 24, horizon_h,
    )

    fold_records: list[dict] = []

    for fold_i, t in enumerate(fold_idxs):
        X_train = X_df.iloc[:t]
        s_train = sol_close.iloc[:t]
        X_tr, y_tr = _build_train_data(X_train, s_train)
        if len(X_tr) < 100:
            continue

        base_model = _train_model(X_tr, y_tr)

        # Oracle prediction: use real future features (no recursive error accumulation)
        X_test   = X_df.iloc[t : t + horizon_h]
        s_test   = sol_close.iloc[t + 1 : t + horizon_h + 1]
        pred_lrs = base_model.predict(X_test.values.astype(np.float32))
        start_p  = float(sol_close.iloc[t])
        pred_prices = start_p * np.exp(np.cumsum(pred_lrs))

        # Timestamp-based lookup: X_df and X_hmm may have different row counts when
        # _filter_24h_features removes 168h features, causing X_df to start earlier.
        ts_at_t     = X_df.index[t]
        regime_at_t = label_series.get(ts_at_t, label_series.iloc[min(t, len(label_series) - 1)])

        for h, (pred_p, ts) in enumerate(zip(pred_prices, s_test.index)):
            act_p = float(s_test.iloc[h]) if h < len(s_test) else float("nan")
            fold_records.append(
                {
                    "fold_id":   fold_i,
                    "timestamp": ts,
                    "horizon_h": h + 1,
                    "actual":    act_p,
                    "xgb_pred":  pred_p,
                    "regime":    regime_at_t,
                }
            )

        if (fold_i + 1) % 10 == 0 or fold_i == n_folds - 1:
            fold_rmse = rmse(
                s_test.values[: len(pred_prices)],
                pred_prices,
            )
            logger.info(
                "  fold %3d/%d  cutoff=%s  regime=%-15s  RMSE=$%.2f",
                fold_i + 1, n_folds,
                X_df.index[t].strftime("%Y-%m-%d"),
                regime_at_t, fold_rmse,
            )

    fold_df = pd.DataFrame(fold_records)
    fold_df["timestamp"] = pd.DatetimeIndex(fold_df["timestamp"]).tz_convert("UTC")
    fold_df = fold_df.set_index("timestamp")

    logger.info(
        "Walk-forward complete: %d folds  %d rows  date range %s → %s",
        fold_df["fold_id"].nunique(),
        len(fold_df),
        fold_df.index.min().strftime("%Y-%m-%d"),
        fold_df.index.max().strftime("%Y-%m-%d"),
    )
    return fold_df, strategy_df
