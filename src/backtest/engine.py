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
  Mild look-ahead bias (model parameters trained on all data) — documented in report.
  Per-fold regime labels for Option A metadata use causal inference (data up to fold cutoff).
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


def _parse_variant(vcfg: dict) -> dict:
    """Parse a single strategy_variants entry into kwargs for RegimeStrategy.apply()."""
    _dt = vcfg.get("discrete_trading")
    _tw = vcfg.get("trading_window")
    _th = vcfg.get("trading_hours")
    stop = vcfg.get("trailing_stop_pct")
    return {
        "discrete_trading": (int(_dt[0]), int(_dt[1])) if _dt else None,
        "trading_window": (int(_tw[0]), int(_tw[1])) if _tw else None,
        "trading_hours": (int(_th[0]), int(_th[1])) if _th else None,
        "trailing_stop_pct": float(stop) if stop else None,
        "long_only": bool(vcfg.get("long_only", False)),
        "xgb_gated": bool(vcfg.get("xgb_gated", False)),
        "allowed_hours": [int(h) for h in vcfg["allowed_hours"]]
        if vcfg.get("allowed_hours")
        else None,
    }


def run(config: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run walk-forward backtest (A) and regime strategies (B).

    Returns
    -------
    fold_df    : one row per (fold × forecast hour)
                 columns: fold_id, horizon_h, actual, xgb_pred, regime
    strategies : dict[variant_name, strategy_df]; one row per hour over full
                 history per variant; columns: regime, position, off_hours,
                 stopped, strategy_lr, bnh_lr, equity_strategy, equity_bnh
    """
    bt_cfg = config.get("backtest", {})
    min_train_h = int(bt_cfg.get("min_train_days", 30)) * 24
    step_h = int(bt_cfg.get("step_days", 7)) * 24
    horizon_h = int(bt_cfg.get("horizon_hours", 24))

    # ── Data & model ──────────────────────────────────────────────────────────
    best = load_best_features(config)
    if best is None:
        raise FileNotFoundError(
            "best_features.json missing — run 'python main.py hmm' first"
        )

    # HMM uses the full feature subset; XGB uses the ≤24h-filtered subset.
    # Using the filtered subset for HMM.predict() causes a shape mismatch because
    # the saved model was trained on all features.
    hmm_features: list[str] = best["feature_subset"]
    xgb_features: list[str] = _filter_24h_features(best["feature_subset"])

    df_common = load_common_dataframe(config)
    X_hmm = build_feature_matrix(df_common.copy(), hmm_features)
    X_df = build_feature_matrix(df_common.copy(), xgb_features)
    sol_close = df_common["SOL_close"].reindex(X_df.index)

    cutoff = config.get(
        "_cutoff", pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(hours=1)
    )
    X_hmm = X_hmm.loc[X_hmm.index <= cutoff]
    X_df = X_df.loc[X_df.index <= cutoff]
    sol_close = sol_close.loc[sol_close.index <= cutoff]

    mpath = models_dir(config) / f"best_hmm_k{best['n_components']}.pkl"
    model = GaussianHMMModel.load(mpath)
    logger.info("HMM model loaded ← %s", mpath)

    # ── HMM regime labels — full history (Option B) ───────────────────────────
    # NOTE: mild look-ahead bias here — model parameters were fitted on all data.
    # Option B strategy uses these full-history labels for position assignment.
    labels = model.predict(X_hmm.values)
    regime_info = _assign_regime_colors_and_labels(model, X_hmm, labels)
    label_series = pd.Series(
        [regime_info[int(lbl)]["label"] for lbl in labels],
        index=X_hmm.index,
        name="regime",
    )
    logger.warning(
        "Option B uses full-history HMM labels (mild look-ahead bias — "
        "model parameters trained on all data). Option A folds use causal labels."
    )

    # ── Causal regime labels for Option A fold metadata ───────────────────────
    # For each fold cutoff t, predict regime using only data up to t.
    # Avoids forward-backward look-ahead in per-fold accuracy reporting.
    # fold_idxs is a range() with step_h > 0, so t values are always distinct;
    # the cache guard `if t not in causal_label_at` is a safety net for
    # hypothetical step_h=0 configs, not a real collision risk.
    causal_label_at: dict[int, str] = {}

    # ── Option B prep ─────────────────────────────────────────────────────────
    sol_lr = np.log(sol_close / sol_close.shift(1)).dropna()

    raw_variants = bt_cfg.get("strategy_variants")
    if raw_variants:
        variant_cfgs: dict[str, dict] = raw_variants
    else:
        variant_cfgs = {
            "default": {
                "discrete_trading": bt_cfg.get("discrete_trading"),
                "trading_window": bt_cfg.get("trading_window"),
                "trading_hours": bt_cfg.get("trading_hours"),
                "trailing_stop_pct": bt_cfg.get("trailing_stop_pct"),
                "long_only": False,
            }
        }

    needs_gate = any(vcfg.get("xgb_gated", False) for vcfg in variant_cfgs.values())

    # ── Option A: XGB walk-forward folds ──────────────────────────────────────
    N = len(X_df)
    fold_idxs = range(min_train_h, N - horizon_h, step_h)
    n_folds = len(fold_idxs)
    logger.info(
        "Walk-forward: %d folds  min_train=%dd  step=%dd  horizon=%dh",
        n_folds,
        min_train_h // 24,
        step_h // 24,
        horizon_h,
    )

    # Pre-compute for Option C: 24-step HMM transition matrix + integer state series
    state_series = pd.Series(labels, index=X_hmm.index)
    trans_24 = np.linalg.matrix_power(model._model.transmat_, 24)

    xgb_direction_at: dict[pd.Timestamp, float] = {}
    persistence_at: dict[pd.Timestamp, float] = {}

    fold_records: list[dict] = []

    for fold_i, t in enumerate(fold_idxs):
        # Causal regime label: Viterbi on data up to fold cutoff only.
        # Sliced by timestamp (not by integer index) so that X_hmm and X_df
        # alignment is preserved even when X_hmm starts later than X_df
        # (e.g. 168h features add warmup-row offset).
        if t not in causal_label_at:
            ts_cutoff = X_df.index[t]
            X_hmm_causal = X_hmm.loc[X_hmm.index <= ts_cutoff]
            if X_hmm_causal.empty:
                causal_label_at[t] = "Neutral"
            else:
                causal_labels_t = model.predict(X_hmm_causal.values)
                causal_label_at[t] = regime_info[int(causal_labels_t[-1])]["label"]

        X_train = X_df.iloc[:t]
        s_train = sol_close.iloc[:t]
        X_tr, y_tr = _build_train_data(X_train, s_train)
        if len(X_tr) < 100:
            continue

        base_model = _train_model(X_tr, y_tr)

        # Oracle prediction: use real future features (no recursive error accumulation)
        X_test = X_df.iloc[t : t + horizon_h]
        s_test = sol_close.iloc[t + 1 : t + horizon_h + 1]
        pred_lrs = base_model.predict(X_test.values.astype(np.float32))
        start_p = float(sol_close.iloc[t])
        pred_prices = start_p * np.exp(np.cumsum(pred_lrs))

        ts_at_t = X_df.index[t]
        regime_at_t = causal_label_at[t]

        # ── Option C signals (collected per fold, free — model already trained) ──
        if needs_gate:
            xgb_direction_at[ts_at_t] = 1.0 if pred_prices[-1] > start_p else -1.0
            state_int_raw = state_series.get(ts_at_t)
            if state_int_raw is None:
                logger.debug(
                    "ts_at_t %s not in state_series (X_hmm/X_df index gap) — "
                    "using last known state for persistence",
                    ts_at_t,
                )
                state_int = int(state_series.iloc[-1])
            else:
                state_int = int(state_int_raw)
            persistence_at[ts_at_t] = float(trans_24[state_int, state_int])

        for h, (pred_p, ts) in enumerate(zip(pred_prices, s_test.index)):
            act_p = float(s_test.iloc[h]) if h < len(s_test) else float("nan")
            fold_records.append(
                {
                    "fold_id": fold_i,
                    "timestamp": ts,
                    "horizon_h": h + 1,
                    "actual": act_p,
                    "xgb_pred": pred_p,
                    "start_price": start_p,
                    "regime": regime_at_t,
                }
            )

        if (fold_i + 1) % 10 == 0 or fold_i == n_folds - 1:
            fold_rmse = rmse(s_test.values[: len(pred_prices)], pred_prices)
            logger.info(
                "  fold %3d/%d  cutoff=%s  regime=%-15s  RMSE=$%.2f",
                fold_i + 1,
                n_folds,
                X_df.index[t].strftime("%Y-%m-%d"),
                regime_at_t,
                fold_rmse,
            )

    if not fold_records:
        logger.warning("No fold records generated — returning empty DataFrame")
        return pd.DataFrame()

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

    # ── Build Option C signal series (forward-filled to hourly) ──────────────
    if needs_gate and xgb_direction_at:
        xgb_dir_series: pd.Series | None = (
            pd.Series(xgb_direction_at)
            .sort_index()
            .reindex(label_series.index, method="ffill")
        )
        pers_series: pd.Series | None = (
            pd.Series(persistence_at)
            .sort_index()
            .reindex(label_series.index, method="ffill")
        )
        logger.info(
            "Option C signals built: xgb_direction coverage=%.1f%%  persistence mean=%.3f",
            xgb_dir_series.notna().mean() * 100,
            pers_series.dropna().mean(),
        )
    else:
        xgb_dir_series = None
        pers_series = None

    # ── Option B: Regime strategies (one per variant) ─────────────────────────
    strategies: dict[str, pd.DataFrame] = {}
    for name, vcfg in variant_cfgs.items():
        kwargs = _parse_variant(vcfg)
        is_gated = kwargs.pop("xgb_gated")
        if is_gated and xgb_dir_series is not None:
            kwargs["xgb_signal"] = xgb_dir_series
            kwargs["persistence"] = pers_series
        sdf = RegimeStrategy().apply(sol_lr, label_series, **kwargs)
        n_active = int((sdf["position"] != 0).sum())
        n_stopped = int(sdf["stopped"].sum()) if sdf["stopped"].any() else 0
        logger.info(
            "Variant %-20s  equity=%.4f  active=%d h  stopped=%d h",
            name,
            sdf["equity_strategy"].iloc[-1],
            n_active,
            n_stopped,
        )
        strategies[name] = sdf

    first_sdf = next(iter(strategies.values()))
    logger.info(
        "Regime strategy: %d hours  variants=%d  primary_equity=%.4f",
        len(first_sdf),
        len(strategies),
        first_sdf["equity_strategy"].iloc[-1],
    )

    return fold_df, strategies
