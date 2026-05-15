"""Unit tests for predict_xgb.py — no real Parquet files needed."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.hmm.predict_xgb import (
    _build_train_data,
    _in_data_predict,
    _init_state,
    _recursive_forecast,
    _train_model,
    _update_row,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_X_df(n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic feature matrix and SOL close price series."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    idx.name = "timestamp"

    lr = np.concatenate([
        rng.normal(0.0, 0.001, n // 2),
        rng.normal(0.0, 0.05,  n // 2),
    ])
    price = 100.0 * np.exp(np.cumsum(lr))

    X_df = pd.DataFrame(index=idx)
    X_df["SOL_log_return"] = lr
    X_df["SOL_vol_24h"]    = pd.Series(lr, index=idx).rolling(24).std().values
    X_df["SOL_vol_168h"]   = pd.Series(lr, index=idx).rolling(168).std().values
    X_df["BTC_log_return"]        = rng.normal(0, 0.002, n)
    X_df["BTC_log_return_lag_6h"] = np.roll(X_df["BTC_log_return"].values, 6)
    X_df = X_df.dropna()

    sol_close = pd.Series(price[:len(X_df)], index=X_df.index, name="SOL_close")
    return X_df, sol_close


# ─────────────────────────────────────────────────────────────────────────────
# _build_train_data
# ─────────────────────────────────────────────────────────────────────────────


def test_build_train_data_shape() -> None:
    X_df, sol_close = _make_X_df(300)
    X, y = _build_train_data(X_df, sol_close)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == len(X_df) - 1
    assert X.shape[1] == X_df.shape[1]


def test_build_train_data_dtype() -> None:
    X_df, sol_close = _make_X_df(200)
    X, y = _build_train_data(X_df, sol_close)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# _train_model
# ─────────────────────────────────────────────────────────────────────────────


def test_train_model_base() -> None:
    X_df, sol_close = _make_X_df(300)
    X, y = _build_train_data(X_df, sol_close)
    model = _train_model(X, y, n_estimators=20)
    preds = model.predict(X[:10])
    assert len(preds) == 10
    assert np.all(np.isfinite(preds))


def test_train_model_quantile() -> None:
    X_df, sol_close = _make_X_df(300)
    X, y = _build_train_data(X_df, sol_close)
    q10 = _train_model(X, y, quantile=0.10, n_estimators=20)
    q90 = _train_model(X, y, quantile=0.90, n_estimators=20)
    p10 = q10.predict(X[:50])
    p90 = q90.predict(X[:50])
    # q90 predictions should be ≥ q10 predictions on average
    assert np.mean(p90) >= np.mean(p10)


# ─────────────────────────────────────────────────────────────────────────────
# _init_state and _update_row
# ─────────────────────────────────────────────────────────────────────────────


def test_init_state_buffer_length() -> None:
    X_df, _ = _make_X_df(300)
    state = _init_state(X_df)
    assert len(state["sol_buffer"]) <= 168
    assert len(state["btc_history"]) == len(X_df)


def test_update_row_sol_fields_updated() -> None:
    X_df, _ = _make_X_df(300)
    state = _init_state(X_df)
    feature_cols = list(X_df.columns)
    row = X_df.iloc[-1].to_dict()
    new_row = _update_row(row, 0.01, state, step=0, feature_cols=feature_cols)
    assert new_row["SOL_log_return"] == pytest.approx(0.01)
    assert "SOL_vol_24h" in new_row


def test_update_row_btc_lag_history_used() -> None:
    X_df, _ = _make_X_df(300)
    state = _init_state(X_df)
    feature_cols = list(X_df.columns)
    row = X_df.iloc[-1].to_dict()
    # step=0, lag=6: hist_idx = n - 6 + 0 = n-6 (within history) → real value
    new_row = _update_row(row, 0.0, state, step=0, feature_cols=feature_cols)
    btc_hist = state["btc_history"]
    expected = float(btc_hist[len(btc_hist) - 6 + 0])
    assert new_row["BTC_log_return_lag_6h"] == pytest.approx(expected)


def test_update_row_btc_lag_persists_last_for_future() -> None:
    X_df, _ = _make_X_df(300)
    state = _init_state(X_df)
    feature_cols = list(X_df.columns)
    row = X_df.iloc[-1].to_dict()
    # step=10, lag=6: hist_idx = n - 6 + 10 = n + 4 ≥ n → future → forward-fill last known
    new_row = _update_row(row, 0.0, state, step=10, feature_cols=feature_cols)
    expected = float(state["btc_history"][-1])
    assert new_row["BTC_log_return_lag_6h"] == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────────────────
# _in_data_predict
# ─────────────────────────────────────────────────────────────────────────────


def test_in_data_predict_returns_correct_shape() -> None:
    X_df, sol_close = _make_X_df(400)
    X, y = _build_train_data(X_df, sol_close)
    model = _train_model(X, y, n_estimators=20)
    ts, pred, actual, rmse, ar2 = _in_data_predict(model, X_df, sol_close, window=48)
    assert len(ts) == 48
    assert len(pred) == 48
    assert len(actual) == 48
    assert np.isfinite(rmse)
    assert rmse > 0
    assert np.isfinite(ar2)


def test_in_data_predict_prices_positive() -> None:
    X_df, sol_close = _make_X_df(400)
    X, y = _build_train_data(X_df, sol_close)
    model = _train_model(X, y, n_estimators=20)
    _, pred, _, _, _ = _in_data_predict(model, X_df, sol_close, window=48)
    assert np.all(pred > 0)


# ─────────────────────────────────────────────────────────────────────────────
# _recursive_forecast
# ─────────────────────────────────────────────────────────────────────────────


def test_recursive_forecast_shape() -> None:
    X_df, sol_close = _make_X_df(400)
    X, y = _build_train_data(X_df, sol_close)
    base = _train_model(X, y, n_estimators=20)
    q10  = _train_model(X, y, quantile=0.10, n_estimators=20)
    q90  = _train_model(X, y, quantile=0.90, n_estimators=20)
    future_ts, exp, lo, hi = _recursive_forecast(base, q10, q90, X_df, sol_close, n_steps=24)
    assert len(future_ts) == 24
    assert len(exp) == 24
    assert len(lo) == 24
    assert len(hi) == 24


def test_recursive_forecast_prices_positive() -> None:
    X_df, sol_close = _make_X_df(400)
    X, y = _build_train_data(X_df, sol_close)
    base = _train_model(X, y, n_estimators=20)
    q10  = _train_model(X, y, quantile=0.10, n_estimators=20)
    q90  = _train_model(X, y, quantile=0.90, n_estimators=20)
    _, exp, lo, hi = _recursive_forecast(base, q10, q90, X_df, sol_close, n_steps=24)
    assert np.all(exp > 0)
    assert np.all(lo > 0)
    assert np.all(hi > 0)


def test_recursive_forecast_starts_at_last_close() -> None:
    X_df, sol_close = _make_X_df(400)
    X, y = _build_train_data(X_df, sol_close)
    base = _train_model(X, y, n_estimators=20)
    q10  = _train_model(X, y, quantile=0.10, n_estimators=20)
    q90  = _train_model(X, y, quantile=0.90, n_estimators=20)
    _, exp, _, _ = _recursive_forecast(base, q10, q90, X_df, sol_close, n_steps=1)
    # First forecast price should be close to last known price (not wildly off)
    assert abs(exp[0] / float(sol_close.iloc[-1]) - 1.0) < 0.5


def test_recursive_forecast_timestamps_after_last_known() -> None:
    X_df, sol_close = _make_X_df(400)
    X, y = _build_train_data(X_df, sol_close)
    base = _train_model(X, y, n_estimators=20)
    q10  = _train_model(X, y, quantile=0.10, n_estimators=20)
    q90  = _train_model(X, y, quantile=0.90, n_estimators=20)
    future_ts, _, _, _ = _recursive_forecast(base, q10, q90, X_df, sol_close, n_steps=5)
    assert future_ts[0] == X_df.index[-1] + pd.Timedelta(hours=1)
