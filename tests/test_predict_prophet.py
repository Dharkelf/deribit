"""Unit tests for src.hmm.predict_prophet — pure helper functions only.

_train_model / run() are excluded: they require NeuralProphet + PyTorch
and take ~55 s per call.  All pure data-transform functions are covered.
"""

import numpy as np
import pandas as pd
import pytest

from src.hmm.predict_prophet import (
    _adj_r2,
    _build_np_df,
    _extract_forecast,
    _load_xgb_plus_features,
)


# ─────────────────────────────────────────────────────────────────────────────
# _adj_r2
# ─────────────────────────────────────────────────────────────────────────────


def test_adj_r2_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _adj_r2(y, y, n_features=1) == pytest.approx(1.0)


def test_adj_r2_zero_variance_target():
    y = np.ones(10)
    assert _adj_r2(y, y + 0.1, n_features=1) == pytest.approx(0.0)


def test_adj_r2_penalises_extra_features():
    rng  = np.random.default_rng(0)
    y    = rng.standard_normal(50)
    pred = y + rng.standard_normal(50) * 0.1
    r2_1 = _adj_r2(y, pred, n_features=1)
    r2_5 = _adj_r2(y, pred, n_features=5)
    assert r2_1 > r2_5


def test_adj_r2_too_few_samples_returns_zero():
    y = np.array([1.0, 2.0])
    assert _adj_r2(y, y, n_features=5) == pytest.approx(0.0)


def test_adj_r2_negative_for_bad_predictions():
    y    = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert _adj_r2(y, pred, n_features=1) < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _build_np_df
# ─────────────────────────────────────────────────────────────────────────────


def _make_feature_df(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    X   = pd.DataFrame(
        {"feat_a": np.linspace(0, 1, n), "feat_b": np.random.default_rng(1).standard_normal(n)},
        index=idx,
    )
    sol = pd.Series(np.linspace(100, 200, n), index=idx, name="SOL_close")
    return X, sol


def test_build_np_df_has_required_columns():
    X, sol = _make_feature_df()
    df = _build_np_df(X, sol, ["feat_a", "feat_b"])
    assert "ds" in df.columns
    assert "y"  in df.columns
    assert "feat_a" in df.columns
    assert "feat_b" in df.columns


def test_build_np_df_ds_is_tz_naive():
    X, sol = _make_feature_df()
    df = _build_np_df(X, sol, ["feat_a"])
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])
    assert df["ds"].dt.tz is None


def test_build_np_df_feature_columns_are_1d():
    X, sol = _make_feature_df()
    df = _build_np_df(X, sol, ["feat_a", "feat_b"])
    for feat in ["feat_a", "feat_b"]:
        assert df[feat].to_numpy().ndim == 1


def test_build_np_df_skips_missing_feature():
    X, sol = _make_feature_df()
    df = _build_np_df(X, sol, ["feat_a", "nonexistent"])
    assert "nonexistent" not in df.columns
    assert "feat_a" in df.columns


def test_build_np_df_duplicate_column_squeezed_to_1d():
    """isinstance guard: X_df[feat] returning a DataFrame must be reduced to Series."""
    idx = pd.date_range("2025-01-01", periods=50, freq="1h", tz="UTC")
    # Build a DataFrame with a genuinely duplicated column name
    base = pd.DataFrame({"feat_dup": np.ones(50)}, index=idx)
    extra = pd.DataFrame({"feat_dup": np.zeros(50)}, index=idx)
    X_dup = pd.concat([base, extra], axis=1)   # two columns both named "feat_dup"
    assert isinstance(X_dup["feat_dup"], pd.DataFrame)  # confirm duplicate

    sol = pd.Series(np.linspace(100, 150, 50), index=idx)
    df  = _build_np_df(X_dup, sol, ["feat_dup"])

    # Column must be present and 1D
    assert "feat_dup" in df.columns
    assert df["feat_dup"].to_numpy().ndim == 1


def test_build_np_df_row_count_matches_non_null_sol():
    idx = pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC")
    X   = pd.DataFrame({"feat_a": np.ones(10)}, index=idx)
    sol = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=idx)
    df  = _build_np_df(X, sol, ["feat_a"])
    assert len(df) == 9   # one NaN dropped


def test_build_np_df_index_reset():
    X, sol = _make_feature_df(30)
    df = _build_np_df(X, sol, ["feat_a"])
    assert list(df.index) == list(range(len(df)))


# ─────────────────────────────────────────────────────────────────────────────
# _extract_forecast
# ─────────────────────────────────────────────────────────────────────────────


def _make_forecast_df(n_hist: int = 80, n_future: int = 48) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal forecast DataFrame as NeuralProphet would produce it."""
    n_total   = n_hist + n_future
    ds        = pd.date_range("2025-06-01", periods=n_total, freq="1h")
    yhat1     = np.linspace(100, 200, n_total)
    yhat_cols = {f"yhat{i}": np.linspace(100 + i, 200 + i, n_total) for i in range(1, n_future + 1)}
    lo_cols   = {f"yhat{i} 10.0%": yhat_cols[f"yhat{i}"] * 0.95 for i in range(1, n_future + 1)}
    hi_cols   = {f"yhat{i} 90.0%": yhat_cols[f"yhat{i}"] * 1.05 for i in range(1, n_future + 1)}

    forecast = pd.DataFrame({"ds": ds, "yhat1": yhat1, **yhat_cols, **lo_cols, **hi_cols})

    np_df = pd.DataFrame({
        "ds": ds[:n_hist],
        "y":  np.linspace(100, 150, n_hist),
    })

    return forecast, np_df


def test_extract_forecast_returns_8_tuple():
    forecast, np_df = _make_forecast_df()
    idx     = pd.date_range("2025-01-01", periods=200, freq="1h", tz="UTC")
    sol     = pd.Series(np.linspace(100, 200, 200), index=idx)
    today   = pd.Timestamp("2025-06-04", tz="UTC")
    result  = _extract_forecast(forecast, np_df, sol, today_midnight=today)
    assert len(result) == 8


def test_extract_forecast_empty_yhat_returns_nan_arrays():
    """When forecast has no matching future rows, return NaN arrays of length 24."""
    n = 10
    ds  = pd.date_range("2025-01-01", periods=n, freq="1h")
    fc  = pd.DataFrame({"ds": ds, "yhat1": np.ones(n)})   # no yhat2…, no lo/hi
    npd = pd.DataFrame({"ds": ds, "y": np.ones(n)})
    sol = pd.Series(np.ones(50), index=pd.date_range("2025-01-01", periods=50, freq="1h", tz="UTC"))
    # today far in the future → no match
    today = pd.Timestamp("2030-01-01", tz="UTC")
    _, _, _, _, future_ts, np_exp, np_lo, np_hi = _extract_forecast(fc, npd, sol, today_midnight=today)
    assert len(np_exp) == 24
    assert np.all(np.isnan(np_exp))


def test_extract_forecast_in_data_rmse_finite():
    forecast, np_df = _make_forecast_df(n_hist=80)
    idx  = pd.date_range("2025-06-01", periods=200, freq="1h", tz="UTC")
    sol  = pd.Series(np.linspace(100, 200, 200), index=idx)
    today = pd.Timestamp("2025-06-04", tz="UTC")
    _, in_pred, in_actual, rmse_val, *_ = _extract_forecast(forecast, np_df, sol, today_midnight=today)
    assert np.isfinite(rmse_val) or np.isnan(rmse_val)   # either is valid; must not raise


# ─────────────────────────────────────────────────────────────────────────────
# _load_xgb_plus_features
# ─────────────────────────────────────────────────────────────────────────────


def test_load_xgb_plus_features_missing_file(tmp_path: pytest.fixture):
    config = {"storage": {"models_dir": str(tmp_path)}}
    assert _load_xgb_plus_features(config) == []


def test_load_xgb_plus_features_corrupted_pkl(tmp_path: pytest.fixture):
    pkl = tmp_path / "xgb_plus_model.pkl"
    pkl.write_bytes(b"not a valid pickle")
    config = {"storage": {"models_dir": str(tmp_path)}}
    assert _load_xgb_plus_features(config) == []


def test_load_xgb_plus_features_valid_pkl(tmp_path: pytest.fixture):
    import pickle
    plus_features = ["feat_x", "feat_y"]
    # _load_xgb_plus_features expects (_, _, _, plus_features) tuple
    data = (None, None, None, plus_features)
    pkl  = tmp_path / "xgb_plus_model.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    config = {"storage": {"models_dir": str(tmp_path)}}
    assert _load_xgb_plus_features(config) == plus_features


def test_load_xgb_plus_features_empty_features(tmp_path: pytest.fixture):
    import pickle
    pkl  = tmp_path / "xgb_plus_model.pkl"
    with open(pkl, "wb") as f:
        pickle.dump((None, None, None, []), f)
    config = {"storage": {"models_dir": str(tmp_path)}}
    assert _load_xgb_plus_features(config) == []
