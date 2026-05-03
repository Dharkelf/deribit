# REPORT — Runtime Behaviour

Documents actual observed behaviour from running the pipeline.
Updated after each significant code change.

---

## Data Collection — observed 2026-05-03

### Parquet files on disk

| File | Rows | Date range | Notes |
|---|---|---|---|
| BTC.parquet | 8 826 | 2025-04-30 → 2026-05-03 05:00 | hourly |
| ETH.parquet | 8 826 | 2025-04-30 → 2026-05-03 05:00 | hourly |
| SOL.parquet | 8 826 | 2025-04-30 → 2026-05-03 05:00 | hourly |
| VIX.parquet | 8 785 | 2025-04-30 → 2026-05-01 | daily ffilled hourly; lags 2 days (yfinance fails on weekends) |
| FEMA.parquet | 368 | 2025-05-01 → 2026-05-03 | daily |
| GDELT.parquet | 368 | 2025-05-01 → 2026-05-03 | daily; almost always rate-limited on first attempt |
| BTC_OPTIONS_MAX_PAIN.parquet | 3 | 2026-05-01 → 2026-05-03 | daily snapshot; XGB+ eligibility ~2026-05-09 (needs ≥200 rows) |
| CRYPTO_FEAR_GREED.parquet | 367 | 2025-05-02 → 2026-05-03 | daily |
| STOCK_FEAR_GREED.parquet | 255 | 2025-05-01 → 2026-05-01 | trading days only (CNN); lags on weekends |
| FED_RATE.parquet | 26 237 | 1954-07-01 → 2026-04-30 | full FRED history; lags 2 days |

### Current signal values (2026-05-03)

| Signal | Value | Interpretation |
|---|---|---|
| SOL/USD close | $84.29 | — |
| Crypto Fear & Greed | 0.47 / 100 | Neutral |
| Stock Fear & Greed (CNN) | 0.67 / 100 | Greed |
| Fed Funds Rate | 3.64% | — |
| Last FOMC change | −0.25 pp | Rate cut |
| GDELT military score | 1.00 | Elevated activity |
| FEMA disaster score | 1.00 | Active DR declarations |

### Collector anomalies

- **VIX** — yfinance returns `YFTzMissingError` on weekends (^VIX has no timezone on non-trading days).
  Last successful fetch: 2026-05-01 (Thursday). VIX is forward-filled until next trading day.
- **GDELT** — API returns HTTP 429 on almost every first attempt; 2–3 retries × 65 s = up to 3 min added
  to every collect run. Data persists gracefully on permanent failure.

---

## Feature Engineering — observed 2026-05-03

### Common DataFrame

```
load_common_dataframe() → 8 826 rows × 28 cols
Range: 2025-04-30 12:00 → 2026-05-03 05:00 UTC
```

BTC/ETH/SOL are inner-joined (authoritative time range); VIX/FEMA/GDELT are left-joined with
`reindex + ffill` so weekend/holiday gaps do not truncate the crypto time series.

### NaN coverage per feature group

| Feature group | NaN share | Cause |
|---|---|---|
| log_return (all symbols) | 0% | first row only, removed by dropna |
| vol_24h, lag features | 0% | — |
| vol_168h, corr_168h, momentum | ~2% | first 168 rows warm-up window |
| max_pain_* | ~100% | 3 snapshots only; needs ≥200 rows for XGB+ (~2026-05-09) |
| crypto_fear_greed | 0% | 367 days available |
| stock_fear_greed | 0% | 255 days available, ffilled over weekends |
| fed_rate, fed_rate_last_change | 0% | full history back to 1954 |

### Features excluded at runtime

- **6 market-close features** — BTC_at_NYSE_close, BTC_at_TSE_close, BTC_at_XETRA_close,
  BTC_return_since_NYSE_close/TSE_close/XETRA_close — excluded because `pandas-market-calendars`
  is not installed. `MarketCloseExtractor` skips silently with a WARNING log.

### FOMC decisions detected (0.10 pp threshold)

```
2024-09-19   −0.50 pp  (emergency cut)
2024-11-08   −0.25 pp
2024-12-19   −0.25 pp
2025-09-18   −0.25 pp
2025-10-30   −0.25 pp
2025-12-11   −0.25 pp   ← last confirmed change
current rate: 3.64%
```

---

## HMM Optimisation — current (2026-05-02, k=5 fixed)

**Settings:** k fixed at 5 regimes, 200 Optuna trials, 5-fold TimeSeriesSplit, composite regime score.

```
score = 3.0·avg_self_transition + 1.5·min_state_fraction
      − 0.25·median_run_days − 2.5·avg_entropy
      + 0.05·loglik_per_obs_feat
```

Eligibility gate: min_state_fraction ≥ 0.02 per fold.

### Optuna study (200 trials, fully completed)

| Metric | Value |
|---|---|
| Trials completed | 200 / 200 |
| Trials pruned | 0 |
| Features excluded (coverage <50%) | 10 (4 max pain + 6 market-close) |
| Viable optional features | 33 |
| Best score | 2.9879 |
| Best n_components | 5 |
| Best feature count | 14 (13 optional + SOL_log_return always included) |

### Best feature subset (best_features.json)

```
ETH_log_return, BTC_vol_168h, ETH_vol_24h, ETH_vol_168h, SOL_vol_24h,
VIX_vol_24h, SOL_BTC_corr_24h, ETH_momentum,
BTC_log_return_lag_1h, BTC_log_return_lag_2h, BTC_log_return_lag_3h,
BTC_log_return_lag_6h, BTC_log_return_lag_18h,
FEMA_score
```

### 7-day SOL/USD HMM forecast (2026-05-03)

```
Last close:   $84.29
E[+7d]:       $83.74  (−0.7%)
±2σ band:     [$68.36, $102.57]
```

---

## XGBoost Forecast — observed 2026-05-03

### Model configuration

```
n_estimators=1500, learning_rate=0.015, max_depth=5
subsample=0.8, colsample_bytree=0.8
min_child_weight=3, reg_alpha=0.1, reg_lambda=1.5
tree_method=hist  (NEON on Apple Silicon M5)
Quantile models: n_estimators=800
```

### Feature filtering

`_filter_24h_features()` removes `*_168h` and `*_momentum` from the 14 HMM-selected features,
leaving 11 features for recursive forecasting (BTC_vol_168h, ETH_vol_168h, ETH_momentum removed).
SOL_log_return is always added (recursive target). Training matrix: **8 796 rows × 12 features**.

Training time on Apple M5: ~3 seconds.

### In-data quality (last 72 h, real features — no recursion)

```
RMSE:    $0.42
adj-R²:  0.19
```

### Forecast (2026-05-03, cutoff = 2026-05-02 23:00 UTC)

```
Last close:       $84.29
E[today 23:00]:   $83.07
CI [10%–90%]:     [$73.04, $95.02]
```

### XGB+ feature search

Evaluated 26 candidate features (viable but not in HMM subset).
Quick base adj-R² on 72 h window: −3.09 (short window, high variance → negative).

**10 features with positive Δadj-R²:**
```
fed_rate, GDELT_military_score, BTC_vol_24h, SOL_ETH_corr_168h,
crypto_fear_greed, SOL_ETH_corr_24h, BTC_log_return, SOL_vol_168h,
stock_fear_greed, VIX_vol_168h
```

All 10 passed the dominance check (HMM-base mean importance ≥ added feature mean importance).

```
XGB+ in-data RMSE:  $0.95  (higher than base — more features on short window)
XGB+ adj-R²:        −4.00
XGB+ E[today 23:00]: $83.97
```

**Note:** adj-R² is negative for both base (−3.09 with 100 trees) and XGB+ (−4.00) on the 72 h
in-data window. This is expected: price prediction on a short noisy window with many features
naturally produces negative adj-R² (SS_res > SS_tot after d.o.f. penalty). The selection still
works because all candidates are evaluated against the same baseline — the relative Δadj-R² is
meaningful even when absolute values are negative.

---

## NeuralProphet Forecast — observed 2026-05-03

```
Training window: 4 000 rows, n_lags=168, n_forecasts=48, 14 lagged regressors
Learning rate: 0.001 (fixed — avoids PyTorch ≥2.6 LR-finder checkpoint bug)
Training time: ~17 s on Apple M5
```

```
Last close:      $84.29
E[+7d]:          $81.80
CI [10%–90%]:    [$65.27, $81.80]
In-data RMSE:    $3.45
```

NeuralProphet always retrains from scratch (PyTorch ≥2.6 Trainer not safely picklable).
Results vary slightly between runs due to stochastic training.

---

## Known Issues

- **VIX weekend gap** — yfinance fails on Saturdays/Sundays with `YFTzMissingError`. VIX is
  forward-filled from the last trading day until the next Monday open. Acceptable for a slow-
  moving index used as a regime feature.
- **Max Pain** — only 3 rows accumulated. XGB+ requires ≥200 rows; auto-qualifies ~2026-05-09.
- **GDELT rate-limiting** — 429 on almost every first attempt; 2–3 retries × 65 s per run.
  Consider running collect in off-peak hours. Data persists on permanent failure.
- **XGB+ adj-R² negative** — expected on a 72 h in-data window with many features. Selection
  criterion is relative Δadj-R² vs baseline, which remains valid.
- **6 market-close features excluded** — `pandas-market-calendars` not installed. These features
  (BTC price relative to NYSE/TSE/XETRA close) would improve regime detection around market opens.
- **Stock F&G** — CNN endpoint covers ~255 trading days maximum; no deeper history available.
- **NeuralProphet CI lower = upper** — `np_lo == np_exp` observed occasionally; NeuralProphet
  quantile estimation can collapse on short training windows or poor convergence.
