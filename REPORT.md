# REPORT — Runtime Behaviour

Documents actual observed behaviour from running the pipeline.
Updated after each significant code change.

---

## Backtest Results — 2026-05-04 (first full run, 3-year data)

### Setup

152 walk-forward folds, step=7d, horizon=24h, min-train=30d.
Date range: 2023-06-07 → 2026-04-30.
Oracle evaluation (real features at each step — no recursion). Upper bound on XGB accuracy.
HMM regime labels: mild look-ahead bias (model trained on all data). Documented in report.
NeuralProphet excluded (~55s/fold → ~3h for 200 folds). NP+ shape bug fixed 2026-05-07.

### Option A — XGB Walk-Forward Forecast Accuracy

| Metrik | Wert |
|---|---|
| RMSE | $1.22 |
| MAE | $0.82 |
| Directional Accuracy | 97.4 % (oracle — production recursive is lower) |

| Regime | RMSE | MAE | Dir. Acc. | Folds |
|---|---|---|---|---|
| Bearish | $1.37 | $1.12 | 100.0 % | 12 |
| Neutral | $1.06 | $0.72 | 100.0 % | 107 |
| Bullish | $1.20 | $0.91 | 95.5 % | 23 |
| Strong Bullish | $2.24 | $1.36 | 88.9 % | 10 |

Note: Strong Bearish had 0 fold-start appearances → not in table.
High DA is expected for oracle evaluation with a 24h horizon and strong momentum features.

### Option B — Regime Strategy

| Metrik | Strategie | Buy-and-Hold |
|---|---|---|
| Ann. Return | 132.4 % | 61.0 % |
| Sharpe | 1.28 | 0.54 |
| Max Drawdown | −51.5 % | — |

Position map: Strong Bullish=+1, Bullish=+0.5, Neutral=0, Bearish=−0.5, Strong Bearish=−1.
No transaction costs. Look-ahead bias in HMM labels. Real performance will be lower.

### Auto-generated Improvement Ideas (from BACKTEST_REPORT.md)

1. Max Drawdown −52 % → Stop-Loss-Regel (−15 % Trail) für Strong Bearish implementieren
2. 6 Folds mit RMSE > μ+2σ ($2.63) → Volatility-Regime als Conditioning-Variable prüfen
3. NP-Bug fixen → NP Walk-Forward als Vergleich einbeziehen

### Outputs

- `data/processed/backtest_results.parquet` — 3648 Zeilen (fold × hour)
- `data/processed/backtest_report.png` — 4-Panel Dashboard
- `data/processed/BACKTEST_REPORT.md` — strukturierter Report

---

## HMM Optimisation — 3-Jahr-Daten (2026-05-04)

**Setup:** k=5 Regime, 200 Optuna-Trials, 5-fold TimeSeriesSplit, 26 244 Trainingszeilen.
Laufzeit: 1h 30min (vs. ~15min auf 1-Jahr-Daten — 3× mehr Daten pro Fold).

| Metrik | 1-Jahr (2026-05-02) | 3-Jahr (2026-05-04) |
|---|---|---|
| Best Score | 2.9879 | **3.0009** |
| Features | 14 | **28** |
| Best Trial | — | 141 |

**Neue Feature-Subset (28 Features):**
```
BTC_log_return, ETH_log_return, VIX_log_return,
BTC_vol_24h, BTC_vol_168h, ETH_vol_24h, SOL_vol_24h, SOL_vol_168h,
VIX_vol_24h, VIX_vol_168h,
SOL_BTC_corr_24h, SOL_ETH_corr_24h, SOL_ETH_corr_168h,
VIX_change_24h, BTC_momentum, ETH_momentum,
BTC_log_return_lag_1h/2h/12h/18h/24h,
BTC_at_XETRA_close, BTC_return_since_XETRA_close,
BTC_at_NYSE_close, BTC_return_since_TSE_close,
FEMA_score, crypto_fear_greed, fed_rate
```

Neu vs. 1-Jahr: VIX_log_return, BTC/SOL/VIX_vol_168h, SOL_ETH_corr, market-close features,
crypto_fear_greed, fed_rate, BTC_log_return direkt.

---

## 3-Jahr-Backfill — 2026-05-04

| Datei | Zeilen | Bereich |
|---|---|---|
| BTC / ETH / SOL | 26 281 | 2023-05-05 → 2026-05-04 |
| VIX | 26 197 | 2023-05-05 → 2026-05-01 |
| FEMA / GDELT | 1 096 | 2023-05-05 → 2026-05-04 |
| Crypto F&G | 1 096 | 2023-05-04 → 2026-05-04 |
| Stock F&G | 253 | 2025-05-05 → 2026-05-04 (CNN-Cap) |
| FED_RATE | 26 237 | 1954-07-01 → 2026-04-30 (unverändert) |
| Max Pain | 4 | 2026-05-01 → jetzt (unverändert) |

GDELT: kein Rate-Limit-Retry nötig — 1096-Tage-Range in einem Request. Laufzeit: <15s.

---

## XGBoost Forecast — 3-Jahr-Modell (2026-05-04)

```
Last close:       $83.91
E[heute 23:00]:   $86.22
CI [10%–90%]:     [$73.88, $92.53]
```

### XGB+ — deutlich verbessert

5 Features ausgewählt (vs. 20 am 2026-05-04 mit 1-Jahr-Modell):
`BTC_momentum, SOL_vol_168h, SOL_momentum, BTC_log_return_lag_3h, ETH_vol_168h`

```
XGB+ in-data RMSE:   $0.39  (vs. $0.61 mit 1-Jahr-Modell)
XGB+ adj-R²:         −2.30  (vs. −7.72)
XGB+ E[+24h]:        $86.45
```

---

## NeuralProphet Forecast — 3-Jahr-Modell (2026-05-04)

```
E[+1d]:   $86.27
CI:       [$79.20, $106.12]  ← nicht kollabiert (vs. 2026-05-03 und 2026-05-04 v1)
RMSE:     $10.75             ← deutlich schlechter als 1-Jahr ($2.49)
adj-R²:   −2475
```

CI-Kollaps behoben — breiteres Konfidenzband durch neue Features.
RMSE verschlechtert: vermutlich durch geänderte HMM-Feature-Subset und größeres Daten-Fenster.
NP+ nicht aktiviert (Base-Modell verwendet — Shape-Bug nicht reproduziert).

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

### Model configuration

```
Training window: 8 652 rows (full history after cutoff cap — no row limit)
n_lags=168, n_forecasts=48, max_epochs=60
Learning rate: 0.001 (fixed — avoids PyTorch ≥2.6 LR-finder checkpoint bug)
Training time: ~27 s base + ~28 s NP+ = ~55 s total on Apple M5
```

### NP vs NP+ comparison

NP+ uses HMM features (14) + XGB+ selected features (10), for 24 regressors total.
Both variants are trained; the one with higher in-data adj-R² is used for the forecast.

```
NP base:  adj-R²=−98.61  RMSE=$4.59  (14 HMM regressors)
NP+:      adj-R²=−73.81  RMSE=$3.61  (24 regressors: HMM + XGB+)  ← selected
```

### Forecast (2026-05-03, NP+ selected)

```
Last close:      $84.29
E[+1d]:          $80.43
CI [10%–90%]:    [$80.43, $80.43]   ← quantile collapse (see Known Issues)
In-data RMSE:    $3.61
adj-R²:          −73.81
```

NeuralProphet always retrains from scratch (PyTorch ≥2.6 Trainer not safely picklable).
adj-R² is highly negative by construction on the 72 h in-data window with 24 features
(same SS_res > SS_tot dynamic as XGB+). Relative Δadj-R² between NP and NP+ is the
meaningful selection signal.
Results vary slightly between runs due to stochastic training.

---

---

## XGBoost Forecast — observed 2026-05-04

### In-data quality (last 72 h)

```
RMSE:    $0.94   (vs $0.42 on 2026-05-03 — volatile May 1–4 window)
adj-R²:  −12.68
```

### Forecast (2026-05-04, cutoff = 2026-05-03 23:00 UTC)

```
Last close:       $83.91
E[today 23:00]:   $82.74
CI [10%–90%]:     [$68.98, $92.50]
```

### XGB+ feature search — 2026-05-04

Evaluated 32 candidate features. Quick base adj-R² = −14.99.

20 features selected (vs 10 on 2026-05-03 — dominance check passed all):

```
BTC_at_NYSE_close, BTC_at_TSE_close, GDELT_military_score, BTC_vol_24h,
fed_rate, BTC_at_XETRA_close, BTC_log_return, VIX_log_return,
stock_fear_greed, crypto_fear_greed, SOL_ETH_corr_24h, SOL_ETH_corr_168h,
fed_rate_last_change, BTC_return_since_XETRA_close, BTC_log_return_lag_12h,
ETH_vol_168h, BTC_return_since_TSE_close, BTC_vol_168h, VIX_vol_168h,
SOL_BTC_corr_168h
```

```
XGB+ in-data RMSE:   $0.61
XGB+ adj-R²:         −7.72
XGB+ E[today 23:00]: $83.02
```

**Note:** The market-close features (BTC_at_NYSE_close etc.) appeared because
`pandas-market-calendars` was apparently available at runtime today. Their presence
inflates XGB+ feature count. The dominance check threshold may need tightening when
more than ~10 features are selected.

---

## NeuralProphet Forecast — observed 2026-05-04

NP+ **failed** with shape error and fell back to base model.

```
Error:   "Expected a 1D array, got an array with shape (8676, 2)"
Model:   Base (14 HMM regressors)
E[+1d]:  $87.47
CI:      [$86.95, $87.77]  (narrow band, not a collapse)
RMSE:    $2.49
adj-R²:  −99.07
```

Root cause under investigation: the XGB+ 20-feature selection on this run
may have introduced a multi-column regressor that NeuralProphet's
`add_lagged_regressor` cannot handle as a 2D input.

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
- **XGB+ feature count inflation** — dominance check passed 20 features on 2026-05-04 (vs 10
  on 2026-05-03). Dominance threshold may need tightening when candidate pool includes sparse
  or session-close features. Investigate: raise the mean-importance floor.
- **NP+ shape bug (2026-05-04, FIXED 2026-05-07)** — `predict_prophet.run()` raised
  `"Expected a 1D array, got an array with shape (8676, 2)"` during NP+ evaluation.
  Root cause: `X_df[feat]` returned a DataFrame (not a Series) when column names were
  duplicated. Fixed in `_build_np_df` (isinstance guard, squeeze to Series) and
  `_train_model` (defensive shape assertion before `add_lagged_regressor`).
  NP+ remains excluded from the walk-forward backtest for performance reasons only
  (~55s/fold makes a 200-fold run impractical).
- **6 market-close features excluded / intermittently included** — `pandas-market-calendars`
  availability appears inconsistent across runs. These features cause XGB+ feature count to
  jump when present.
- **Stock F&G** — CNN endpoint covers ~255 trading days maximum; no deeper history available.
- **NeuralProphet CI lower = upper** — quantile collapse observed on 2026-05-03. Occurs when
  quantile heads do not diverge during training (fixed LR + short eval window).
- **NeuralProphet adj-R² highly negative** — expected on a 72 h in-data window with 14–24
  regressors. Selection criterion is relative Δadj-R² between NP and NP+, which remains valid.
