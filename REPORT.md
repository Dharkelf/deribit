# REPORT — Runtime Behaviour

Documents actual observed behaviour from running the pipeline.
Updated after each significant code change.

---

## Data Collection — observed 2026-05-01

### Parquet files on disk

| File | Rows | Date range | Notes |
|---|---|---|---|
| BTC.parquet | 8 792 | 2025-04-30 → 2026-05-01 | hourly |
| ETH.parquet | 8 792 | 2025-04-30 → 2026-05-01 | hourly |
| SOL.parquet | 8 792 | 2025-04-30 → 2026-05-01 | hourly |
| VIX.parquet | 8 761 | 2025-04-30 → 2026-04-30 | daily ffilled hourly; lags 1 day behind Deribit |
| FEMA.parquet | 366 | 2025-05-01 → 2026-05-01 | daily |
| GDELT.parquet | 366 | 2025-05-01 → 2026-05-01 | daily |
| BTC_OPTIONS_MAX_PAIN.parquet | 1 | 2026-05-01 | single snapshot; accumulates with daily collect |
| CRYPTO_FEAR_GREED.parquet | 365 | 2025-05-02 → 2026-05-01 | daily |
| STOCK_FEAR_GREED.parquet | 254 | 2025-05-01 → 2026-05-01 | trading days only (CNN) |
| FED_RATE.parquet | 26 236 | 1954-07-01 → 2026-04-29 | full FRED history; lags 2 days |

### Current signal values (2026-05-01)

| Signal | Value | Interpretation |
|---|---|---|
| SOL/USD close | $84.10 | — |
| Crypto Fear & Greed | 0.26 / 100 | Extreme Fear |
| Stock Fear & Greed (CNN) | 0.66 / 100 | Greed |
| Fed Funds Rate | 3.64% | — |
| Last FOMC change | −0.25 pp | Rate cut |

---

## Feature Engineering — observed 2026-05-01

### Common DataFrame

```
load_common_dataframe() → 8 749 rows × 28 cols
Range: 2025-04-30 → 2026-04-30
```

VIX lags Deribit by one day; inner-join clips the most recent Deribit row.

### NaN coverage per feature group

| Feature group | NaN share | Cause |
|---|---|---|
| log_return (all symbols) | 0% | first row only (1 of 8 749), removed by dropna |
| vol_24h, lag features | 0% | — |
| vol_168h, corr_168h, momentum | 2% | first 168 rows (warm-up window) |
| max_pain_* | **100%** | only 1 collection snapshot so far; excluded from optimization |
| crypto_fear_greed | 0% | 365 days available |
| stock_fear_greed | 0% | 254 days available, ffilled over weekends |
| fed_rate, fed_rate_last_change | 0% | full history back to 1954 |

### Feature matrix dimensions

| Config | Shape |
|---|---|
| All features except max pain | 8 581 rows × 40 cols |
| All features including max pain | 0 rows (100% NaN → dropna kills all rows) |

**Conclusion:** Max Pain features must be excluded until ≥7 days of daily collection have run.
The optimizer dynamically detects coverage (<50% non-NaN threshold) and excludes them automatically.

### FOMC decisions detected (0.10 pp threshold)

Most recent decisions from FRED DFF:
```
2024-09-19   −0.50 pp  (emergency cut)
2024-11-08   −0.25 pp
2024-12-19   −0.25 pp
2025-09-18   −0.25 pp
2025-10-30   −0.25 pp
2025-12-11   −0.25 pp   ← last confirmed change
current rate: 3.64%
```

Daily noise (±0.01 pp) is correctly filtered by the 0.10 pp threshold.

---

## HMM Optimisation — run 2 (2026-05-01, blueprint selection score)

**Objective changed** from mean CV log-likelihood per sample to composite regime-quality score:
```
score = 3.0·avg_self_transition + 1.5·min_state_fraction
      − 0.25·median_run_days − 2.5·avg_entropy
      + 0.05·loglik_per_obs_feat
```
Eligibility gate: min_state_fraction ≥ 0.05 per fold (collapsed-state models pruned).
Optuna minimises −mean(score); higher score = better regime structure.

```
python main.py hmm
Runtime: ~2.9 minutes (100 trials, 5 folds, 33 viable optional features)
```

### Optuna study

| Metric | Value |
|---|---|
| Trials completed | 99 / 100 |
| Trials pruned | 1 |
| Features excluded (coverage <50%) | 10 (4 max pain + 6 market-close; pandas-market-calendars not installed) |
| Viable optional features | 33 |
| Best score (−objective) | 3.2904 |
| Best n_components | 2 |
| Best feature count | 14 (13 optional + SOL_log_return) |

### Top 5 feature configs

| Rank | k | Features | Score |
|---|---|---|---|
| 1 | 2 | 14 | 3.2904 |
| 2 | 2 | 14 | 3.2646 |
| 3 | 2 | 17 | 3.2530 |
| 4 | 2 | 12 | 3.2496 |
| 5 | 2 | 12 | 3.2375 |

All top-5 configs use k=2 regimes. The selection score heavily penalises
state collapse (min_state_fraction gate) and uncertain assignments (avg_entropy),
which disfavours k=3 on this dataset length.

Features consistently selected across top configs:
- `ETH_log_return`, `BTC_log_return` (log-returns)
- `ETH_vol_168h`, `SOL_vol_24h`, `VIX_vol_24h` (volatility)
- `SOL_ETH_corr_24h`, `VIX_change_24h` (cross-asset)
- `BTC_log_return_lag_6h`, `BTC_log_return_lag_12h` (medium-term BTC lags)

Features consistently NOT selected:
- `BTC_log_return_lag_1h`, `BTC_log_return_lag_2h` (very short lags)
- `VIX_zscore`, `BTC_momentum`, `SOL_momentum`, `ETH_momentum`
- `fed_rate_last_change` (fed level `fed_rate` appears in rank 3)
- `FEMA_score`, `GDELT_military_score`, `stock_fear_greed`

`crypto_fear_greed` appears in rank 4 (only new-signal selected in top 5).

### Final model fit (best config, full dataset)

```
Feature matrix:  8 581 rows × 14 cols
n_components:    2
BIC:             −792 415
Log-likelihood:  397 299
```

### 7-day SOL/USD forecast

```
Last close:   $83.90  (2026-04-30)
E[+7d]:       $83.53  (−0.4%)
±2σ band:     [$68.24, $102.24]
```

k=2 model (Bearish / Bullish) places the last observation in a low-drift regime.
The wide ±2σ band (±20%) reflects cumulative uncertainty over 168 hourly steps.

---

## HMM Optimisation — run 1 (2026-05-01, CV log-likelihood — superseded)

Objective was mean CV log-likelihood per sample. Resulted in k=3 with 33 features.
Superseded by run 2 with blueprint selection score.

---

## Known Issues

- **Max Pain** — only 1 row of data. Becomes useful after ≥7 consecutive daily `collect` runs.
- **VIX lag** — VIX always lags Deribit by 1 day because yfinance daily data is not intraday.
  VIX_close for the current partial day is forward-filled from the previous close.
- **GDELT rate-limiting** — 429 responses trigger 3 retries with 65s delay each. On slow days
  this can add ~3 minutes to the collect run.
- **Stock F&G** — CNN endpoint covers ~254 trading days maximum; no deeper history available.
