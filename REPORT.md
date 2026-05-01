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

## HMM Optimisation — first full run (2026-05-01)

```
python main.py hmm
Runtime: 1.7 minutes (100 trials, 5 folds, 39 viable features)
```

### Optuna study

| Metric | Value |
|---|---|
| Trials completed | 100 / 100 |
| Trials pruned | 0 |
| Features excluded (coverage <50%) | 4 (all max pain) |
| Viable optional features | 39 |
| Best objective (−mean CV LL/sample) | −88.71 |
| Best n_components | 3 |
| Best feature count | 33 (incl. SOL_log_return) |

### Top 5 feature configs

| Rank | k | Features | Objective |
|---|---|---|---|
| 1 | 3 | 33 | −88.71 |
| 2 | 3 | 32 | −86.64 |
| 3 | 3 | 32 | −86.33 |
| 4 | 3 | 31 | −84.39 |
| 5 | 3 | 26 | −84.26 |

All top-5 configs use k=3 regimes. Features consistently selected across top configs:
- All log-returns (BTC, ETH, SOL, VIX)
- All rolling volatilities (24h + 168h, all symbols)
- SOL↔BTC/ETH correlations (24h + 168h)
- BTC lags 2h–24h (1h lag consistently dropped)
- Market close features (XETRA, NYSE, TSE)
- FEMA score, Crypto Fear & Greed, `fed_rate_last_change`
- `VIX_change_24h`, ETH + SOL momentum

Features consistently NOT selected in top configs:
- `BTC_log_return_lag_1h` (1h lag)
- `VIX_zscore` (but `VIX_change_24h` selected)
- `BTC_momentum`
- `fed_rate` level (but signed change selected)

### Final model fit (best config, full dataset)

```
Feature matrix:  8 581 rows × 33 cols
n_components:    3
BIC:             −2 055 153
Log-likelihood:  1 035 683
```

### 7-day SOL/USD forecast

```
Last close:   $83.90  (2026-04-30)
E[+7d]:       $83.15  (−0.9%)
±2σ band:     [$67.42, $102.54]
```

The narrow expected return (−0.9%) reflects the HMM's current-regime view: the model
places the last observation in a low-drift regime consistent with the SOL sideways
consolidation visible since Feb 2026. The wide ±2σ band (±22%) reflects genuine
regime uncertainty over a 168-step horizon.

---

## Known Issues

- **Max Pain** — only 1 row of data. Becomes useful after ≥7 consecutive daily `collect` runs.
- **VIX lag** — VIX always lags Deribit by 1 day because yfinance daily data is not intraday.
  VIX_close for the current partial day is forward-filled from the previous close.
- **GDELT rate-limiting** — 429 responses trigger 3 retries with 65s delay each. On slow days
  this can add ~3 minutes to the collect run.
- **Stock F&G** — CNN endpoint covers ~254 trading days maximum; no deeper history available.
