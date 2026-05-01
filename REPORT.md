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
The optimizer will naturally avoid them (pruned trial if < n_components × 10 rows).

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

## HMM — smoke test (observed earlier session)

Model selection on 8 581 × 36 feature matrix (no max pain, before F&G/Fed were added):

| n_components | BIC |
|---|---|
| 2 | − |
| 3 | − |
| 4 | −700 396 (winner) |

Regime frequencies (n=4):
- Regime 0: 38.9%
- Regime 1: 15.2%
- Regime 2: 24.6%
- Regime 3: 21.3%

Convergence warnings from hmmlearn are expected and normal — they indicate the EM
algorithm approached numerical precision near the optimum.

---

## Known Issues

- **Max Pain** — only 1 row of data. Becomes useful after ≥7 consecutive daily `collect` runs.
- **VIX lag** — VIX always lags Deribit by 1 day because yfinance daily data is not intraday.
  VIX_close for the current partial day is forward-filled from the previous close.
- **GDELT rate-limiting** — 429 responses trigger 3 retries with 65s delay each. On slow days
  this can add ~3 minutes to the collect run.
- **Stock F&G** — CNN endpoint covers ~254 trading days maximum; no deeper history available.
