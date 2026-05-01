# deribit

Automated market data collection and 1-week Solana price forecast using Hidden Markov Models
with Bayesian feature selection.

---

## Overview

Two-module Python pipeline:

1. **Collector** — fetches hourly OHLCV data for BTC, ETH, SOL from Deribit, VIX from Yahoo
   Finance, plus five daily soft signals: FEMA disaster score, GDELT military activity score,
   BTC Options Max Pain, Crypto Fear & Greed Index, Stock Fear & Greed Index, and US Federal
   Funds Rate. Incremental on-demand updates; no scheduler.

2. **HMM** — detects market regimes (2–4 states) and forecasts the most probable SOL/USD price
   7 days ahead with ±2σ confidence bands. Feature subset selected via Optuna Bayesian optimisation
   with `TimeSeriesSplit` walk-forward cross-validation.

---

## Architecture

### Data Flow

```
┌──────────────┐  ┌───────────┐  ┌──────────────┐  ┌──────────────┐
│ Deribit REST │  │ yfinance  │  │ alternative  │  │  CNN dataviz │
│ BTC/ETH/SOL  │  │   (VIX)   │  │  .me (F&G)   │  │  Stock F&G   │
└──────┬───────┘  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘
       │                │                │                  │
┌──────┴───────┐  ┌─────┴─────┐  ┌──────┴───────┐  ┌──────┴───────┐
│ Deribit/Vix  │  │  Options  │  │  FEMA/GDELT  │  │   FedClient  │
│  Client      │  │  Client   │  │  Clients     │  │  (FRED DFF)  │
└──────┬───────┘  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘
       └────────────────┴────────────────┴──────────────────┘
                                  │
                        ┌─────────▼──────────┐
                        │  ParquetRepository │
                        │   append-only I/O  │
                        └─────────┬──────────┘
                                  │
                          data/raw/*.parquet
                                  │
                        ┌─────────▼──────────┐
                        │  features.py        │  49 optional features
                        │  (Strategy pattern) │  + SOL_log_return (forced)
                        └─────────┬──────────┘
                                  │
                        ┌─────────▼──────────┐
                        │  optimizer.py       │  Optuna TPESampler
                        │  (Bayesian search)  │  × TimeSeriesSplit CV
                        └─────────┬──────────┘
                                  │
                        ┌─────────▼──────────┐
                        │  model.py           │  GaussianHMM fit
                        │  (GaussianHMM)      │  BIC model selection
                        └─────────┬──────────┘
                                  │
                        ┌─────────▼──────────┐
                        │  predict.py         │  7-day forecast
                        │  k-step Markov      │  ±2σ confidence band
                        └────────────────────┘
```

### Component Overview

```
src/
├── collector/
│   ├── deribit_client.py      ← Repository: chunked OHLCV fetch from Deribit REST
│   ├── vix_client.py          ← Repository: yfinance daily → hourly resample
│   ├── options_client.py      ← BTC Options Max Pain (Deribit, 7d + 30d windows)
│   ├── fema_client.py         ← FEMA OpenFEMA API → daily disaster severity [0,1]
│   ├── gdelt_client.py        ← GDELT DOC 2.0 → daily US military activity [0,1]
│   ├── fear_greed_client.py   ← Crypto F&G (alternative.me) + Stock F&G (CNN) [0,1]
│   ├── fed_client.py          ← FRED DFF → daily Fed Funds Rate + last FOMC change
│   ├── repository.py          ← Repository: Parquet append / load / last_timestamp
│   ├── fetcher.py             ← Template Method: orchestrates all 8 sources
│   ├── inspect.py             ← Standalone: 12-panel data visualisation
│   └── inspect_opt_regime.py  ← Standalone: top 3-5 regime models over 1 year
│
├── hmm/
│   ├── features.py            ← Strategy: 13 pluggable feature extractors
│   ├── model.py               ← GaussianHMMModel wrapper (BIC, save/load)
│   ├── optimizer.py           ← Optuna TPESampler + TimeSeriesSplit CV
│   └── predict.py             ← run() pipeline: optimise → fit → forecast → plot
│
└── utils/
    └── paths.py               ← Central path resolution from settings.yaml
```

### Incremental Fetch Sequence

```
Fetcher               ParquetRepository        External API
   │                        │                       │
   │── last_timestamp() ───▶│                       │
   │◀─ ts / None ───────────│                       │
   │                        │                       │
   │  start = last_ts + 1h  (or now − 365d if None) │
   │                        │                       │
   │── fetch(start, now) ──────────────────────────▶│
   │◀─ DataFrame ───────────────────────────────────│
   │                        │                       │
   │── append(symbol, df) ─▶│                       │
   │                        │── write Parquet ──▶ disk
```

### Forecast Maths (7-day, hourly)

```
π₀  = posterior state distribution at last timestamp
      (model.predict_proba(X)[-1])

πⱼ  = π₀ · Aʲ          A = HMM transition matrix

μⱼ  = πⱼ · means[:,sol_idx]          expected SOL log-return at step j
σ²ⱼ = πⱼ · (means² + diag_cov)[:,sol_idx] − μⱼ²

E[SOL t+k] = SOL_last · exp(Σⱼ μⱼ)     k = 168 (7 days × 24h)
CI ±2σ     = SOL_last · exp(Σⱼ μⱼ ± 2·√(Σⱼ σ²ⱼ))
```

---

## Setup

**Prerequisites:** Python 3.11+, git

```bash
git clone https://github.com/Dharkelf/deribit.git
cd deribit

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pre-commit install
```

---

## Configuration

All runtime parameters in `config/settings.yaml`. No hard-coded values in source.

```yaml
symbols:
  deribit:
    - instrument: BTC-PERPETUAL
      symbol: BTC
    - instrument: ETH-PERPETUAL
      symbol: ETH
    - instrument: SOL_USDC-PERPETUAL
      symbol: SOL
  vix: "^VIX"

collector:
  resolution: 60          # candle interval in minutes (60 = 1h)
  history_days: 365       # initial backfill window

storage:
  raw_dir: data/raw
  models_dir: data/processed/models

options:
  max_pain_days_ahead: 30       # expiry window for Max Pain mean (calendar days)
  max_pain_days_ahead_short: 7  # short window

hmm:
  n_components: [2, 3, 4]  # regime count search space
  n_splits: 5               # TimeSeriesSplit folds
  n_trials: 100             # Optuna trials
  random_state: 42
  covariance_type: full
  n_iter: 200               # max EM iterations per fit

logging:
  level: INFO
```

---

## Usage

```bash
# Fetch all data (backfill on first run, incremental thereafter)
python main.py collect

# Run HMM pipeline (optimise features → fit → 7-day forecast)
python main.py hmm

# Both in sequence
python main.py

# Visual inspection — 12 panels of raw collected data
python -m src.collector.inspect

# Visual inspection — top 3-5 regime models over last year
python -m src.collector.inspect_opt_regime
```

---

## Data Sources

| Source | API | Auth | Output | Parquet |
|---|---|---|---|---|
| Deribit OHLCV | `/public/get_tradingview_chart_data` | none | hourly BTC/ETH/SOL | BTC/ETH/SOL.parquet |
| VIX | yfinance `^VIX` | none | daily → hourly ffill | VIX.parquet |
| BTC Options Max Pain | `/public/get_book_summary_by_currency` | none | daily, 7d + 30d windows | BTC_OPTIONS_MAX_PAIN.parquet |
| FEMA Disasters | OpenFEMA `/DisasterDeclarationsSummaries` | none | daily score [0,1] | FEMA.parquet |
| GDELT Military | DOC 2.0 API | none | daily score [0,1] | GDELT.parquet |
| Crypto Fear & Greed | alternative.me `/fng/` | none | daily [0,1] | CRYPTO_FEAR_GREED.parquet |
| Stock Fear & Greed | CNN dataviz endpoint | none | daily [0,1] | STOCK_FEAR_GREED.parquet |
| US Fed Funds Rate | FRED DFF CSV export | none | daily %, FOMC change | FED_RATE.parquet |

All timestamps are UTC. All Parquet files are append-only.

### OHLCV Schema (BTC / ETH / SOL / VIX)

| Column | Type | Description |
|---|---|---|
| `timestamp` | `datetime64[ms, UTC]` | Candle open time (index) |
| `open` | `float64` | Open price |
| `high` | `float64` | High price |
| `low` | `float64` | Low price |
| `close` | `float64` | Close price |
| `volume` | `float64` | Trade volume |

---

## Feature Engineering

`load_common_dataframe()` inner-joins all symbols on UTC timestamp, reindexes to a complete
24h hourly grid, and forward-fills gaps. SOL_log_return is always included in the HMM
observation matrix; all other features are selected by Optuna.

| Extractor | Features | Count |
|---|---|---|
| LogDiffReturnExtractor | `{BTC,ETH,SOL,VIX}_log_return` | 4 |
| RollingVolatilityExtractor | `{BTC,ETH,SOL,VIX}_vol_{24,168}h` | 8 |
| RollingCorrelationExtractor | `SOL_{BTC,ETH}_corr_{24,168}h` | 4 |
| VixLevelExtractor | `VIX_zscore`, `VIX_change_24h` | 2 |
| MomentumExtractor | `{BTC,ETH,SOL}_momentum` | 3 |
| BtcLagExtractor | `BTC_log_return_lag_{1,2,3,6,12,18,24}h` | 7 |
| MarketCloseExtractor | `BTC_at_{XETRA,NYSE,TSE}_close`, `BTC_return_since_*` | 6 |
| MaxPainExtractor | `max_pain_{diff_usd,diff_pct,7d_diff_usd,7d_diff_pct}` | 4 |
| DisasterExtractor | `FEMA_score` | 1 |
| MilitaryExtractor | `GDELT_military_score` | 1 |
| CryptoFearGreedExtractor | `crypto_fear_greed` | 1 |
| StockFearGreedExtractor | `stock_fear_greed` | 1 |
| FedRateExtractor | `fed_rate`, `fed_rate_last_change` | 2 |
| **Total optional** | | **44** |

Features from sources not yet collected (Max Pain, F&G, Fed) fall back to NaN;
`build_feature_matrix()` drops NaN rows so affected features are naturally excluded
from optimization until data accumulates.

---

## Development

```bash
# Run all tests
pytest tests/

# Lint + type check
pre-commit run --all-files

# Pin dependencies after installing a new package
pip freeze > requirements.txt
git add requirements.txt
```

**Adding a new data source:**
1. Create `src/collector/<source>_client.py`
2. Add `_fetch_<source>()` in `fetcher.py`
3. Add loading in `features.py::load_common_dataframe()`
4. Add extractor class + register in `ALL_EXTRACTORS`
5. Add tests in `tests/test_features.py`
6. Update `inspect.py` panels
7. Update this README and REPORT.md in the same commit

---

## Known Limitations

- **BTC Options Max Pain** requires at least 1 day of collected data before features are usable.
  The first run produces 1 row; subsequent daily `collect` runs accumulate history.
- **GDELT** has a rate limit of ~1 request/minute; the client retries 3 times with 65s delay.
- **Stock Fear & Greed** (CNN) covers ~254 trading days; older history is unavailable.
- **MarketCloseExtractor** requires `pandas-market-calendars`. If unavailable, columns are NaN.
- HMM training assumes stationarity of log-returns; structural breaks (e.g. exchange hacks)
  can cause regime mislabelling for short windows after the event.

---

## References

- [Deribit API docs](https://docs.deribit.com) — `/public/get_tradingview_chart_data`
- [hmmlearn](https://hmmlearn.readthedocs.io)
- [Optuna](https://optuna.readthedocs.io)
- [FRED DFF series](https://fred.stlouisfed.org/series/DFF) — no API key required
- [alternative.me Crypto F&G](https://alternative.me/crypto/fear-and-greed-index/)
