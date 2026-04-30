# deribit

Automated market data collection from the Deribit public API and regime-change detection
for Solana using Hidden Markov Models with Bayesian feature optimization.

---

## Overview

Two-module Python pipeline:

1. **Collector** — pulls hourly OHLCV data (1h candles) for BTC, ETH, SOL from Deribit and VIX from
   Yahoo Finance on demand at project start. Backfills up to 1 year of history on first run,
   then fetches only missing candles on subsequent runs. No automated scheduler.

2. **HMM** — detects and predicts market regimes for Solana using a Gaussian HMM.
   Features are drawn from BTC, ETH and VIX time series. Optimal feature subset is found via
   Bayesian optimization (Optuna) with time-series K-Fold cross-validation.

---

## Architecture

### Data Flow

```
┌──────────────────┐     ┌──────────────────┐
│   Deribit REST   │     │  Yahoo Finance   │
│  (BTC/ETH/SOL)   │     │      (VIX)       │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────┐
│  DeribitClient  │    │   VixClient      │
│  chunked fetch  │    │  daily → hourly  │
└────────┬────────┘    └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────┐
         │ ParquetRepository│
         │  append-only I/O │
         └────────┬─────────┘
                  │
                  ▼
         data/raw/*.parquet
         (BTC / ETH / SOL / VIX)
                  │
                  ▼
         ┌──────────────────┐
         │  src/hmm/        │
         │  features.py     │◄── BTC, ETH, VIX as auxiliary features
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │  optimizer.py    │  Optuna × TimeSeriesSplit
         │  (feature select)│  → optimal feature subset
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │   model.py       │  GaussianHMM fit
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │   predict.py     │──► regime labels + forecasts
         └──────────────────┘
```

### Component Overview

```
src/
├── collector/
│   ├── deribit_client.py   ← Repository: REST API, chunked OHLCV fetch
│   ├── vix_client.py       ← Repository: yfinance, daily→hourly resample
│   ├── repository.py       ← Repository: Parquet append/load/last_timestamp
│   └── fetcher.py          ← Template Method: orchestrates all clients (on-demand)
│
├── hmm/
│   ├── features.py         ← Strategy: pluggable feature extractors
│   ├── model.py            ← GaussianHMM wrapper
│   ├── optimizer.py        ← Optuna + TimeSeriesSplit K-Fold
│   └── predict.py          ← run() entry point for HMM pipeline
│
└── utils/
    └── paths.py            ← central path resolution from settings.yaml
```

### Sequence: Incremental Fetch

```
Fetcher                  ParquetRepository        DeribitClient
   │                            │                       │
   │── last_timestamp(symbol) ─▶│                       │
   │◀─ timestamp / None ────────│                       │
   │                            │                       │
   │  [if None]                 │                       │
   │  start = now − 365 days    │                       │
   │  [if timestamp]            │                       │
   │  start = last_ts + 1h      │                       │
   │                            │                       │
   │── fetch_ohlcv(start, now) ─────────────────────────▶│
   │◀─ DataFrame ───────────────────────────────────────│
   │                            │                       │
   │── append(symbol, df) ─────▶│                       │
   │                            │── write Parquet ──▶ disk
```

### Sequence: On-Demand Fetch (project start)

```
main.py          Fetcher          ParquetRepository     API (Deribit / VIX)
   │                │                    │                       │
   │── run(config) ▶│                    │                       │
   │                │── last_timestamp ─▶│                       │
   │                │◀─ ts / None ───────│                       │
   │                │                    │                       │
   │                │  start = last_ts + 1h  (or now − 365d)     │
   │                │                    │                       │
   │                │── fetch_ohlcv(start, now) ────────────────▶│
   │                │◀─ DataFrame ───────────────────────────────│
   │                │                    │                       │
   │                │── append(symbol) ─▶│                       │
   │                │                    │── write Parquet ──▶ disk
   │                │                    │                       │
   │◀── done ───────│                    │                       │
```

### Module Breakdown

| Module | Path | Pattern | Responsibility |
|---|---|---|---|
| collector | `src/collector/` | Repository, Template Method | Deribit client, VIX fetch, incremental Parquet storage (on-demand) |
| hmm | `src/hmm/` | Strategy, Factory | Feature engineering, GaussianHMM, Bayesian optimization, prediction |

---

## Setup

**Prerequisites:** Python 3.11+, git

```bash
git clone https://github.com/Dharkelf/deribit.git
cd deribit

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Install pre-commit hooks (once)
pre-commit install
```

---

## Configuration

All runtime parameters live in `config/settings.yaml`. No hard-coded values in source files.

```yaml
symbols:
  deribit:
    - instrument: BTC-PERPETUAL       # Deribit instrument name
      symbol: BTC                     # local file/column identifier
    - instrument: ETH-PERPETUAL
      symbol: ETH
    - instrument: SOL_USDC-PERPETUAL
      symbol: SOL
  vix: "^VIX"                         # yfinance ticker

collector:
  resolution: 60                      # candle interval in minutes (60 = 1h)
  history_days: 365                   # initial backfill window in days

storage:
  raw_dir: data/raw                   # Parquet output directory (gitignored)

hmm:
  n_components: [2, 3, 4]            # regime count search space for Optuna
  n_splits: 5                        # TimeSeriesSplit K-Fold count
  n_trials: 100                      # Optuna optimization trials

logging:
  level: INFO                        # DEBUG / INFO / WARNING / ERROR
```

---

## Usage

**Fetch all market data (backfill on first run, incremental on subsequent runs):**
```bash
python main.py collect
```

**Run the HMM pipeline:**
```bash
python main.py hmm
```

**Run both:**
```bash
python main.py
```

---

## Data

Raw data is stored in `data/raw/` as Parquet files, one file per symbol.

| Column | Type | Description |
|---|---|---|
| `timestamp` | `datetime64[ns, UTC]` | Candle open time |
| `open` | `float64` | Open price |
| `high` | `float64` | High price |
| `low` | `float64` | Low price |
| `close` | `float64` | Close price |
| `volume` | `float64` | Volume |

- Filenames: `BTC.parquet`, `ETH.parquet`, `SOL.parquet`, `VIX.parquet`
- Raw data is append-only — existing files are never overwritten
- All timestamps are UTC

---

## Development

**Run tests:**
```bash
pytest tests/
```

**Run linting and type checks:**
```bash
pre-commit run --all-files
```

**Update dependencies after installing a new package:**
```bash
pip freeze > requirements.txt
git add requirements.txt
```

**Add a new module:**
1. Create `src/<module>/` with `__init__.py`
2. Add tests in `tests/test_<module>.py`
3. Register any new config keys in `config/settings.yaml`
4. Update this README in the same commit

---

## References

- [Deribit public API docs](https://docs.deribit.com) — endpoint `/public/get_tradingview_chart_data`
- [Regime change detection blueprint](https://github.com/sergejschweizer/regimechangedetection)
- [hmmlearn](https://hmmlearn.readthedocs.io)
- [Optuna](https://optuna.readthedocs.io)
- VIX data via `yfinance` ticker `^VIX`
