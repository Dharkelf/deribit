# deribit

Automated market data collection from the Deribit public API and regime-change detection
for Solana using Hidden Markov Models with Bayesian feature optimization.

---

## Overview

Two-module Python pipeline:

1. **Collector** — pulls hourly OHLCV data for BTC, ETH, SOL from Deribit and VIX from Yahoo Finance.
   Runs 24/7 via APScheduler, backfills up to 1 year of history, stores data as Parquet files locally.

2. **HMM** — detects and predicts market regimes for Solana using a Gaussian HMM.
   Features are drawn from BTC, ETH and VIX time series. Optimal feature subset is found via
   Bayesian optimization (Optuna) with time-series K-Fold cross-validation.

---

## Architecture

```
Deribit API ──┐
              ├──► src/collector/ ──► data/raw/*.parquet
yfinance VIX ─┘
                                          │
                                          ▼
                              src/hmm/features.py  ◄── BTC, ETH, VIX series
                                          │
                                          ▼
                              src/hmm/optimizer.py  (Optuna + TimeSeriesSplit)
                                          │
                                          ▼
                              src/hmm/model.py  (GaussianHMM)
                                          │
                                          ▼
                              src/hmm/predict.py  ──► regime labels + forecasts
```

### Module Breakdown

| Module | Path | Responsibility |
|---|---|---|
| collector | `src/collector/` | Deribit client, VIX fetch, Parquet storage, APScheduler |
| hmm | `src/hmm/` | Feature engineering, GaussianHMM, Bayesian optimization, prediction |

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
# config/settings.yaml (example — actual file is the authoritative reference)

symbols:
  deribit: [BTC, ETH, SOL]       # instruments to collect
  vix: "^VIX"                    # yfinance ticker

collector:
  resolution: 3600               # candle interval in seconds (1h)
  history_days: 365              # initial backfill window
  schedule_interval: 3600        # APScheduler interval in seconds

storage:
  raw_dir: data/raw              # Parquet output directory

hmm:
  n_components: [2, 3, 4]        # regime count search space
  n_splits: 5                    # TimeSeriesSplit folds
  n_trials: 100                  # Optuna trials

logging:
  level: INFO
```

---

## Usage

**Run the collector once (backfill + schedule):**
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
