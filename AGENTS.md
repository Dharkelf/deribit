# AGENTS.md — deribit

This file governs how AI agents (Claude Code, Codex, etc.) work in this repository.
Read it before making any structural or architectural decisions.

---

## Project Purpose

Automated collection of Deribit spot data (BTC, ETH, SOL) and VIX, stored locally as Parquet files.
Regime-change detection and prediction for Solana via Hidden Markov Models,
with Bayesian feature-subset optimization (Optuna) and time-series cross-validation.

---

## Standard Directory Layout

All Python projects in this workspace follow this canonical structure.
Do NOT deviate from it without explicit user instruction.

```
<project-root>/
├── AGENTS.md               # this file — agent instructions
├── CLAUDE.md               # optional project-level Claude overrides
├── README.md
├── requirements.txt        # pinned dependencies (pip-compile output)
├── requirements-dev.txt    # dev/test dependencies
├── .gitignore
├── .env.example            # env-var template, never commit .env
│
├── config/
│   └── settings.yaml       # all runtime config (symbols, paths, schedules)
│
├── data/                   # excluded from git (see .gitignore)
│   ├── raw/                # immutable source data — never modify in-place
│   └── processed/          # derived/feature data
│
├── src/
│   ├── __init__.py
│   └── <module>/           # one subdirectory per logical module
│       ├── __init__.py
│       └── *.py
│
├── tests/
│   ├── conftest.py
│   └── test_<module>.py
│
├── notebooks/              # exploration only — no production logic here
│
└── main.py                 # CLI entry point
```

### Module Layout Rules

- Each functional domain lives in its own subdirectory under `src/`.
- Module names are lowercase, underscore-separated (`hmm_analysis`, not `HMMAnalysis`).
- No business logic in `main.py` — it only wires modules together and calls `run()`.
- `config/settings.yaml` is the single source of truth for all tunable parameters.
  Hard-coded values in source files are not allowed.

---

## Modules in This Project

| Module | Path | Responsibility |
|---|---|---|
| collector | `src/collector/` | Deribit API client, VIX via yfinance, Parquet storage, hourly scheduler |
| hmm | `src/hmm/` | Feature engineering, GaussianHMM, Bayesian optimization, K-Fold evaluation |

---

## Pre-commit Hooks

Every commit is automatically checked by `ruff` (linting + formatting) and `mypy` (static type checking)
via pre-commit hooks. No commit may pass with ruff errors or mypy type violations.

Setup (once per developer machine):
```bash
pip install pre-commit
pre-commit install
```

Configuration lives in `.pre-commit-config.yaml` at the project root.
Add `pre-commit` to `requirements-dev.txt`.

- **ruff** — replaces flake8, isort, pyupgrade; auto-fixes where possible (`ruff check --fix`, `ruff format`)
- **mypy** — strict mode; all public functions must have type annotations

To run manually against all files:
```bash
pre-commit run --all-files
```

---

## Testing

Every significant code change must be covered by tests before it is committed.

- **Unit tests** — test individual functions and classes in isolation; mock external dependencies (API calls, file I/O).
- **Integration tests** — test module interactions end-to-end; use real Parquet files and real data structures, not mocks.
- All tests live in `tests/` and follow the naming convention `test_<module>.py`.
- Run the full test suite before every commit: `pytest tests/`
- A change is considered covered when both the happy path and the main failure modes are tested.
- Time-series models: integration tests must use `TimeSeriesSplit` — never shuffle data in tests.

---

## Coding Conventions

- **Python 3.11+**
- Type hints on all public functions and class methods.
- No comments explaining *what* the code does — only *why* when non-obvious.
- No `print()` in library code — use the stdlib `logging` module; configure level in `settings.yaml`.
- All file I/O goes through path helpers in `src/utils/paths.py` (derive from `config/settings.yaml`).
- Parquet is the default storage format. CSV only for human-readable exports.
- Time-series cross-validation: always `sklearn.model_selection.TimeSeriesSplit` — never shuffle TS data.

---

## Data Conventions

- Raw data is **append-only**. Existing Parquet files are never overwritten; new data is appended.
- All timestamps are UTC, stored as `datetime64[ns, UTC]` in Parquet metadata.
- Symbol naming: `BTC`, `ETH`, `SOL`, `VIX` (uppercase, no suffixes in column names).

---

## External References

- Deribit public API: `https://docs.deribit.com` — endpoint `/public/get_tradingview_chart_data`
- Regime-change blueprint: `https://github.com/sergejschweizer/regimechangedetection`
- VIX source: `yfinance` ticker `^VIX`

---

## Key Dependencies

| Purpose | Library |
|---|---|
| HMM modelling | `hmmlearn` |
| Bayesian optimization | `optuna` |
| Cross-validation | `scikit-learn` |
| Scheduling | `APScheduler` |
| Storage | `pyarrow`, `pandas` |
| VIX data | `yfinance` |
| HTTP client | `httpx` |
| Config | `pyyaml`, `python-dotenv` |

---

## README.md — Technical Wiki

`README.md` is mandatory in every project and MUST be committed to git.
It serves as the authoritative technical wiki for the project.

### Required Sections

| Section | Content |
|---|---|
| **Overview** | What the project does and why it exists |
| **Architecture** | Module breakdown, data flow diagram (ASCII or Mermaid) |
| **Setup** | Prerequisites, venv creation, `pip install -r requirements.txt` |
| **Configuration** | All keys in `config/settings.yaml` explained |
| **Usage** | How to run the collector, the HMM pipeline, the scheduler |
| **Data** | Schema of stored Parquet files, symbol conventions |
| **Development** | How to run tests, linting, adding a new module |
| **References** | External APIs, papers, related repos |

### Rules

- Keep `README.md` up to date whenever a module, config key, or CLI argument changes.
- When adding a new feature, update `README.md` in the same commit.
- Do not summarise code that can be read directly — document *how to use* and *why it works this way*.

---

## Git Rules

- `README.md` is always committed — it is the project wiki.
- `data/` and `.venv/` are in `.gitignore` — never commit raw data or virtual environments.
- `config/settings.yaml` IS committed (no secrets in it).
- Secrets go in `.env` (gitignored); document them in `.env.example`.
- Commit messages follow **Conventional Commits**: `<type>(<scope>): <subject>`
  - Subject: imperative mood, max 72 chars, no full stop
  - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`
  - Body (optional): explain *why*, not *what*
  - Footer (optional): `BREAKING CHANGE: ...` or issue refs
  - Examples: `feat(collector): add hourly APScheduler job` · `fix(hmm): correct TimeSeriesSplit fold count`
