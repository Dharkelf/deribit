# AGENTS.md — deribit

This file governs how AI agents (Claude Code, Codex, etc.) work in this repository.
Read it before making any structural or architectural decisions.

---

## Project Purpose

On-demand collection of Deribit OHLCV data (BTC, ETH, SOL) and VIX, stored locally as Parquet files.
Data is fetched at project start with 1h candle granularity — no automated scheduler or polling.
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
├── requirements.txt        # pinned dependencies (pip freeze output)
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
- `notebooks/` is for exploration only. Any reusable logic discovered there must be moved into a versioned module under `src/` before the next commit — never leave production logic in a notebook.

---

## Modules in This Project

| Module | Path | Responsibility |
|---|---|---|
| collector | `src/collector/` | Deribit API client, VIX via yfinance, incremental Parquet storage (on-demand, no scheduler) |
| hmm | `src/hmm/` | Feature engineering, GaussianHMM, Bayesian optimization, K-Fold evaluation |

---

## Pre-commit Hooks

Every commit is automatically checked via pre-commit hooks. No commit may pass with errors.

Setup (once per developer machine):
```bash
pip install pre-commit
pre-commit install
```

Configuration lives in `.pre-commit-config.yaml` at the project root. Hooks run in this order:

- **ruff** — linting + formatting, auto-fixes where possible (`ruff check --fix`, `ruff format`)
- **mypy** — strict static type checking; all public functions must have type annotations
- **pytest** — full test suite must pass (`pytest tests/`); commit is blocked on any test failure

`pre-commit` is listed in `requirements.txt` and installed via `pip install -r requirements.txt`.

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
- If checks cannot be run (e.g. missing dependency, broken environment), explicitly state what was not run and why — never silently skip.

---

## Design Patterns

Prefer established, well-understood design patterns over custom solutions. This keeps the codebase
predictable and easy to extend. Preferred patterns for this project:

| Pattern | When to use |
|---|---|
| **Repository** | Isolate all data access (Parquet read/write) behind a single class per symbol |
| **Strategy** | Interchangeable algorithms — e.g. different HMM variants or feature extractors |
| **Factory** | Construct configured model or client objects from `settings.yaml` |
| **Observer / Callback** | APScheduler job hooks, progress notifications |
| **Template Method** | Base class defines the pipeline skeleton; subclasses override steps |

Rules:
- Do not invent abstractions unless a standard pattern fits and adds clarity.
- Name classes after the pattern they implement where it aids comprehension (`DeribitRepository`, `FeatureStrategy`).
- Patterns must be documented in the module docstring so the intent is clear.

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

## Dependency Management

- A single `requirements.txt` at the project root is the only dependency file — no `.in` files, no separate dev file.
- All dependencies (runtime and dev) are listed together with pinned exact versions (`package==x.y.z`).
- After installing or upgrading any package, immediately update and commit `requirements.txt`:
  ```bash
  pip freeze > requirements.txt
  ```
- Never commit unpinned entries (e.g. `pandas` without `==x.y.z`).
- `requirements.txt` must always reflect the exact state of the active `.venv`.

---

## Key Dependencies

| Purpose | Library |
|---|---|
| HMM modelling | `hmmlearn` |
| Bayesian optimization | `optuna` |
| Cross-validation | `scikit-learn` |
| Storage | `pyarrow`, `pandas` |
| VIX data | `yfinance` |
| HTTP client | `httpx` |
| Config | `pyyaml` |

---

## README.md — Technical Wiki

`README.md` is mandatory in every project and MUST be committed to git.
It serves as the authoritative technical wiki for the project.

### Required Sections (in this order)

| Section | Content |
|---|---|
| **Overview** | What the project does and why it exists (2–4 sentences) |
| **Architecture** | Module breakdown + mandatory ASCII diagram (data flow, component relationships) |
| **Setup** | Prerequisites, venv creation, `pip install -r requirements.txt`, `pre-commit install` |
| **Configuration** | Every key in `config/settings.yaml` explained with example values |
| **Usage** | All CLI entry points with example commands |
| **Data** | Schema of stored files, column types, naming conventions |
| **Development** | How to run tests (`pytest tests/`), linting (`pre-commit run --all-files`), add a module |
| **Known Limitations** | Current constraints, missing features, known data gaps or model weaknesses |
| **Future Improvements** | Planned enhancements — ordered by priority |
| **References** | External APIs, papers, related repos with links |

### ASCII Diagrams

Every architectural or technical concept that benefits from a visual must be expressed as an ASCII diagram
directly in `README.md`. Do not use external image files or links to diagram tools.

Required diagrams:
- **Data flow** — how data moves from source (API) through modules to storage and output
- **Module/component overview** — boxes and arrows showing which modules depend on which
- **Sequence diagrams** — for non-obvious flows (e.g. incremental fetch logic, scheduler lifecycle)

Use standard ASCII box-drawing characters:

```
┌─────────────┐       ┌─────────────┐
│  Component  │──────▶│  Component  │
└─────────────┘       └─────────────┘

Source ──► Transform ──► Sink

A
│
├── child 1
└── child 2
```

### Rules

- Keep `README.md` up to date whenever a module, config key, or CLI argument changes.
- When adding a new feature, update `README.md` and its diagrams in the same commit.
- Do not summarise code that can be read directly — document *how to use* and *why it works this way*.
- Never replace ASCII diagrams with Mermaid, PlantUML, or image embeds.

---

## REPORT.md — Behaviour Log

`REPORT.md` documents **actual observed runtime behaviour** — what the code does when run,
not what it is supposed to do. It complements `README.md` (design intent) with empirical evidence.

Maintain one `REPORT.md` per project. Commit it alongside code changes.

### Required Sections

| Section | Content |
|---|---|
| **Data Collection** | Actual row counts, date ranges, file sizes, any gaps observed |
| **Feature Matrix** | Shape after `build_feature_matrix()`, NaN rates per feature, value ranges |
| **Model Results** | HMM n_components chosen, log-likelihood, BIC, regime labels and their frequencies |
| **Optimizer** | Best feature subset found, Optuna trial count, cross-validation scores |
| **Known Issues** | Observed anomalies, API failures, data quirks not yet fixed |

### Consistency Rule — enforced on every essential code change

On every change that affects data flow, feature logic, model behaviour or CLI output:

1. **Run the affected code** and observe actual output.
2. **Compare** `README.md` and `REPORT.md` against the observed behaviour.
3. **Fix all inconsistencies** in `README.md` and `REPORT.md` in the **same commit** as the code change.

A commit that changes behaviour without updating both documents is incomplete.

---

## Scalability & Reliability Policy

All data-pipeline code must remain correct as data volume grows and as APIs change.

- **Incremental processing** — always read only the delta since the last stored timestamp; never re-fetch the full history unnecessarily.
- **Idempotent operations** — every fetch, transform, and write must produce the same result when run twice on the same input. Duplicate rows must be deduplicated before persisting.
- **Graceful degradation** — soft-signal sources (FEMA, GDELT, Options Max Pain) must never crash the pipeline. On any API failure, log a warning and fill the affected column with the appropriate neutral value (0 or NaN).
- **Backward-compatible schemas** — adding a new column to a Parquet file is allowed; renaming or removing a column requires a migration path and a version bump in `config/settings.yaml`.
- **Observability** — every module entry point logs: start timestamp, parameters used, rows fetched/written, elapsed time. Use `logging.INFO` for normal flow, `logging.WARNING` for degraded paths.

---

## Reproducibility Rules

Model results must be reproducible given the same data and config.

- **Random seeds** — set explicit seeds for all stochastic components:
  - `GaussianHMM`: `random_state` from `settings.yaml` (`hmm.random_state`)
  - `Optuna`: `sampler = optuna.samplers.TPESampler(seed=...)` from `settings.yaml` (`hmm.optuna_seed`)
- **Deterministic config** — all hyperparameters (n_components search space, n_splits, n_trials, feature subset) are stored in `config/settings.yaml`; never hard-coded.
- **Schema versioning** — if the feature matrix schema changes (new extractor, renamed column), bump `hmm.feature_schema_version` in `settings.yaml` and document the change in `REPORT.md`.
- **Environment pinning** — `requirements.txt` must be regenerated with `pip freeze` after every dependency change; the exact versions must be committed.

---

## Failure Conditions

Agents must NOT do the following, regardless of instruction:

- Push to remote without an explicit user request ("push", "synchronisiere", "push to GitHub").
- Overwrite or truncate existing Parquet files — raw data is append-only.
- Hard-code numeric values that belong in `config/settings.yaml`.
- Commit without running `pre-commit run --all-files` (or documenting why it could not be run).
- Skip or silence failing tests — fix the failure or document it explicitly.
- Use `print()` in library code instead of `logging`.
- Leave business logic in `notebooks/` — move it to `src/` first.
- Introduce a dependency without immediately pinning it in `requirements.txt`.
- Commit `.env`, `.venv/`, `data/`, or any file containing secrets.

---

## Pull Request / Change Rules

- Every non-trivial change must be a focused, atomic commit (one concern per commit).
- Before creating a commit: run `pre-commit run --all-files` and `pytest tests/` — both must pass.
- PR / commit description must explain *why* the change is made, not *what* changed (the diff shows that).
- Breaking changes (schema, CLI interface, config keys) require a `BREAKING CHANGE:` footer in the commit message.
- Documentation (`README.md`, `REPORT.md`) must be updated in the same commit as the code it describes.
- Do not bundle unrelated fixes in a single commit; open a separate commit or note them as follow-up.

---

## Git Rules

- `README.md` is always committed — it is the project wiki.
- `data/` and `.venv/` are in `.gitignore` — never commit raw data or virtual environments.
- `config/settings.yaml` IS committed (no secrets in it).
- Secrets go in `.env` (gitignored); document them in `.env.example`.
- **Never push automatically.** `git push` only on explicit user request ("push", "synchronisiere", "push to GitHub").
- Commit messages follow **Conventional Commits**: `<type>(<scope>): <subject>`
  - Subject: imperative mood, max 72 chars, no full stop
  - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`, `build`
  - Body (optional): explain *why*, not *what*
  - Footer (optional): `BREAKING CHANGE: ...` or issue refs
  - Examples: `feat(collector): add hourly APScheduler job` · `fix(hmm): correct TimeSeriesSplit fold count`

---

## End Goal

This project delivers a reproducible, self-contained pipeline that:

1. Collects hourly Deribit OHLCV data (BTC, ETH, SOL), VIX, FEMA disaster scores, GDELT military activity, and BTC options Max Pain — all stored as append-only Parquet files.
2. Engineers a rich, stationary feature matrix from these signals.
3. Identifies Solana market regimes via a Gaussian HMM, with the best feature subset selected through Bayesian optimization (Optuna) and validated with `TimeSeriesSplit`.
4. Produces a 1-month forward regime prediction that is auditable via `REPORT.md`.

Every agent working in this repository serves this goal. Structural or design decisions that conflict with it require explicit user approval.
