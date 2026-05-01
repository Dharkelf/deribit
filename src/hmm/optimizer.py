"""Bayesian feature selection for the HMM pipeline.

Design
------
Template Method: run_optimization() is the fixed algorithm skeleton.
Optuna TPESampler explores the binary feature-inclusion hyperspace plus
n_components; TimeSeriesSplit provides walk-forward CV.

Objective: minimise negative mean CV log-likelihood per sample.
Lower objective = better model (Optuna minimises by default).

Study caching: fitted Optuna study persisted as pickle next to model artefacts.
top_n_results() extracts the N best *distinct* feature-set + n_components combos.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.hmm.features import ALL_EXTRACTORS, ALL_FEATURE_NAMES, build_feature_matrix, load_common_dataframe
from src.hmm.model import build_model
from src.utils.paths import models_dir

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_STUDY_FILENAME = "optuna_study.pkl"

# SOL_log_return is always included — never toggled by Optuna
_OPTIONAL_FEATURES: list[str] = [f for f in ALL_FEATURE_NAMES if f != "SOL_log_return"]


# ─────────────────────────────────────────────────────────────────────────────
# Study persistence
# ─────────────────────────────────────────────────────────────────────────────


def save_study(study: optuna.Study, config: dict) -> Path:
    path = models_dir(config) / _STUDY_FILENAME
    with open(path, "wb") as f:
        pickle.dump(study, f)
    logger.info("Optuna study saved → %s", path)
    return path


def load_study(config: dict) -> optuna.Study | None:
    path = models_dir(config) / _STUDY_FILENAME
    if not path.exists():
        return None
    with open(path, "rb") as f:
        study: optuna.Study = pickle.load(f)
    logger.info("Optuna study loaded ← %s  (%d trials)", path, len(study.trials))
    return study


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────


def _viable_optional_features(df_common: pd.DataFrame) -> list[str]:
    """Return optional features with ≥50% non-NaN coverage in df_common.

    Runs all extractors once and checks NaN share per column. Features from
    sources not yet collected (e.g. Max Pain) would produce 100% NaN and are
    excluded so Optuna never wastes trials on them.
    """
    df = df_common.copy()
    for ext in ALL_EXTRACTORS:
        df = ext.transform(df)

    viable = [
        f for f in _OPTIONAL_FEATURES
        if f in df.columns and df[f].notna().mean() >= 0.5
    ]
    excluded = set(_OPTIONAL_FEATURES) - set(viable)
    if excluded:
        logger.info(
            "Excluded %d features with <50%% coverage: %s",
            len(excluded), sorted(excluded),
        )
    return viable


def _build_objective(
    config: dict,
    df_common: pd.DataFrame,
) -> Any:
    """Return a closure that Optuna can call as objective(trial)."""
    hmm_cfg = config["hmm"]
    n_components_choices: list[int] = hmm_cfg["n_components"]
    n_splits: int = hmm_cfg.get("n_splits", 5)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Compute once — exclude features without sufficient data
    viable = _viable_optional_features(df_common)
    logger.info("Optimizer: %d viable optional features", len(viable))

    def objective(trial: optuna.Trial) -> float:
        # --- Hyperparameters ---
        n_components: int = trial.suggest_categorical(
            "n_components", n_components_choices
        )
        feature_subset: list[str] = [
            f
            for f in viable
            if trial.suggest_categorical(f"use_{f}", [True, False])
        ]

        try:
            X_full = build_feature_matrix(df_common.copy(), feature_subset)
        except ValueError as e:
            logger.debug("Feature build failed: %s", e)
            raise optuna.exceptions.TrialPruned()

        if len(X_full) < n_components * 20:
            raise optuna.exceptions.TrialPruned()

        # --- Walk-forward CV ---
        scores: list[float] = []
        for train_idx, val_idx in tscv.split(X_full):
            X_train = X_full.iloc[train_idx].values
            X_val = X_full.iloc[val_idx].values

            if len(X_train) < n_components * 10 or len(X_val) < 1:
                continue

            try:
                model = build_model(config, n_components=n_components)
                model.fit(X_train)
                ll_per_sample = model.score(X_val) / len(X_val)
                # Discard degenerate fits (covariance collapse, near-singular matrix)
                if np.isfinite(ll_per_sample) and ll_per_sample > -1e4:
                    scores.append(ll_per_sample)
            except Exception as e:
                logger.debug("CV fold failed (k=%d): %s", n_components, e)
                continue

        if not scores:
            raise optuna.exceptions.TrialPruned()

        return -float(np.mean(scores))  # minimise negative LL

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_optimization(
    config: dict,
    *,
    force: bool = False,
) -> optuna.Study:
    """Run Optuna study; return cached study if available and not *force*.

    Loads the common DataFrame internally so callers need only pass config.
    """
    if not force:
        cached = load_study(config)
        if cached is not None:
            return cached

    logger.info("Loading common DataFrame for feature optimisation …")
    df_common = load_common_dataframe(config)

    hmm_cfg = config["hmm"]
    n_trials: int = hmm_cfg.get("n_trials", 100)
    random_state: int = hmm_cfg.get("random_state", 42)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    objective = _build_objective(config, df_common)
    logger.info("Starting Optuna optimisation: %d trials …", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Best trial: objective=%.4f  n_components=%d  n_features=%d",
        study.best_trial.value,
        study.best_trial.params["n_components"],
        sum(1 for k, v in study.best_trial.params.items() if k.startswith("use_") and v),
    )

    save_study(study, config)
    return study


# ─────────────────────────────────────────────────────────────────────────────
# Result extraction
# ─────────────────────────────────────────────────────────────────────────────


def _params_to_feature_subset(params: dict[str, Any]) -> list[str]:
    return [f for f in _OPTIONAL_FEATURES if params.get(f"use_{f}", False)]


def top_n_results(
    study: optuna.Study,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Return the N best completed trials as structured dicts.

    Each dict has:
      n_components : int
      feature_subset: list[str]   # optional features (SOL_log_return excluded)
      objective     : float        # negative mean CV LL (lower = better)
      trial_number  : int
    """
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    completed.sort(key=lambda t: t.value)

    results: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    for trial in completed:
        k = trial.params["n_components"]
        subset = _params_to_feature_subset(trial.params)
        key = (k, tuple(sorted(subset)))
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "n_components": k,
                "feature_subset": subset,
                "objective": trial.value,
                "trial_number": trial.number,
            }
        )
        if len(results) >= n:
            break

    return results
