"""Bayesian feature selection for the HMM pipeline.

Design
------
Template Method: run_optimization() is the fixed algorithm skeleton.
Optuna TPESampler explores the binary feature-inclusion hyperspace plus
n_components; TimeSeriesSplit provides walk-forward CV.

Objective (composite selection score, higher = better regime structure):
  score = 3.0·avg_self_transition
        + 1.5·min_state_fraction
        − 0.25·median_run_days          (run length in days = hours/24)
        − 2.5·avg_entropy
        + 0.05·loglik_per_obs_per_feat

  Eligibility: model is only scored when min_state_fraction ≥ 0.05 in every
  accepted fold — models that collapse a state are pruned.

  Optuna minimises the NEGATIVE mean score (→ maximises the score).

  Ref: blueprint from course notes (adapted for hourly data — run_length/24).

Study caching: fitted Optuna study persisted as pickle next to model artefacts.
top_n_results() extracts the N best *distinct* feature-set + n_components combos.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.hmm.features import ALL_EXTRACTORS, ALL_FEATURE_NAMES, build_feature_matrix, load_common_dataframe
from src.hmm.model import GaussianHMMModel, build_model
from src.utils.paths import models_dir

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_STUDY_FILENAME        = "optuna_study.pkl"
_BEST_FEATURES_FILENAME = "best_features.json"

# Features excluded from HMM candidate set but still available for XGB/NP:
# - Max Pain: options-expiry sparse data, unavailable for historical periods → too many NaN rows
#   when the training window is extended to 4+ years
# - Stock F&G: CNN API hard cap ~253 trading days → ~99% NaN on multi-year windows
_HMM_EXCLUDED_FEATURES: frozenset[str] = frozenset({
    "max_pain_ratio", "max_pain_diff_pct",
    "max_pain_7d_ratio", "max_pain_7d_diff_pct",
    "stock_fear_greed",
})

# SOL_log_return is always included — never toggled by Optuna
_OPTIONAL_FEATURES: list[str] = [
    f for f in ALL_FEATURE_NAMES
    if f != "SOL_log_return" and f not in _HMM_EXCLUDED_FEATURES
]

# Lowered from 0.05 → 0.02 to support k=5 without excessive pruning.
# Each state still needs ≥2% of training rows (≈170 rows on 8500-row dataset).
_MIN_STATE_FRACTION = 0.02


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
    try:
        with open(path, "rb") as f:
            study: optuna.Study = pickle.load(f)
    except Exception as exc:
        logger.warning("Optuna study corrupt or incompatible (%s) — will re-run", exc)
        return None
    logger.info("Optuna study loaded ← %s  (%d trials)", path, len(study.trials))
    return study


def save_best_features(study: optuna.Study, config: dict) -> Path:
    """Persist best feature config to JSON for downstream models (XGB, Prophet)."""
    best = top_n_results(study, n=1)[0]
    path = models_dir(config) / _BEST_FEATURES_FILENAME
    payload = {
        "n_components":   best["n_components"],
        "feature_subset": best["feature_subset"],
        "score":          best["score"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "Best features saved → %s  (k=%d, %d optional features, score=%.4f)",
        path, payload["n_components"], len(payload["feature_subset"]), payload["score"],
    )
    return path


def load_best_features(config: dict) -> dict | None:
    path = models_dir(config) / _BEST_FEATURES_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        payload: dict = json.load(f)
    logger.info(
        "Best features loaded ← %s  (k=%d, %d optional features)",
        path, payload["n_components"], len(payload["feature_subset"]),
    )
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Selection-score helpers
# ─────────────────────────────────────────────────────────────────────────────


def _median_run_length(labels: np.ndarray) -> float:
    """Median length (in steps) of contiguous same-regime blocks."""
    if len(labels) == 0:
        return 0.0
    runs: list[int] = []
    count = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            count += 1
        else:
            runs.append(count)
            count = 1
    runs.append(count)
    return float(np.median(runs))


def _selection_score(
    model: GaussianHMMModel,
    X_train: np.ndarray,
) -> float | None:
    """Compute the composite regime-quality score on the training fold.

    Returns None when the model is ineligible (a state occupies < 5% of
    training observations).

    Score components
    ----------------
    avg_self_transition  : mean diagonal of the transition matrix — rewards
                           stable, persistent regimes.
    min_state_fraction   : fraction of the rarest state — penalises degenerate
                           models where a regime is nearly unused.
    median_run_days      : median contiguous-block length in days (hours/24) —
                           mild penalty for excessively short or erratic runs.
    avg_entropy          : mean posterior entropy per step — penalises uncertain
                           regime assignments (lower entropy = crisper regimes).
    loglik_per_obs_feat  : train log-likelihood / (T·d) — rewards fit quality.

    Weights from blueprint (adapted: run_length divided by 24 for hourly data):
      score = 3.0·avg_self_transition + 1.5·min_state_fraction
            − 0.25·median_run_days − 2.5·avg_entropy
            + 0.05·loglik_per_obs_feat
    """
    k = model.n_components
    labels = model.predict(X_train)

    # ── State fractions ────────────────────────────────────────────────────
    fractions = np.array([(labels == r).mean() for r in range(k)])
    min_state_fraction = float(fractions.min())
    if min_state_fraction < _MIN_STATE_FRACTION:
        return None                              # ineligible

    # ── avg self-transition ────────────────────────────────────────────────
    avg_self_transition = float(np.diag(model._model.transmat_).mean())

    # ── median run length (hours → days) ──────────────────────────────────
    median_run_days = _median_run_length(labels) / 24.0

    # ── average posterior entropy ──────────────────────────────────────────
    proba = model.predict_proba(X_train)                 # (T, k)
    entropy = -np.sum(proba * np.log(proba + 1e-9), axis=1)
    avg_entropy = float(entropy.mean())

    # ── normalised log-likelihood ──────────────────────────────────────────
    T, d = X_train.shape
    loglik_per_obs_feat = model.score(X_train) / (T * d)

    score = (
        3.00 * avg_self_transition
        + 1.50 * min_state_fraction
        - 0.25 * median_run_days
        - 2.50 * avg_entropy
        + 0.05 * loglik_per_obs_feat
    )
    return float(score)


# ─────────────────────────────────────────────────────────────────────────────
# Feature coverage filter
# ─────────────────────────────────────────────────────────────────────────────


def _viable_optional_features(df_common: pd.DataFrame) -> list[str]:
    """Return optional features with sufficient non-NaN coverage in df_common.

    General rule: ≥50% non-NaN rows required.
    Exception: features in _SPARSE_OK_FEATURES (Max Pain) only need ≥1 non-NaN
    row — they are options-expiry-based and naturally sparse.
    """
    df = df_common.copy()
    for ext in ALL_EXTRACTORS:
        df = ext.transform(df)

    def _ok(f: str) -> bool:
        if f not in df.columns:
            return False
        return df[f].notna().mean() >= 0.5

    viable = [f for f in _OPTIONAL_FEATURES if _ok(f)]
    excluded = set(_OPTIONAL_FEATURES) - set(viable)
    if excluded:
        logger.info(
            "Excluded %d features with insufficient coverage: %s",
            len(excluded), sorted(excluded),
        )
    return viable


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────


def _build_objective(
    config: dict,
    df_common: pd.DataFrame,
) -> Any:
    """Return a closure that Optuna can call as objective(trial)."""
    hmm_cfg = config["hmm"]
    n_components_choices: list[int] = hmm_cfg["n_components"]
    n_splits: int = hmm_cfg.get("n_splits", 5)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    viable = _viable_optional_features(df_common)
    logger.info("Optimizer: %d viable optional features", len(viable))

    def objective(trial: optuna.Trial) -> float:
        # ── Hyperparameters ────────────────────────────────────────────────
        n_components: int = trial.suggest_categorical(
            "n_components", n_components_choices
        )
        feature_subset: list[str] = [
            f for f in viable
            if trial.suggest_categorical(f"use_{f}", [True, False])
        ]

        try:
            X_full = build_feature_matrix(df_common.copy(), feature_subset)
        except ValueError as e:
            logger.debug("Feature build failed: %s", e)
            raise optuna.exceptions.TrialPruned()

        if len(X_full) < n_components * 20:
            raise optuna.exceptions.TrialPruned()

        # ── Walk-forward CV (score on training folds) ──────────────────────
        scores: list[float] = []
        for train_idx, _ in tscv.split(X_full):
            X_train = X_full.iloc[train_idx].values
            if len(X_train) < n_components * 10:
                continue

            try:
                model = build_model(config, n_components=n_components)
                model.fit(X_train)
                s = _selection_score(model, X_train)
                if s is not None:
                    scores.append(s)
            except Exception as e:
                logger.debug("CV fold failed (k=%d): %s", n_components, e)
                continue

        if not scores:
            raise optuna.exceptions.TrialPruned()

        return -float(np.mean(scores))      # Optuna minimises → maximise score

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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        "Best trial: score=%.4f  n_components=%d  n_features=%d",
        -study.best_trial.value,
        study.best_trial.params["n_components"],
        sum(1 for k, v in study.best_trial.params.items() if k.startswith("use_") and v),
    )

    save_study(study, config)
    save_best_features(study, config)
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
      n_components  : int
      feature_subset: list[str]   # optional features (SOL_log_return excluded)
      objective     : float        # Optuna objective (negative score, lower = better)
      score         : float        # selection score (higher = better)
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
                "n_components":  k,
                "feature_subset": subset,
                "objective":     trial.value,
                "score":         -trial.value,   # higher = better
                "trial_number":  trial.number,
            }
        )
        if len(results) >= n:
            break

    return results
