"""GaussianHMM wrapper for Solana regime detection.

Design
------
Strategy pattern: HMMStrategy defines the interface; GaussianHMMModel
implements it using hmmlearn's GaussianHMM.

Factory: build_model() constructs a configured instance from settings.yaml.

Model selection: select_n_components() iterates over the configured search
space and returns the model with the lowest BIC (penalises complexity without
under-fitting — lower BIC is better).

Persistence: save() / load() via pickle so fitted models survive between runs.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy base
# ─────────────────────────────────────────────────────────────────────────────


class HMMStrategy(ABC):
    """Abstract base for all HMM variants."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "HMMStrategy":
        """Fit on observation matrix X (n_samples, n_features)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return most-likely regime label per observation (0-indexed)."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities (n_samples, n_components)."""

    @abstractmethod
    def score(self, X: np.ndarray) -> float:
        """Return total log-likelihood."""

    @abstractmethod
    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion — lower is better."""


# ─────────────────────────────────────────────────────────────────────────────
# Concrete strategy
# ─────────────────────────────────────────────────────────────────────────────


class GaussianHMMModel(HMMStrategy):
    """GaussianHMM with configurable covariance type.

    Thin wrapper around hmmlearn.hmm.GaussianHMM that adds BIC scoring,
    per-sample log-likelihood normalisation, and pickle persistence.
    """

    def __init__(
        self,
        n_components: int,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self._model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray) -> "GaussianHMMModel":
        self._model.fit(X)
        ll = self.score(X)
        logger.info(
            "GaussianHMM n_components=%d  log-likelihood=%.4f  BIC=%.2f",
            self.n_components,
            ll,
            self.bic(X),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)  # type: ignore[no-any-return]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities, shape (n_samples, n_components)."""
        return self._model.predict_proba(X)  # type: ignore[no-any-return]

    def score(self, X: np.ndarray) -> float:
        """Total log-likelihood (not per-sample)."""
        return float(self._model.score(X))

    def bic(self, X: np.ndarray) -> float:
        """BIC = −2·ℓ + k·ln(n).

        k = number of free parameters:
          (k−1) initial probabilities
          k·(k−1) transition probabilities
          k·d means
          covariance parameters depend on type
        """
        n_samples, n_features = X.shape
        k = self._n_free_params(n_features)
        return float(-2.0 * self.score(X) + k * np.log(n_samples))

    def _n_free_params(self, n_features: int) -> int:
        k = self.n_components
        d = n_features
        cov_params = {
            "full":     k * d * (d + 1) // 2,
            "diag":     k * d,
            "tied":     d * (d + 1) // 2,
            "spherical": k,
        }
        return (
            (k - 1)          # initial state distribution
            + k * (k - 1)    # transition matrix rows
            + k * d          # means
            + cov_params.get(self.covariance_type, k * d * (d + 1) // 2)
        )

    def regime_stats(self, X: np.ndarray) -> dict[int, dict[str, float]]:
        """Return frequency and mean log-likelihood contribution per regime."""
        labels = self.predict(X)
        stats: dict[int, dict[str, float]] = {}
        for r in range(self.n_components):
            mask = labels == r
            stats[r] = {
                "frequency": float(mask.mean()),
                "n_observations": int(mask.sum()),
            }
        return stats

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "GaussianHMMModel":
        with open(path, "rb") as f:
            model: GaussianHMMModel = pickle.load(f)
        logger.info("Model loaded ← %s", path)
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def build_model(config: dict, n_components: int) -> GaussianHMMModel:
    """Factory: construct a GaussianHMMModel from settings.yaml."""
    hmm_cfg = config.get("hmm", {})
    return GaussianHMMModel(
        n_components=n_components,
        covariance_type=hmm_cfg.get("covariance_type", "full"),
        n_iter=hmm_cfg.get("n_iter", 200),
        random_state=hmm_cfg.get("random_state", 42),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────────


def select_n_components(
    config: dict,
    X: np.ndarray,
) -> GaussianHMMModel:
    """Fit all configured n_components, return the model with lowest BIC.

    Uses the search space defined in config['hmm']['n_components'].
    All models are fitted on the full X — this is model selection, not
    cross-validation (CV is handled in optimizer.py).
    """
    candidates: list[int] = config["hmm"]["n_components"]
    results: list[tuple[float, GaussianHMMModel]] = []

    for k in candidates:
        model = build_model(config, n_components=k)
        model.fit(X)
        b = model.bic(X)
        logger.info("  n_components=%d  BIC=%.2f", k, b)
        results.append((b, model))

    best_bic, best_model = min(results, key=lambda t: t[0])
    logger.info(
        "Selected n_components=%d  BIC=%.2f",
        best_model.n_components,
        best_bic,
    )
    return best_model
