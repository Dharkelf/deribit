"""Feature engineering for the HMM pipeline.

Strategy pattern: each FeatureExtractor adds one or more named columns
to the common DataFrame. The Bayesian optimizer selects the best subset
by name. All features are derived from BTC, ETH and VIX; SOL close/return
is the HMM observation series.

Common DataFrame layout (inner join on UTC timestamp index):
    BTC_close, ETH_close, SOL_close, VIX_close  — raw close prices
    + all engineered features added by extractors
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import raw_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbols and windows
# ---------------------------------------------------------------------------

_SYMBOLS = ["BTC", "ETH", "SOL", "VIX"]
_SHORT_WINDOW = 24    # 1 day in hours
_LONG_WINDOW = 168    # 1 week in hours


# ---------------------------------------------------------------------------
# Common DataFrame loader
# ---------------------------------------------------------------------------

def load_common_dataframe(config: dict) -> pd.DataFrame:
    """Load all four Parquet files and inner-join on UTC timestamp.

    Returns a DataFrame with columns:
        <SYMBOL>_open/high/low/close/volume  for each symbol
    Only rows present in all four series are kept (inner join).
    """
    rd = raw_dir(config)
    frames: dict[str, pd.DataFrame] = {}

    for symbol in _SYMBOLS:
        path: Path = rd / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Parquet file for {symbol} at {path}. "
                "Run 'python main.py collect' first."
            )
        df = pd.read_parquet(path)
        df.columns = [f"{symbol}_{col}" for col in df.columns]
        frames[symbol] = df

    combined = pd.concat(frames.values(), axis=1, join="inner")
    combined.index.name = "timestamp"
    logger.info(
        "Common DataFrame: %d rows, %s → %s",
        len(combined),
        combined.index.min().date(),
        combined.index.max().date(),
    )
    return combined


# ---------------------------------------------------------------------------
# Strategy: FeatureExtractor base class
# ---------------------------------------------------------------------------

class FeatureExtractor(ABC):
    """Base strategy for adding features to the common DataFrame."""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Names of columns this extractor produces."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature columns to *df* and return it."""


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class LogReturnExtractor(FeatureExtractor):
    """1-hour log returns for BTC, ETH, SOL, VIX."""

    @property
    def feature_names(self) -> list[str]:
        return [f"{s}_log_return" for s in _SYMBOLS]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for sym in _SYMBOLS:
            df[f"{sym}_log_return"] = np.log(
                df[f"{sym}_close"] / df[f"{sym}_close"].shift(1)
            )
        return df


class RollingVolatilityExtractor(FeatureExtractor):
    """Rolling std of log returns over short (24h) and long (168h) windows."""

    @property
    def feature_names(self) -> list[str]:
        names = []
        for sym in ["BTC", "ETH", "SOL", "VIX"]:
            names += [f"{sym}_vol_{_SHORT_WINDOW}h", f"{sym}_vol_{_LONG_WINDOW}h"]
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for sym in ["BTC", "ETH", "SOL", "VIX"]:
            ret = np.log(df[f"{sym}_close"] / df[f"{sym}_close"].shift(1))
            df[f"{sym}_vol_{_SHORT_WINDOW}h"] = ret.rolling(_SHORT_WINDOW).std()
            df[f"{sym}_vol_{_LONG_WINDOW}h"] = ret.rolling(_LONG_WINDOW).std()
        return df


class RollingCorrelationExtractor(FeatureExtractor):
    """Rolling Pearson correlation of SOL log return vs BTC and ETH."""

    @property
    def feature_names(self) -> list[str]:
        return [
            f"SOL_BTC_corr_{_SHORT_WINDOW}h",
            f"SOL_BTC_corr_{_LONG_WINDOW}h",
            f"SOL_ETH_corr_{_SHORT_WINDOW}h",
            f"SOL_ETH_corr_{_LONG_WINDOW}h",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        sol_ret = np.log(df["SOL_close"] / df["SOL_close"].shift(1))
        for sym in ["BTC", "ETH"]:
            sym_ret = np.log(df[f"{sym}_close"] / df[f"{sym}_close"].shift(1))
            df[f"SOL_{sym}_corr_{_SHORT_WINDOW}h"] = sol_ret.rolling(_SHORT_WINDOW).corr(sym_ret)
            df[f"SOL_{sym}_corr_{_LONG_WINDOW}h"] = sol_ret.rolling(_LONG_WINDOW).corr(sym_ret)
        return df


class VixLevelExtractor(FeatureExtractor):
    """Normalised VIX level and 24h VIX change — proxy for global risk appetite."""

    @property
    def feature_names(self) -> list[str]:
        return ["VIX_zscore", "VIX_change_24h"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        vix = df["VIX_close"]
        df["VIX_zscore"] = (vix - vix.rolling(_LONG_WINDOW).mean()) / (
            vix.rolling(_LONG_WINDOW).std() + 1e-9
        )
        df["VIX_change_24h"] = vix.diff(_SHORT_WINDOW)
        return df


class MomentumExtractor(FeatureExtractor):
    """Price momentum: ratio of short-window mean to long-window mean (BTC, ETH, SOL)."""

    @property
    def feature_names(self) -> list[str]:
        return [f"{s}_momentum" for s in ["BTC", "ETH", "SOL"]]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for sym in ["BTC", "ETH", "SOL"]:
            close = df[f"{sym}_close"]
            df[f"{sym}_momentum"] = (
                close.rolling(_SHORT_WINDOW).mean()
                / close.rolling(_LONG_WINDOW).mean()
            ) - 1
        return df


# ---------------------------------------------------------------------------
# Registry and builder
# ---------------------------------------------------------------------------

ALL_EXTRACTORS: list[FeatureExtractor] = [
    LogReturnExtractor(),
    RollingVolatilityExtractor(),
    RollingCorrelationExtractor(),
    VixLevelExtractor(),
    MomentumExtractor(),
]

ALL_FEATURE_NAMES: list[str] = [
    name for ext in ALL_EXTRACTORS for name in ext.feature_names
]


def build_feature_matrix(
    df: pd.DataFrame,
    feature_subset: list[str],
) -> pd.DataFrame:
    """Apply all extractors, select *feature_subset* columns, drop NaN rows.

    Returns a clean matrix ready for HMM fitting.
    The SOL_log_return column is always included as the HMM observation.
    """
    for extractor in ALL_EXTRACTORS:
        df = extractor.transform(df)

    # SOL_log_return is the primary HMM observation — always present
    required = ["SOL_log_return"]
    cols = required + [f for f in feature_subset if f != "SOL_log_return"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown feature(s): {missing}. Available: {ALL_FEATURE_NAMES}")

    return df[cols].dropna()
