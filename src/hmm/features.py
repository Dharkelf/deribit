"""Feature engineering for the HMM pipeline.

Design
------
Strategy pattern: each FeatureExtractor adds named columns to the common
DataFrame. The Bayesian optimizer selects the best subset by name.

All price-based features use log-difference returns  log(P_t) − log(P_{t−1})
which makes the series stationary (removes price-level non-stationarity).

Common DataFrame
----------------
Built by load_common_dataframe(): inner-joins BTC/ETH/SOL/VIX on UTC timestamp,
then reindexes to a complete 24-hour hourly grid and forward-fills gaps
(VIX weekends, FEMA/GDELT daily data). FEMA and GDELT are joined as soft
[0, 1] float signals.

Available feature groups
-------------------------
  LogDiffReturnExtractor     — log-diff returns for all symbols
  RollingVolatilityExtractor — rolling std of log-diff returns (24h, 168h)
  RollingCorrelationExtractor— rolling Pearson correlation SOL↔BTC/ETH
  VixLevelExtractor          — VIX z-score + 24h change
  MomentumExtractor          — short/long MA ratio (BTC, ETH, SOL)
  BtcLagExtractor            — BTC log-diff return lagged 24h/48h/72h/168h
  MarketCloseExtractor       — BTC price at XETRA / NYSE / TSE session close
  DisasterExtractor          — FEMA disaster severity score [0, 1]
  MilitaryExtractor          — GDELT US military activity score [0, 1]
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import raw_dir

logger = logging.getLogger(__name__)

_SYMBOLS = ["BTC", "ETH", "SOL", "VIX"]
_SHORT_WINDOW = 24    # 1 day in hours
_LONG_WINDOW = 168    # 1 week in hours

# ─────────────────────────────────────────────────────────────────────────────
# Common DataFrame loader
# ─────────────────────────────────────────────────────────────────────────────


def load_common_dataframe(config: dict) -> pd.DataFrame:
    """Load all Parquet files and produce a complete 24-hour UTC hourly DataFrame.

    Steps:
      1. Load BTC, ETH, SOL, VIX — inner-join on timestamp.
      2. Reindex to a complete hourly UTC grid; forward-fill all gaps
         (VIX weekends/holidays, FEMA/GDELT daily → hourly expansion).
      3. Join FEMA and GDELT as daily soft signals (forward-filled to hourly).

    Raises FileNotFoundError if a core symbol file (BTC/ETH/SOL/VIX) is missing.
    FEMA and GDELT are optional — missing files produce a zero-filled column.
    """
    rd = raw_dir(config)

    # --- Core OHLCV symbols ---
    frames: list[pd.DataFrame] = []
    for symbol in _SYMBOLS:
        path: Path = rd / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Parquet for {symbol} at {path}. "
                "Run 'python main.py collect' first."
            )
        df = pd.read_parquet(path)
        df.columns = [f"{symbol}_{col}" for col in df.columns]
        frames.append(df)

    combined = pd.concat(frames, axis=1, join="inner")

    # Complete 24-hour hourly UTC grid — forward-fill all gaps
    full_index = pd.date_range(
        start=combined.index.min(),
        end=combined.index.max(),
        freq="1h",
        tz="UTC",
    )
    full_index.name = "timestamp"
    combined = combined.reindex(full_index).ffill()

    # --- Soft signals (optional) ---
    for symbol, col in [("FEMA", "FEMA_score"), ("GDELT", "GDELT_military_score")]:
        path = rd / f"{symbol}.parquet"
        if path.exists():
            s = pd.read_parquet(path)[col]
            s = s.reindex(full_index).ffill().fillna(0.0)
        else:
            logger.warning("%s.parquet not found — filling %s with 0", symbol, col)
            s = pd.Series(0.0, index=full_index, name=col)
        combined[col] = s

    combined.index.name = "timestamp"
    logger.info(
        "Common DataFrame: %d rows × %d cols | %s → %s",
        len(combined),
        len(combined.columns),
        combined.index.min().date(),
        combined.index.max().date(),
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Strategy base
# ─────────────────────────────────────────────────────────────────────────────


class FeatureExtractor(ABC):
    """Base strategy for adding features to the common DataFrame."""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Names of columns this extractor produces."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature columns to *df* in-place and return it."""


# ─────────────────────────────────────────────────────────────────────────────
# Concrete strategies
# ─────────────────────────────────────────────────────────────────────────────


class LogDiffReturnExtractor(FeatureExtractor):
    """Hourly log-difference returns: log(P_t) − log(P_{t−1}).

    Transforms price levels into stationary log-returns for all symbols.
    SOL_log_return is the primary HMM observation series.
    """

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
    """Rolling std of log-diff returns over 24h and 168h windows."""

    @property
    def feature_names(self) -> list[str]:
        return [
            f"{s}_vol_{w}h"
            for s in _SYMBOLS
            for w in (_SHORT_WINDOW, _LONG_WINDOW)
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for sym in _SYMBOLS:
            ret = np.log(df[f"{sym}_close"] / df[f"{sym}_close"].shift(1))
            df[f"{sym}_vol_{_SHORT_WINDOW}h"] = ret.rolling(_SHORT_WINDOW).std()
            df[f"{sym}_vol_{_LONG_WINDOW}h"] = ret.rolling(_LONG_WINDOW).std()
        return df


class RollingCorrelationExtractor(FeatureExtractor):
    """Rolling Pearson correlation of SOL log-return vs BTC and ETH."""

    @property
    def feature_names(self) -> list[str]:
        return [
            f"SOL_{sym}_corr_{w}h"
            for sym in ("BTC", "ETH")
            for w in (_SHORT_WINDOW, _LONG_WINDOW)
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        sol = np.log(df["SOL_close"] / df["SOL_close"].shift(1))
        for sym in ("BTC", "ETH"):
            other = np.log(df[f"{sym}_close"] / df[f"{sym}_close"].shift(1))
            df[f"SOL_{sym}_corr_{_SHORT_WINDOW}h"] = (
                sol.rolling(_SHORT_WINDOW).corr(other)
            )
            df[f"SOL_{sym}_corr_{_LONG_WINDOW}h"] = (
                sol.rolling(_LONG_WINDOW).corr(other)
            )
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
    """Price momentum: short MA / long MA − 1 (BTC, ETH, SOL)."""

    @property
    def feature_names(self) -> list[str]:
        return [f"{s}_momentum" for s in ("BTC", "ETH", "SOL")]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for sym in ("BTC", "ETH", "SOL"):
            close = df[f"{sym}_close"]
            df[f"{sym}_momentum"] = (
                close.rolling(_SHORT_WINDOW).mean()
                / close.rolling(_LONG_WINDOW).mean()
            ) - 1.0
        return df


class BtcLagExtractor(FeatureExtractor):
    """BTC log-diff return lagged by 24h, 48h, 72h and 168h.

    Captures BTC's autocorrelation structure and delayed market reactions.
    BTC tends to lead altcoins — lagged values improve SOL regime prediction.
    """

    _LAGS = (24, 48, 72, 168)

    @property
    def feature_names(self) -> list[str]:
        return [f"BTC_log_return_lag_{h}h" for h in self._LAGS]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        btc_ret = np.log(df["BTC_close"] / df["BTC_close"].shift(1))
        for h in self._LAGS:
            df[f"BTC_log_return_lag_{h}h"] = btc_ret.shift(h)
        return df


class MarketCloseExtractor(FeatureExtractor):
    """BTC price at the most recent XETRA, NYSE and TSE session close.

    For each timestamp, looks up the BTC close price at the last session
    close of each exchange (DST-aware, incl. public and exchange holidays).
    Also adds the log-return from each exchange close to the current price
    (overnight / off-hours gap return).

    Requires: pandas-market-calendars
    """

    _EXCHANGES: dict[str, str] = {
        "XETRA": "XETRA",
        "NYSE": "NYSE",
        "TSE": "TSE",
    }

    @property
    def feature_names(self) -> list[str]:
        names = []
        for ex in self._EXCHANGES:
            names += [
                f"BTC_at_{ex}_close",          # BTC log-price at last close
                f"BTC_return_since_{ex}_close", # log-return from close to now
            ]
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            import pandas_market_calendars as mcal  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "pandas-market-calendars not installed — "
                "MarketCloseExtractor skipped. "
                "Install with: pip install pandas-market-calendars"
            )
            for name in self.feature_names:
                df[name] = np.nan
            return df

        start = df.index.min()
        end = df.index.max()
        btc = df["BTC_close"]

        for label, cal_name in self._EXCHANGES.items():
            try:
                cal = mcal.get_calendar(cal_name)
                schedule = cal.schedule(
                    start_date=start.date(), end_date=end.date()
                )
                close_times: pd.Series = schedule["market_close"]

                # BTC price at each session close (asof = last known value)
                btc_at_closes = pd.Series(
                    [btc.asof(t) for t in close_times],
                    index=close_times,
                    dtype=float,
                )

                # merge_asof: for each timestamp assign the most recent close
                ref = pd.DataFrame(
                    {"close_time": close_times.values,
                     "btc_close_price": btc_at_closes.values}
                )
                main = df[[]].reset_index()  # just the timestamp index
                merged = pd.merge_asof(
                    main,
                    ref,
                    left_on="timestamp",
                    right_on="close_time",
                    direction="backward",
                )
                merged = merged.set_index("timestamp")

                btc_close_ref = merged["btc_close_price"]
                df[f"BTC_at_{label}_close"] = np.log(btc_close_ref)
                df[f"BTC_return_since_{label}_close"] = np.log(
                    btc / btc_close_ref
                )
            except Exception as exc:
                logger.warning("MarketCloseExtractor %s failed: %s", label, exc)
                df[f"BTC_at_{label}_close"] = np.nan
                df[f"BTC_return_since_{label}_close"] = np.nan

        return df


class DisasterExtractor(FeatureExtractor):
    """FEMA US disaster severity score [0, 1] as a soft feature.

    Higher values indicate more active major disaster declarations.
    Already forward-filled in load_common_dataframe.
    """

    @property
    def feature_names(self) -> list[str]:
        return ["FEMA_score"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Column already present from load_common_dataframe — pass-through
        if "FEMA_score" not in df.columns:
            df["FEMA_score"] = 0.0
        return df


class MilitaryExtractor(FeatureExtractor):
    """GDELT US military activity score [0, 1] as a soft feature.

    Higher values indicate elevated news volume about US military operations.
    Already forward-filled in load_common_dataframe.
    """

    @property
    def feature_names(self) -> list[str]:
        return ["GDELT_military_score"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Column already present from load_common_dataframe — pass-through
        if "GDELT_military_score" not in df.columns:
            df["GDELT_military_score"] = 0.0
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Registry and builder
# ─────────────────────────────────────────────────────────────────────────────

ALL_EXTRACTORS: list[FeatureExtractor] = [
    LogDiffReturnExtractor(),
    RollingVolatilityExtractor(),
    RollingCorrelationExtractor(),
    VixLevelExtractor(),
    MomentumExtractor(),
    BtcLagExtractor(),
    MarketCloseExtractor(),
    DisasterExtractor(),
    MilitaryExtractor(),
]

ALL_FEATURE_NAMES: list[str] = [
    name for ext in ALL_EXTRACTORS for name in ext.feature_names
]


def build_feature_matrix(
    df: pd.DataFrame,
    feature_subset: list[str],
) -> pd.DataFrame:
    """Apply all extractors, select *feature_subset*, drop NaN rows.

    SOL_log_return is always included as the HMM observation series.
    Returns a clean matrix ready for GaussianHMM fitting.
    """
    for extractor in ALL_EXTRACTORS:
        df = extractor.transform(df)

    required = ["SOL_log_return"]
    cols = required + [f for f in feature_subset if f not in required]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Unknown feature(s): {missing}. "
            f"Available: {ALL_FEATURE_NAMES}"
        )
    return df[cols].dropna()
