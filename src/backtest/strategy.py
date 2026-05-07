"""Regime-based trading strategy.

Design
------
Template Method: apply() is the fixed skeleton.

Position map (v1, no transaction costs):
  Strong Bullish  → +1.0   full long
  Bullish         → +0.5   half long
  Neutral         →  0.0   cash
  Bearish         → −0.5   half short
  Strong Bearish  → −1.0   full short

Hourly P&L = position × SOL_log_return.
Transaction costs are not modelled; backtest results represent an upper bound.
"""

import numpy as np
import pandas as pd

_POSITION_MAP: dict[str, float] = {
    "Strong Bullish":  1.0,
    "Bullish":         0.5,
    "Neutral":         0.0,
    "Bearish":        -0.5,
    "Strong Bearish": -1.0,
}


class RegimeStrategy:
    """Maps HMM regime label strings to leveraged positions and computes P&L."""

    def __init__(self, position_map: dict[str, float] | None = None) -> None:
        self.position_map = position_map or _POSITION_MAP

    def apply(
        self,
        sol_log_returns: pd.Series,
        regime_labels: pd.Series,
    ) -> pd.DataFrame:
        """Compute hourly strategy and buy-and-hold returns.

        Returns DataFrame with columns:
            regime, position, strategy_lr, bnh_lr, equity_strategy, equity_bnh
        """
        idx = sol_log_returns.index.intersection(regime_labels.index)
        lr  = sol_log_returns.loc[idx].fillna(0.0)
        lbl = regime_labels.loc[idx]
        pos = lbl.map(self.position_map).fillna(0.0)

        strat_lr = pos * lr
        bnh_lr   = lr.copy()

        equity_strat = np.exp(np.cumsum(strat_lr.values))
        equity_bnh   = np.exp(np.cumsum(bnh_lr.values))

        return pd.DataFrame(
            {
                "regime":          lbl.values,
                "position":        pos.values,
                "strategy_lr":     strat_lr.values,
                "bnh_lr":          bnh_lr.values,
                "equity_strategy": equity_strat,
                "equity_bnh":      equity_bnh,
            },
            index=idx,
        )

    def per_regime_pnl(self, result: pd.DataFrame) -> pd.DataFrame:
        """Aggregate strategy and buy-and-hold P&L by regime label."""
        return (
            result.groupby("regime")
            .agg(
                hours=("strategy_lr", "count"),
                strategy_lr_sum=("strategy_lr", "sum"),
                bnh_lr_sum=("bnh_lr", "sum"),
            )
            .assign(
                strategy_cum_ret=lambda d: np.exp(d["strategy_lr_sum"]) - 1,
                bnh_cum_ret=lambda d: np.exp(d["bnh_lr_sum"]) - 1,
            )
        )
