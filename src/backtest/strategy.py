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

Trailing Stop
-------------
Optional trailing stop-loss: when the strategy's cumulative equity falls more
than `trailing_stop_pct` % below its running peak, the position is forced to
zero for the remainder of that regime phase.  The stop resets on the next
regime-label change, allowing a fresh entry.  The peak itself is never reset —
only the "stopped" flag is cleared so drawdown can continue accumulating against
the same all-time-high equity.

Example: trailing_stop_pct=15 → stop fires when equity drops 15 % from peak.
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


def _apply_trailing_stop(
    positions: np.ndarray,
    log_returns: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (adjusted_positions, stopped_mask).

    Iterates hour-by-hour, maintaining running peak equity.
    Stop fires when (equity / peak) - 1 < -threshold.
    Clears on regime-label change.
    """
    n        = len(positions)
    adj_pos  = positions.copy()
    stopped  = np.zeros(n, dtype=bool)
    equity   = 1.0
    peak     = 1.0
    is_stopped  = False
    prev_label  = labels[0] if n > 0 else ""

    for i in range(n):
        # Regime change → allow re-entry
        if labels[i] != prev_label:
            is_stopped = False
            prev_label = labels[i]

        if is_stopped:
            adj_pos[i] = 0.0

        stopped[i] = is_stopped

        # Realise this step's return with (possibly zeroed) position
        equity *= np.exp(adj_pos[i] * log_returns[i])
        if equity > peak:
            peak = equity

        # Check whether stop should fire for the next step
        if not is_stopped and (equity / peak) - 1.0 < -threshold:
            is_stopped = True

    return adj_pos, stopped


class RegimeStrategy:
    """Maps HMM regime label strings to leveraged positions and computes P&L."""

    def __init__(self, position_map: dict[str, float] | None = None) -> None:
        self.position_map = position_map or _POSITION_MAP

    def apply(
        self,
        sol_log_returns: pd.Series,
        regime_labels: pd.Series,
        trailing_stop_pct: float | None = None,
    ) -> pd.DataFrame:
        """Compute hourly strategy and buy-and-hold returns.

        Parameters
        ----------
        sol_log_returns   : hourly SOL log-returns
        regime_labels     : HMM regime label per hour
        trailing_stop_pct : optional trailing stop threshold in percent
                            (e.g. 15 → stop fires when equity drops 15 % from peak)

        Returns DataFrame with columns:
            regime, position, stopped, strategy_lr, bnh_lr,
            equity_strategy, equity_bnh
        """
        idx = sol_log_returns.index.intersection(regime_labels.index)
        lr  = sol_log_returns.loc[idx].fillna(0.0)
        lbl = regime_labels.loc[idx]
        pos = lbl.map(self.position_map).fillna(0.0).to_numpy()

        stopped_mask = np.zeros(len(pos), dtype=bool)
        if trailing_stop_pct is not None and trailing_stop_pct > 0:
            pos, stopped_mask = _apply_trailing_stop(
                pos,
                lr.to_numpy(),
                lbl.to_numpy(),
                trailing_stop_pct / 100.0,
            )

        strat_lr = pos * lr.to_numpy()
        bnh_lr   = lr.to_numpy()

        equity_strat = np.exp(np.cumsum(strat_lr))
        equity_bnh   = np.exp(np.cumsum(bnh_lr))

        return pd.DataFrame(
            {
                "regime":          lbl.values,
                "position":        pos,
                "stopped":         stopped_mask,
                "strategy_lr":     strat_lr,
                "bnh_lr":          bnh_lr,
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
