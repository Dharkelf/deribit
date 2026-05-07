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

Discrete Trading
----------------
Optional signal-based mode: instead of re-evaluating every hour, the regime
signal is evaluated at discrete entry points and the resulting position is
held for a fixed number of wall-clock hours.

  min_hold_hours : hold duration for weak regimes (Bullish, Bearish, Neutral)
  max_hold_hours : hold duration for strong regimes (Strong Bullish, Strong Bearish)

Entries are only taken within the UTC trading window [window_start, window_end).
Position is zero outside that window.  After a hold expires the next entry is
the first in-window hour ≥ previous_entry + hold_h.

Trailing Stop
-------------
Optional trailing stop-loss (threshold in %).

Hourly mode  : when cumulative equity falls more than threshold % below its
               running peak, the position is forced to zero for the remainder
               of that regime phase.  Peak and stopped-flag both reset on the
               next regime-label change.

Discrete mode: peak resets at every new trade entry.  Stop is checked hourly
               within the hold; if triggered, position is zeroed for the
               remainder of that hold period.

Trading Hours (hourly mode only)
---------------------------------
Optional UTC hour range [start, end) — e.g. [8, 18] means 08:00–17:59 UTC.
Positions outside this window are zeroed before trailing stop.  Use
trading_window in discrete mode instead.
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


# ── Hourly-mode helpers ───────────────────────────────────────────────────────

def _apply_trailing_stop(
    positions: np.ndarray,
    log_returns: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (adjusted_positions, stopped_mask) for hourly trading mode.

    Peak and stopped-flag reset on regime-label change.
    """
    n        = len(positions)
    adj_pos  = positions.copy()
    stopped  = np.zeros(n, dtype=bool)
    equity   = 1.0
    peak     = 1.0
    is_stopped  = False
    prev_label  = labels[0] if n > 0 else ""

    for i in range(n):
        if labels[i] != prev_label:
            is_stopped = False
            peak = equity
            prev_label = labels[i]

        if is_stopped:
            adj_pos[i] = 0.0

        stopped[i] = is_stopped

        equity *= np.exp(adj_pos[i] * log_returns[i])
        if equity > peak:
            peak = equity

        if not is_stopped and (equity / peak) - 1.0 < -threshold:
            is_stopped = True

    return adj_pos, stopped


# ── Discrete-mode helper ──────────────────────────────────────────────────────

def _apply_discrete_trading(
    positions: np.ndarray,
    labels: np.ndarray,
    log_returns: np.ndarray,
    hour_of_day: np.ndarray,
    min_hold_h: int,
    max_hold_h: int,
    window_start: int,
    window_end: int,
    trailing_stop_pct: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signal-based discrete trading within a UTC trading window.

    At each evaluation point (first in-window hour after previous hold expires)
    the regime label determines the hold duration:
      "Strong *" → max_hold_h wall-clock hours
      otherwise  → min_hold_h wall-clock hours

    Position is zero outside [window_start, window_end) UTC.
    If trailing_stop_pct is set, equity is tracked hourly within each hold;
    a >threshold % drop from the trade-entry equity fires the stop for the
    remainder of that hold.

    Returns
    -------
    adj_pos   : np.ndarray  position per hour
    stopped   : np.ndarray  bool, True where stop zeroed a position
    off_hours : np.ndarray  bool, True where outside trading window
    """
    n         = len(positions)
    adj_pos   = np.zeros(n)
    stopped   = np.zeros(n, dtype=bool)
    off_hours = (hour_of_day < window_start) | (hour_of_day >= window_end)
    threshold = (trailing_stop_pct / 100.0) if trailing_stop_pct is not None else None

    equity = 1.0
    i = 0
    while i < n:
        if off_hours[i]:
            i += 1
            continue

        # ── Evaluation point ─────────────────────────────────────────────────
        label = str(labels[i])
        hold  = max_hold_h if "Strong" in label else min_hold_h
        pos   = positions[i]
        peak  = equity      # per-trade peak for trailing stop

        is_stopped_trade = False
        end = min(i + hold, n)

        for j in range(i, end):
            if off_hours[j]:
                pass                         # flat, equity unchanged
            elif is_stopped_trade:
                stopped[j] = True            # still flat, equity unchanged
            else:
                adj_pos[j] = pos
                equity *= np.exp(pos * log_returns[j])
                if threshold is not None and (equity / peak) - 1.0 < -threshold:
                    is_stopped_trade = True

        i += hold

    return adj_pos, stopped, off_hours


# ── Strategy ──────────────────────────────────────────────────────────────────

class RegimeStrategy:
    """Maps HMM regime label strings to leveraged positions and computes P&L."""

    def __init__(self, position_map: dict[str, float] | None = None) -> None:
        self.position_map = position_map or _POSITION_MAP

    def apply(
        self,
        sol_log_returns: pd.Series,
        regime_labels: pd.Series,
        trailing_stop_pct: float | None = None,
        trading_hours: tuple[int, int] | None = None,
        discrete_trading: tuple[int, int] | None = None,
        trading_window: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Compute hourly strategy and buy-and-hold returns.

        Parameters
        ----------
        sol_log_returns   : hourly SOL log-returns
        regime_labels     : HMM regime label per hour
        trailing_stop_pct : optional trailing stop threshold in percent
        trading_hours     : (start_h, end_h) UTC — hourly mode only; positions
                            outside window zeroed before trailing stop
        discrete_trading  : (min_hold_h, max_hold_h) — enables discrete mode;
                            regime evaluated at entry, held for N wall-clock hours
        trading_window    : (start_h, end_h) UTC — discrete mode only; new entries
                            only within this window; position=0 outside

        Returns DataFrame with columns:
            regime, position, off_hours, stopped,
            strategy_lr, bnh_lr, equity_strategy, equity_bnh
        """
        idx = sol_log_returns.index.intersection(regime_labels.index)
        lr  = sol_log_returns.loc[idx].fillna(0.0)
        lbl = regime_labels.loc[idx]
        pos = lbl.map(self.position_map).fillna(0.0).to_numpy()

        if discrete_trading is not None:
            # ── Discrete signal-based trading ─────────────────────────────────
            min_hold_h, max_hold_h = discrete_trading
            win_start, win_end = trading_window if trading_window is not None else (0, 24)
            pos, stopped_mask, off_hours_mask = _apply_discrete_trading(
                pos,
                lbl.to_numpy(),
                lr.to_numpy(),
                idx.hour.to_numpy(),
                min_hold_h,
                max_hold_h,
                win_start,
                win_end,
                trailing_stop_pct=trailing_stop_pct,
            )
        else:
            # ── Hourly evaluation mode ────────────────────────────────────────
            off_hours_mask = np.zeros(len(pos), dtype=bool)
            if trading_hours is not None:
                start_h, end_h = trading_hours
                hour_of_day = idx.hour.to_numpy()
                if start_h < end_h:
                    off_hours_mask = (hour_of_day < start_h) | (hour_of_day >= end_h)
                else:
                    off_hours_mask = (hour_of_day < start_h) & (hour_of_day >= end_h)
                pos[off_hours_mask] = 0.0

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
                "off_hours":       off_hours_mask,
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
