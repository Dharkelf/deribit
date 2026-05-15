"""Backtesting metrics — forecast accuracy and strategy performance."""

import numpy as np


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    valid = ~np.isnan(actual) & ~np.isnan(predicted)
    if valid.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual[valid] - predicted[valid]) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    valid = ~np.isnan(actual) & ~np.isnan(predicted)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(actual[valid] - predicted[valid])))


def directional_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
    starts: np.ndarray | None = None,
) -> float:
    """Fraction of predictions where sign(Δpred) == sign(Δactual).

    Without *starts*: compares consecutive time-series changes (np.diff).
      Use this when actual/predicted are adjacent time-steps of one series.

    With *starts*: compares each element's direction from its own start price.
      Use this for multi-fold backtests where consecutive rows are NOT adjacent
      in time (e.g. walk-forward folds 7 days apart).
      sign(predicted[i] − starts[i]) == sign(actual[i] − starts[i])
    """
    if starts is not None:
        if len(actual) == 0:
            return float("nan")
        act_dir  = np.sign(actual - starts)
        pred_dir = np.sign(predicted - starts)
        valid    = act_dir != 0
        if valid.sum() == 0:
            return float("nan")
        return float((act_dir[valid] == pred_dir[valid]).mean())
    # Time-series mode: consecutive differences
    if len(actual) < 2 or len(predicted) < 2:
        return float("nan")
    act_dir  = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    valid    = act_dir != 0
    if valid.sum() == 0:
        return float("nan")
    return float((act_dir[valid] == pred_dir[valid]).mean())


def sharpe(returns: np.ndarray, periods_per_year: int = 8760) -> float:
    """Annualised Sharpe ratio (rf = 0)."""
    if len(returns) == 0 or np.std(returns) < 1e-12:
        return float("nan")
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a fraction (negative = loss)."""
    if len(equity_curve) == 0:
        return 0.0
    peak     = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / np.where(peak > 0, peak, 1.0)
    return float(drawdown.min())


def annualized_return(log_returns: np.ndarray, periods_per_year: int = 8760) -> float:
    """Annualised compounded return from log-returns."""
    if len(log_returns) == 0:
        return float("nan")
    total_lr = float(np.sum(log_returns))
    years    = len(log_returns) / periods_per_year
    return float(np.exp(total_lr / max(years, 1e-9)) - 1)
