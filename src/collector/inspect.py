"""Visual inspection of all collected data sources.

Standalone script — not imported by any other module.

Usage:
    python -m src.collector.inspect
    python src/collector/inspect.py
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "config" / "settings.yaml"


def _raw_dir() -> Path:
    with open(_CONFIG) as f:
        config = yaml.safe_load(f)
    return _ROOT / config["storage"]["raw_dir"]


def _load(raw: Path, symbol: str, col: str) -> pd.Series:
    path = raw / f"{symbol}.parquet"
    if not path.exists():
        return pd.Series(dtype=float, name=col)
    df = pd.read_parquet(path)
    return df[col].rename(symbol)


def _fmt_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    raw = _raw_dir()

    panels: list[tuple[str, pd.Series, str]] = [
        ("BTC close (USD)",              _load(raw, "BTC",  "close"),                       "#f7931a"),
        ("ETH close (USD)",              _load(raw, "ETH",  "close"),                       "#627eea"),
        ("SOL close (USD)",              _load(raw, "SOL",  "close"),                       "#9945ff"),
        ("VIX",                          _load(raw, "VIX",  "close"),                       "#e74c3c"),
        ("BTC Max Pain 30d (USD)",       _load(raw, "BTC_OPTIONS_MAX_PAIN",
                                               "BTC_options_max_pain"),                     "#f39c12"),
        ("BTC Max Pain 7d (USD)",        _load(raw, "BTC_OPTIONS_MAX_PAIN",
                                               "BTC_options_max_pain_7d"),                  "#e67e22"),
        ("FEMA Disaster Score",          _load(raw, "FEMA",  "FEMA_score"),                 "#2ecc71"),
        ("GDELT Military Score",         _load(raw, "GDELT", "GDELT_military_score"),        "#3498db"),
    ]

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=False)
    fig.suptitle("Deribit — Collected Data (last 365 days)", fontsize=13, y=1.002)

    for ax, (title, series, color) in zip(axes, panels):
        if series.empty:
            ax.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.set_title(title, fontsize=9, loc="left")
            _fmt_axis(ax)
            continue

        ax.plot(series.index, series.values, color=color, linewidth=0.9, alpha=0.9)
        ax.fill_between(series.index, series.values, alpha=0.08, color=color)
        ax.set_title(
            f"{title}   "
            f"[min {series.min():.2f}  max {series.max():.2f}  "
            f"last {series.iloc[-1]:.2f}]",
            fontsize=9, loc="left",
        )
        _fmt_axis(ax)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
