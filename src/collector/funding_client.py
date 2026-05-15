"""DeribitFundingRateClient + DeribitOIClient — Deribit perpetual market metrics.

FundingRateClient: fetches 8h funding rate history for a perpetual instrument.
  Endpoint: GET /public/get_funding_rate_history
  Stored hourly (forward-filled from 8h intervals).
  Column: funding_rate_8h (decimal, e.g. 0.0001 = 0.01%)

OIClient: daily open interest snapshot, SOL vs BTC notional in USD.
  Endpoint: GET /public/get_book_summary_by_currency?kind=future
  Column: SOL_oi_usd, BTC_oi_usd, SOL_OI_BTC_ratio

Both clients require no API key (Deribit public endpoints).
"""

import logging
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.deribit.com/api/v2"
_MAX_RECORDS = 700  # Deribit caps at 744; stay safely below


class DeribitFundingRateClient:
    """Repository for Deribit perpetual funding rate history."""

    def __init__(self, instrument: str = "SOL_USDC-PERPETUAL") -> None:
        self._instrument = instrument
        self._http = httpx.Client(base_url=_BASE_URL, timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "DeribitFundingRateClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_hourly(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch funding rate history and return forward-filled hourly DataFrame.

        Deribit returns 8h funding snapshots; this method forward-fills to hourly.
        Returns empty DataFrame (with correct column) when no data is available.
        """
        all_records: list[dict] = []
        cursor = int(start.timestamp() * 1_000)
        end_ms = int(end.timestamp() * 1_000)
        chunk_ms = _MAX_RECORDS * 8 * 3_600 * 1_000  # 700 × 8h windows

        while cursor < end_ms:
            chunk_end = min(cursor + chunk_ms, end_ms)
            try:
                resp = self._http.get(
                    "/public/get_funding_rate_history",
                    params={
                        "instrument_name": self._instrument,
                        "start_timestamp": cursor,
                        "end_timestamp": chunk_end,
                    },
                )
                resp.raise_for_status()
                records: list[dict] = resp.json().get("result", [])
                if records:
                    all_records.extend(records)
                    cursor = int(records[-1]["timestamp"]) + 1
                else:
                    cursor = chunk_end
            except Exception as exc:
                logger.warning("Funding rate chunk error (%s): %s", self._instrument, exc)
                cursor = chunk_end

        if not all_records:
            logger.warning("No funding rate data for %s", self._instrument)
            return pd.DataFrame(columns=["funding_rate_8h"])

        df = pd.DataFrame(all_records)
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.index.name = "timestamp"
        df = df[["interest_8h"]].rename(columns={"interest_8h": "funding_rate_8h"})
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # Forward-fill 8h snapshots to complete hourly grid
        full_idx = pd.date_range(
            start=df.index[0], end=df.index[-1], freq="1h", tz="UTC", name="timestamp"
        )
        df = df.reindex(full_idx).ffill()

        logger.info(
            "Funding rate %s: %d hourly rows (%s → %s)",
            self._instrument,
            len(df),
            df.index[0].date(),
            df.index[-1].date(),
        )
        return df


class DeribitOIClient:
    """Repository for Deribit perpetual open interest (daily snapshot)."""

    def __init__(self) -> None:
        self._http = httpx.Client(base_url=_BASE_URL, timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "DeribitOIClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_daily_snapshot(self) -> pd.DataFrame:
        """Fetch current OI for SOL and BTC perpetuals; return SOL/BTC notional ratio.

        Columns: SOL_oi_usd, BTC_oi_usd, SOL_OI_BTC_ratio
        SOL_OI_BTC_ratio > typical = elevated SOL leverage vs BTC (risk-on signal).
        """
        today = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        index = pd.DatetimeIndex([today], name="timestamp")

        # Deribit OI semantics:
        #   BTC-PERPETUAL (inverse): open_interest is in USD directly → no price mult.
        #   SOL_USDC-PERPETUAL (linear): open_interest is in SOL → multiply by mark_price.
        _QUERY: dict[str, tuple[str, str, bool]] = {
            "SOL": ("USDC", "SOL_USDC-PERPETUAL", True),   # linear: OI in contracts
            "BTC": ("BTC",  "BTC-PERPETUAL",       False),  # inverse: OI already in USD
        }
        oi_usd: dict[str, float] = {}
        for asset, (currency, instrument, needs_price_mult) in _QUERY.items():
            try:
                resp = self._http.get(
                    "/public/get_book_summary_by_currency",
                    params={"currency": currency, "kind": "future"},
                )
                resp.raise_for_status()
                results: list[dict] = resp.json().get("result", [])
                perp = next(
                    (r for r in results if r.get("instrument_name") == instrument),
                    None,
                )
                if perp:
                    oi_raw = float(perp.get("open_interest") or 0)
                    if needs_price_mult:
                        price = float(
                            perp.get("underlying_price")
                            or perp.get("mark_price")
                            or 0
                        )
                        oi_usd[asset] = oi_raw * price if price > 0 else np.nan
                    else:
                        oi_usd[asset] = oi_raw if oi_raw > 0 else np.nan
                    logger.debug(
                        "OI %s: instrument=%s  OI_raw=%.0f  oi_usd=%.0f",
                        asset,
                        instrument,
                        oi_raw,
                        oi_usd[asset] if not np.isnan(oi_usd[asset]) else -1,
                    )
                else:
                    logger.warning("OI: instrument %s not found", instrument)
                    oi_usd[asset] = np.nan
            except Exception as exc:
                logger.warning("OI fetch error for %s: %s", asset, exc)
                oi_usd[asset] = np.nan

        sol = oi_usd.get("SOL", np.nan)
        btc = oi_usd.get("BTC", np.nan)
        ratio = (
            sol / btc
            if (not np.isnan(sol) and not np.isnan(btc) and btc > 0)
            else np.nan
        )
        logger.info(
            "OI snapshot — SOL: $%.0fM  BTC: $%.0fM  SOL/BTC ratio=%.4f",
            (sol or 0) / 1e6,
            (btc or 0) / 1e6,
            ratio if not np.isnan(ratio) else -1,
        )
        return pd.DataFrame(
            {
                "SOL_oi_usd": [sol],
                "BTC_oi_usd": [btc],
                "SOL_OI_BTC_ratio": [ratio],
            },
            index=index,
        )
