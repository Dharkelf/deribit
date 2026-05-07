"""DeribitClient — Repository pattern wrapping the public REST API.

Fetches OHLCV candles from /public/get_tradingview_chart_data with automatic
chunking (Deribit caps responses at 5 000 candles per request).
"""

import logging
from datetime import datetime, timezone

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.deribit.com/api/v2"
_MAX_CANDLES = 5_000


class DeribitClient:
    """Repository for Deribit OHLCV data. One instance per session."""

    def __init__(self, resolution_minutes: int = 60) -> None:
        self._resolution = resolution_minutes
        self._http = httpx.Client(base_url=_BASE_URL, timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "DeribitClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_ohlcv(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV for *instrument* between *start* and *end* (UTC).

        Splits the range into chunks to stay within the 5 000-candle limit.
        Returns a DataFrame indexed by UTC timestamp with columns:
        open, high, low, close, volume.
        """
        start_ms = int(start.timestamp() * 1_000)
        end_ms = int(end.timestamp() * 1_000)
        chunk_ms = _MAX_CANDLES * self._resolution * 60 * 1_000

        frames: list[pd.DataFrame] = []
        cursor = start_ms

        while cursor < end_ms:
            chunk_end = min(cursor + chunk_ms, end_ms)
            logger.debug(
                "Fetching %s [%s → %s]",
                instrument,
                _ms_to_iso(cursor),
                _ms_to_iso(chunk_end),
            )
            frame = self._fetch_chunk(instrument, cursor, chunk_end)
            if frame.empty:
                # Instrument may not yet exist for this window (e.g. SOL_USDC-PERPETUAL
                # launched 2022-03). Skip the window and try the next chunk rather than
                # stopping — once the instrument is live, subsequent chunks will have data.
                cursor = chunk_end
                continue
            frames.append(frame)
            # advance past the last returned candle to avoid duplicates
            cursor = int(frame.index[-1].timestamp() * 1_000) + self._resolution * 60 * 1_000

        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        result = pd.concat(frames)
        result = result[~result.index.duplicated(keep="last")].sort_index()
        logger.info("Fetched %d candles for %s", len(result), instrument)
        return result

    def _fetch_chunk(
        self, instrument: str, start_ms: int, end_ms: int
    ) -> pd.DataFrame:
        params = {
            "instrument_name": instrument,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "resolution": str(self._resolution),
        }
        response = self._http.get("/public/get_tradingview_chart_data", params=params)
        response.raise_for_status()
        payload = response.json()

        if "error" in payload:
            raise RuntimeError(f"Deribit API error: {payload['error']}")

        data = payload["result"]
        if data.get("status") != "ok" or not data.get("ticks"):
            return pd.DataFrame()

        index = pd.to_datetime(data["ticks"], unit="ms", utc=True)
        index.name = "timestamp"
        return pd.DataFrame(
            {
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "volume": data["volume"],
            },
            index=index,
        )


def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1_000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")
