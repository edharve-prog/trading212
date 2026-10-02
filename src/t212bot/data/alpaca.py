"""Daily bars from the Alpaca market data API (v2).

Docs: https://docs.alpaca.markets/reference/stockbarsingle-1

OHLC and volume are split-adjusted (matching Yahoo's Close); ``adj_close`` is the
split-and-dividend-adjusted close, fetched with a second request. On the free plan the
SIP feed only serves data at least 15 minutes old, so ``end`` is capped accordingly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from .market_data import BAR_COLUMNS

log = logging.getLogger(__name__)

DATA_URL = "https://data.alpaca.markets"
MARKET_TZ = "America/New_York"


class AlpacaError(Exception):
    def __init__(self, status: int, body: str, symbol: str):
        super().__init__(f"Alpaca bars for {symbol} returned HTTP {status}: {body[:300]}")
        self.status = status


class AlpacaProvider:
    def __init__(
        self,
        api_key_id: str,
        api_secret_key: str,
        *,
        feed: str = "sip",
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        if feed not in ("sip", "iex"):
            raise ValueError(f"feed must be 'sip' or 'iex', got {feed!r}")
        self.feed = feed
        self.max_retries = max_retries
        self._sleep = sleep
        self._now = now
        self._http = httpx.Client(
            base_url=DATA_URL,
            headers={
                "APCA-API-KEY-ID": api_key_id,
                "APCA-API-SECRET-KEY": api_secret_key,
                "Accept": "application/json",
            },
            timeout=30.0,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def daily_bars(self, symbol: str, start: date, end: date | None = None) -> pd.DataFrame:
        params: dict[str, Any] = {
            "timeframe": "1Day",
            "start": start.isoformat(),
            "limit": 10000,
            "feed": self.feed,
        }
        end_param = self._end_param(end)
        if end_param is not None:
            params["end"] = end_param
        prices = self._fetch(symbol, {**params, "adjustment": "split"})
        if not prices:
            return pd.DataFrame(columns=BAR_COLUMNS)
        adjusted = self._fetch(symbol, {**params, "adjustment": "all"})
        return bars_to_frame(prices, adjusted)

    def _end_param(self, end: date | None) -> str | None:
        """``end`` is exclusive, so request through the previous day. Omit it if too recent."""
        if end is None:
            return None
        last_day = end - timedelta(days=1)
        cutoff = self._now() - timedelta(minutes=16)
        end_of_day = datetime.combine(last_day, datetime.max.time(), tzinfo=UTC)
        if self.feed == "sip" and end_of_day > cutoff:
            return None  # Alpaca then defaults to the latest data the plan allows
        return last_day.isoformat()

    def _fetch(self, symbol: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        bars: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            query = dict(params)
            if page_token:
                query["page_token"] = page_token
            data = self._get(f"/v2/stocks/{symbol}/bars", query, symbol)
            bars.extend(data.get("bars") or [])
            page_token = data.get("next_page_token")
            if not page_token:
                return bars

    def _get(self, path: str, params: dict[str, Any], symbol: str) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            response = self._http.get(path, params=params)
            retryable = response.status_code == 429 or response.status_code >= 500
            if retryable and attempt < self.max_retries:
                wait = float(2 ** (attempt + 1))
                log.warning(
                    "Alpaca HTTP %s for %s, retrying in %.0fs", response.status_code, symbol, wait
                )
                self._sleep(wait)
                continue
            if response.is_error:
                raise AlpacaError(response.status_code, response.text, symbol)
            result: dict[str, Any] = response.json()
            return result
        raise AssertionError("unreachable")


def _bar_dates(bars: list[dict[str, Any]]) -> pd.Series:
    # Daily bars are stamped at midnight New York time, expressed in UTC.
    stamps = pd.to_datetime([b["t"] for b in bars], utc=True).tz_convert(MARKET_TZ)
    return pd.Series(stamps.tz_localize(None).normalize().astype("datetime64[ns]"))


def bars_to_frame(prices: list[dict[str, Any]], adjusted: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": _bar_dates(prices),
            "open": [float(b["o"]) for b in prices],
            "high": [float(b["h"]) for b in prices],
            "low": [float(b["l"]) for b in prices],
            "close": [float(b["c"]) for b in prices],
            "volume": [int(b["v"]) for b in prices],
        }
    )
    if adjusted:
        adj = pd.DataFrame({"date": _bar_dates(adjusted), "adj_close": [b["c"] for b in adjusted]})
        df = df.merge(adj, on="date", how="left")
        df["adj_close"] = df["adj_close"].astype("float64").fillna(df["close"])
    else:
        df["adj_close"] = df["close"]
    df["volume"] = df["volume"].astype("int64")
    return (
        df[BAR_COLUMNS]
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
