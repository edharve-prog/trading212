"""Daily price bars from an external provider (Trading212 has no price endpoint)."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd

BAR_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


class MarketData(Protocol):
    def daily_bars(self, symbol: str, start: date, end: date | None = None) -> pd.DataFrame:
        """Return bars with BAR_COLUMNS, one row per trading day, sorted by date.

        ``end`` is exclusive. An empty frame means no data in the range.
        """
        ...


def normalise_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a provider frame (date index, Title Case columns) onto BAR_COLUMNS."""
    if raw.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    df["date"] = index.normalize()
    df = df[BAR_COLUMNS].dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0).astype("int64")
    for col in ["open", "high", "low", "close", "adj_close"]:
        df[col] = df[col].astype("float64")
    df["date"] = df["date"].astype("datetime64[ns]")
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


class YFinanceProvider:
    """Yahoo Finance via yfinance. Free and unofficial: it can break or throttle."""

    def daily_bars(self, symbol: str, start: date, end: date | None = None) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=end.isoformat() if end else None,
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
        return normalise_bars(raw)
