"""Parquet price store, one file per symbol per year, queried with DuckDB.

Layout: <data_dir>/bars/interval=1d/symbol=<SYMBOL>/year=<YYYY>.parquet
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from .market_data import BAR_COLUMNS, MarketData


class PriceStore:
    def __init__(self, data_dir: Path | str):
        self.root = Path(data_dir) / "bars" / "interval=1d"

    def _symbol_dir(self, symbol: str) -> Path:
        if not symbol or any(c in symbol for c in '/\\:*?"<>|'):
            raise ValueError(f"unsafe symbol for a path: {symbol!r}")
        return self.root / f"symbol={symbol}"

    def symbols(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name.split("=", 1)[1] for p in self.root.glob("symbol=*") if p.is_dir())

    def write_bars(self, symbol: str, bars: pd.DataFrame) -> int:
        """Merge bars into the store; later rows win for duplicate dates. Returns rows written."""
        if bars.empty:
            return 0
        bars = bars[BAR_COLUMNS].copy()
        bars["date"] = pd.to_datetime(bars["date"]).astype("datetime64[ns]")
        sym_dir = self._symbol_dir(symbol)
        sym_dir.mkdir(parents=True, exist_ok=True)
        for year, chunk in bars.groupby(bars["date"].dt.year):
            path = sym_dir / f"year={year}.parquet"
            if path.exists():
                chunk = pd.concat([pd.read_parquet(path), chunk], ignore_index=True)
            chunk = chunk.drop_duplicates("date", keep="last").sort_values("date")
            chunk = chunk.reset_index(drop=True)
            tmp = path.with_suffix(".parquet.tmp")
            chunk.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(path)
        return len(bars)

    def last_date(self, symbol: str) -> date | None:
        sym_dir = self._symbol_dir(symbol)
        files = sorted(sym_dir.glob("year=*.parquet"))
        if not files:
            return None
        latest = pd.read_parquet(files[-1], columns=["date"])
        if latest.empty:
            return None
        return pd.Timestamp(latest["date"].max()).date()

    def read_bars(
        self,
        symbols: Iterable[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Return bars with a ``symbol`` column, sorted by symbol then date.

        ``end`` is inclusive.
        """
        columns = ["symbol", *BAR_COLUMNS]
        if not self.root.exists() or not any(self.root.glob("symbol=*/*.parquet")):
            return pd.DataFrame(columns=columns)
        pattern = (self.root / "symbol=*" / "*.parquet").as_posix()
        where: list[str] = []
        params: list[object] = []
        if symbols is not None:
            syms = list(symbols)
            if not syms:
                return pd.DataFrame(columns=columns)
            where.append(f"symbol IN ({', '.join('?' for _ in syms)})")
            params.extend(syms)
        if start is not None:
            where.append("date >= ?")
            params.append(pd.Timestamp(start).to_pydatetime())
        if end is not None:
            where.append("date <= ?")
            params.append(pd.Timestamp(end).to_pydatetime())
        sql = (
            f"SELECT {', '.join(columns)} FROM read_parquet(?, hive_partitioning = true)"
            + (f" WHERE {' AND '.join(where)}" if where else "")
            + " ORDER BY symbol, date"
        )
        with duckdb.connect() as con:
            return con.execute(sql, [pattern, *params]).df()

    def update(
        self, provider: MarketData, symbol: str, history_start: date, today: date | None = None
    ) -> int:
        """Fetch bars after the last stored date (or from history_start) and store them."""
        last = self.last_date(symbol)
        start = history_start if last is None else last + timedelta(days=1)
        end = (today or date.today()) + timedelta(days=1)
        if start >= end:
            return 0
        return self.write_bars(symbol, provider.daily_bars(symbol, start, end))
