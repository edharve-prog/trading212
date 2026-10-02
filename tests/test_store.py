from datetime import date

import pandas as pd

from t212bot.data.market_data import BAR_COLUMNS, normalise_bars
from t212bot.data.store import PriceStore


def bars(start, periods, base=100.0):
    dates = pd.bdate_range(start, periods=periods)
    close = [base + i for i in range(periods)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "adj_close": close,
            "volume": 1000,
        }
    )


class FakeProvider:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def daily_bars(self, symbol, start, end=None):
        self.calls.append((symbol, start, end))
        d = self.frame
        mask = d["date"] >= pd.Timestamp(start)
        if end is not None:
            mask &= d["date"] < pd.Timestamp(end)
        return d[mask].reset_index(drop=True)


def test_write_and_read_across_years(tmp_path):
    store = PriceStore(tmp_path)
    store.write_bars("AAPL", bars("2024-12-20", 10))
    store.write_bars("MSFT", bars("2025-01-02", 3, base=400))
    files = sorted(p.name for p in (tmp_path / "bars/interval=1d/symbol=AAPL").iterdir())
    assert files == ["year=2024.parquet", "year=2025.parquet"]

    out = store.read_bars()
    assert list(out.columns) == ["symbol", *BAR_COLUMNS]
    assert len(out) == 13
    assert store.symbols() == ["AAPL", "MSFT"]

    only = store.read_bars(["MSFT"], start=date(2025, 1, 3))
    assert only["symbol"].unique().tolist() == ["MSFT"]
    assert len(only) == 2


def test_rewrite_dedupes_and_later_rows_win(tmp_path):
    store = PriceStore(tmp_path)
    store.write_bars("AAPL", bars("2025-03-03", 5))
    revised = bars("2025-03-05", 5, base=500)
    store.write_bars("AAPL", revised)
    out = store.read_bars(["AAPL"])
    assert len(out) == 7
    assert out.loc[out["date"] == pd.Timestamp("2025-03-05"), "close"].item() == 500
    assert store.last_date("AAPL") == date(2025, 3, 11)


def test_update_fetches_only_new_bars(tmp_path):
    store = PriceStore(tmp_path)
    provider = FakeProvider(bars("2025-01-01", 30))
    first = store.update(provider, "SPY", date(2025, 1, 1), today=date(2025, 1, 20))
    assert first > 0
    last = store.last_date("SPY")
    second = store.update(provider, "SPY", date(2025, 1, 1), today=date(2025, 2, 28))
    assert provider.calls[1][1] > last
    assert len(store.read_bars(["SPY"])) == first + second == 30


def test_empty_store_reads_empty(tmp_path):
    store = PriceStore(tmp_path)
    assert store.read_bars().empty
    assert store.last_date("AAPL") is None


def test_normalise_bars_from_yahoo_shape():
    idx = pd.DatetimeIndex(["2025-01-02 00:00", "2025-01-03 00:00"]).tz_localize("America/New_York")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.5, 1.5],
            "Close": [1.2, 2.2],
            "Adj Close": [1.1, 2.1],
            "Volume": [10, 20],
        },
        index=idx,
    )
    out = normalise_bars(raw)
    assert list(out.columns) == BAR_COLUMNS
    assert out["date"].tolist() == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]
    assert out["adj_close"].tolist() == [1.1, 2.1]
