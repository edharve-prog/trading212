from datetime import UTC, date, datetime

import httpx
import pytest

from t212bot.data.alpaca import AlpacaError, AlpacaProvider
from t212bot.data.market_data import BAR_COLUMNS


def bar(t, o, h, low, c, v):
    return {"t": t, "o": o, "h": h, "l": low, "c": c, "v": v, "n": 1, "vw": c}


SPLIT = [
    bar("2025-01-02T05:00:00Z", 100, 102, 99, 101, 1000),
    bar("2025-01-03T05:00:00Z", 101, 104, 100, 103, 1200),
]
ALL = [
    bar("2025-01-02T05:00:00Z", 99, 101, 98, 100.5, 1000),
    bar("2025-01-03T05:00:00Z", 100, 103, 99, 102.5, 1200),
]


def make(handler, now=datetime(2025, 6, 1, 12, tzinfo=UTC), **kwargs):
    sleeps = []
    provider = AlpacaProvider(
        "id",
        "secret",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        now=lambda: now,
        **kwargs,
    )
    return provider, sleeps


def test_daily_bars_merges_split_and_total_return_close():
    requests = []

    def handler(request):
        requests.append(request)
        adj = request.url.params["adjustment"]
        return httpx.Response(200, json={"bars": SPLIT if adj == "split" else ALL})

    provider, _ = make(handler)
    df = provider.daily_bars("AAPL", date(2025, 1, 1), date(2025, 1, 4))

    assert list(df.columns) == BAR_COLUMNS
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-02", "2025-01-03"]
    assert df["close"].tolist() == [101.0, 103.0]
    assert df["adj_close"].tolist() == [100.5, 102.5]
    assert df["volume"].dtype == "int64"

    first = requests[0]
    assert first.url.path == "/v2/stocks/AAPL/bars"
    assert first.headers["APCA-API-KEY-ID"] == "id"
    assert first.headers["APCA-API-SECRET-KEY"] == "secret"
    assert first.url.params["timeframe"] == "1Day"
    assert first.url.params["feed"] == "sip"
    assert first.url.params["start"] == "2025-01-01"
    assert first.url.params["end"] == "2025-01-03"  # end is exclusive in our interface


def test_follows_pagination():
    def handler(request):
        if request.url.params.get("page_token") == "p2":
            return httpx.Response(200, json={"bars": SPLIT[1:], "next_page_token": None})
        return httpx.Response(200, json={"bars": SPLIT[:1], "next_page_token": "p2"})

    provider, _ = make(handler)
    df = provider.daily_bars("AAPL", date(2025, 1, 1), date(2025, 1, 4))
    assert len(df) == 2


def test_recent_end_is_omitted_for_sip():
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"bars": []})

    provider, _ = make(handler, now=datetime(2025, 1, 3, 15, tzinfo=UTC))
    df = provider.daily_bars("AAPL", date(2025, 1, 1), date(2025, 1, 4))
    assert df.empty
    assert "end" not in seen[0]
    assert len(seen) == 1  # no second request when there are no bars


def test_iex_feed_keeps_recent_end():
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"bars": []})

    provider, _ = make(handler, now=datetime(2025, 1, 3, 15, tzinfo=UTC), feed="iex")
    provider.daily_bars("AAPL", date(2025, 1, 1), date(2025, 1, 4))
    assert seen[0]["end"] == "2025-01-03"
    assert seen[0]["feed"] == "iex"


def test_retries_rate_limit_then_raises_on_auth_error():
    responses = iter(
        [httpx.Response(429, text="slow"), httpx.Response(200, json={"bars": SPLIT})]
        + [httpx.Response(200, json={"bars": ALL})]
    )
    provider, sleeps = make(lambda r: next(responses))
    assert len(provider.daily_bars("AAPL", date(2025, 1, 1))) == 2
    assert sleeps == [2.0]

    bad, _ = make(lambda r: httpx.Response(403, text="forbidden"))
    with pytest.raises(AlpacaError) as err:
        bad.daily_bars("AAPL", date(2025, 1, 1))
    assert err.value.status == 403


def test_rejects_unknown_feed():
    with pytest.raises(ValueError):
        AlpacaProvider("id", "secret", feed="otc")
