import base64
import json

import httpx
import pytest

from t212bot.broker.t212_client import (
    DryRunOrder,
    LiveTradingDisabled,
    T212Client,
    T212Error,
)


class FakeTime:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self):
        return self.now


def make_client(handler, environment="demo", **kwargs):
    t = FakeTime()
    client = T212Client(
        "key",
        "secret",
        environment,
        transport=httpx.MockTransport(handler),
        sleep=t.sleep,
        clock=t.clock,
        wall_clock=t.clock,
        **kwargs,
    )
    return client, t


def test_basic_auth_and_demo_url():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json={"cash": 100})

    client, _ = make_client(handler)
    assert client.account_summary() == {"cash": 100}
    assert seen["url"] == "https://demo.trading212.com/api/v0/equity/account/summary"
    assert seen["auth"] == "Basic " + base64.b64encode(b"key:secret").decode()


def test_limit_order_payload():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 1})

    client, _ = make_client(handler)
    client.place_limit_order("AAPL_US_EQ", 2.5, 180.0)
    assert seen["path"] == "/api/v0/equity/orders/limit"
    assert seen["body"] == {
        "ticker": "AAPL_US_EQ",
        "quantity": 2.5,
        "limitPrice": 180.0,
        "timeValidity": "DAY",
    }


def test_stop_sell_uses_negative_quantity():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": 2})

    client, _ = make_client(handler)
    client.place_stop_order("AAPL_US_EQ", -2.5, 170.0)
    assert seen["body"]["quantity"] == -2.5
    assert seen["body"]["timeValidity"] == "GOOD_TILL_CANCEL"


def test_rejects_bad_inputs():
    client, _ = make_client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        client.place_market_order("AAPL_US_EQ", 0)
    with pytest.raises(ValueError):
        client.place_limit_order("AAPL_US_EQ", 1, -5)


def test_live_orders_blocked_unless_allowed():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"id": 3})

    client, _ = make_client(handler, environment="live")
    with pytest.raises(LiveTradingDisabled):
        client.place_market_order("AAPL_US_EQ", 1)
    assert calls == []
    client.account_summary()  # reads are fine
    allowed, _ = make_client(handler, environment="live", allow_live=True)
    allowed.place_market_order("AAPL_US_EQ", 1)
    assert calls[-1].url.host == "live.trading212.com"


def test_dry_run_sends_nothing():
    calls = []
    client, _ = make_client(lambda r: calls.append(r) or httpx.Response(200, json={}), dry_run=True)
    result = client.place_market_order("AAPL_US_EQ", 1)
    assert isinstance(result, DryRunOrder)
    client.cancel_order(5)
    assert calls == []


def test_retries_after_429_using_reset_header():
    responses = iter(
        [
            httpx.Response(429, headers={"x-ratelimit-reset": "1003"}, text="slow down"),
            httpx.Response(200, json=[{"ticker": "AAPL_US_EQ"}]),
        ]
    )
    client, t = make_client(lambda r: next(responses))
    assert client.positions() == [{"ticker": "AAPL_US_EQ"}]
    assert t.sleeps and t.sleeps[0] == pytest.approx(3.25)


def test_raises_after_retries_exhausted():
    client, _ = make_client(lambda r: httpx.Response(429, text="no"), max_retries=1)
    with pytest.raises(T212Error) as err:
        client.positions()
    assert err.value.status == 429


def test_client_side_pacing():
    client, t = make_client(lambda r: httpx.Response(200, json={}))
    client.account_summary()
    client.account_summary()
    assert t.sleeps == [pytest.approx(5.0)]


def test_post_not_retried_on_server_error():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="down")

    client, _ = make_client(handler)
    with pytest.raises(T212Error):
        client.place_market_order("AAPL_US_EQ", 1)
    assert len(calls) == 1
