import httpx
import pytest

from t212bot.broker.demo_order_check import NotDemoError, run_demo_order_check
from t212bot.broker.t212_client import T212Client

from .fake_t212 import FakeT212


def make_client(fake, environment="demo", **kwargs):
    return T212Client(
        "key",
        "secret",
        environment,
        transport=httpx.MockTransport(fake.handler),
        sleep=lambda s: None,
        **kwargs,
    )


HELD = [{"ticker": "AAPL_US_EQ", "quantity": 3, "currentPrice": 200.0}]


def test_resting_buy_and_sell_are_placed_and_cancelled():
    fake = FakeT212(positions=HELD)
    result = run_demo_order_check(make_client(fake), "AAPL_US_EQ", 1)
    assert result.ok, result.steps
    assert fake.placed == [
        (
            "limit",
            {"ticker": "AAPL_US_EQ", "quantity": 1, "limitPrice": 180.0, "timeValidity": "DAY"},
        ),
        (
            "limit",
            {"ticker": "AAPL_US_EQ", "quantity": -1, "limitPrice": 220.0, "timeValidity": "DAY"},
        ),
    ]
    assert fake.cancelled == [101, 102]
    assert fake.pending == {}
    assert fake.hosts == {"demo.trading212.com"}


def test_buy_without_position_needs_a_price_and_sell_is_skipped():
    fake = FakeT212()
    result = run_demo_order_check(make_client(fake), "AAPL_US_EQ", 1)
    assert not result.ok
    assert [s.name for s in result.steps] == ["read positions", "place buy", "place sell"]
    assert "--buy-price" in result.steps[1].detail
    assert "skipped" in result.steps[2].detail
    assert fake.placed == []


def test_explicit_buy_price_without_position():
    fake = FakeT212()
    result = run_demo_order_check(make_client(fake), "AAPL_US_EQ", 0.5, buy_price=50)
    buy_steps = [s for s in result.steps if "buy" in s.name]
    assert all(s.ok for s in buy_steps) and len(buy_steps) == 3
    assert fake.placed[0][1]["limitPrice"] == 50
    assert fake.pending == {}


def test_fill_mode_round_trip():
    fake = FakeT212()
    result = run_demo_order_check(make_client(fake), "AAPL_US_EQ", 1, fill=True)
    assert result.ok, result.steps
    assert [(k, b["quantity"]) for k, b in fake.placed] == [("market", 1), ("market", -1)]


def test_fill_mode_cancels_when_market_does_not_fill():
    fake = FakeT212(fill_market=False)
    sleeps = []
    result = run_demo_order_check(
        make_client(fake), "AAPL_US_EQ", 1, fill=True, fill_timeout=10, poll=5, sleep=sleeps.append
    )
    assert not result.ok
    assert sleeps == [5, 5]
    assert len(fake.placed) == 1  # no sell after an unfilled buy
    assert fake.cancelled == [101] and fake.pending == {}


def test_refuses_live_and_dry_run_clients():
    fake = FakeT212()
    with pytest.raises(NotDemoError):
        run_demo_order_check(make_client(fake, "live", allow_live=True), "AAPL_US_EQ", 1)
    with pytest.raises(ValueError):
        run_demo_order_check(make_client(fake, dry_run=True), "AAPL_US_EQ", 1)
    assert fake.placed == []


def test_rejection_is_reported_not_raised():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(400, text="InsufficientFunds")
        return httpx.Response(200, json=HELD if "positions" in request.url.path else [])

    client = T212Client("k", "s", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    result = run_demo_order_check(client, "AAPL_US_EQ", 1)
    assert not result.ok
    assert "InsufficientFunds" in result.steps[1].detail
