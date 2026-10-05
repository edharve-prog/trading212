"""End-to-end check of order placement against the Trading212 demo account.

Resting mode (default) places a limit buy below the market and, if the ticker is held,
a limit sell above it, confirms each shows up as a pending order, then cancels it.
Fill mode instead sends a market buy and, once it fills, a market sell of the same size.

This only ever talks to the demo environment: ``run_demo_order_check`` refuses any
client that is not a demo client, and refuses a client in dry-run mode, since that
would test nothing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .t212_client import T212Client


class NotDemoError(Exception):
    pass


@dataclass
class Step:
    name: str
    ok: bool
    detail: str


@dataclass
class CheckResult:
    steps: list[Step] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)

    def add(self, name: str, ok: bool, detail: str) -> bool:
        self.steps.append(Step(name, ok, detail))
        return ok


def position_ticker(position: dict[str, Any]) -> str | None:
    ticker = position.get("ticker")
    if ticker is None and isinstance(position.get("instrument"), dict):
        ticker = position["instrument"].get("ticker")
    return ticker


def _find_position(positions: list[dict[str, Any]], ticker: str) -> dict[str, Any] | None:
    for p in positions:
        if position_ticker(p) == ticker:
            return p
    return None


def _pending_ids(client: T212Client) -> set[str]:
    return {str(o.get("id")) for o in client.orders()}


def _order_id(result: Any) -> str | None:
    if isinstance(result, dict) and result.get("id") is not None:
        return str(result["id"])
    return None


def _round_price(price: float) -> float:
    return round(price, 2) if price >= 1 else round(price, 4)


def _resting_order(
    client: T212Client,
    result: CheckResult,
    side: str,
    ticker: str,
    quantity: float,
    price: float,
) -> None:
    signed = quantity if side == "buy" else -quantity
    label = f"limit {side} {quantity:g} {ticker} @ {price:g}"
    try:
        placed = client.place_limit_order(ticker, signed, price, "DAY")
    except Exception as exc:
        result.add(f"place {side}", False, f"{label}: {exc}")
        return
    order_id = _order_id(placed)
    if not result.add(f"place {side}", order_id is not None, f"{label} -> id {order_id}"):
        return
    assert order_id is not None
    pending = order_id in _pending_ids(client)
    result.add(f"{side} is pending", pending, "listed in open orders" if pending else "not listed")
    try:
        client.cancel_order(int(order_id))
    except Exception as exc:
        result.add(f"cancel {side}", False, f"cancel {order_id}: {exc}")
        return
    gone = order_id not in _pending_ids(client)
    result.add(f"cancel {side}", gone, "cancelled" if gone else "still listed after cancel")


def _wait_filled(
    client: T212Client, order_id: str, timeout: float, poll: float, sleep: Callable[[float], None]
) -> bool:
    waited = 0.0
    while True:
        if order_id not in _pending_ids(client):
            return True
        if waited >= timeout:
            return False
        sleep(poll)
        waited += poll


def _market_round_trip(
    client: T212Client,
    result: CheckResult,
    ticker: str,
    quantity: float,
    timeout: float,
    poll: float,
    sleep: Callable[[float], None],
) -> None:
    for side, signed in (("buy", quantity), ("sell", -quantity)):
        label = f"market {side} {quantity:g} {ticker}"
        try:
            placed = client.place_market_order(ticker, signed)
        except Exception as exc:
            result.add(f"place {side}", False, f"{label}: {exc}")
            return
        order_id = _order_id(placed)
        if not result.add(f"place {side}", order_id is not None, f"{label} -> id {order_id}"):
            return
        assert order_id is not None
        if _wait_filled(client, order_id, timeout, poll, sleep):
            result.add(f"{side} filled", True, "no longer pending")
            continue
        # Most likely the market is closed. Do not leave a queued order behind.
        result.add(f"{side} filled", False, f"still pending after {timeout:g}s (market closed?)")
        try:
            client.cancel_order(int(order_id))
            result.add(f"cancel {side}", True, f"cancelled unfilled order {order_id}")
        except Exception as exc:
            result.add(f"cancel {side}", False, f"cancel {order_id}: {exc}")
        return


def run_demo_order_check(
    client: T212Client,
    ticker: str,
    quantity: float,
    *,
    fill: bool = False,
    buy_price: float | None = None,
    sell_price: float | None = None,
    offset: float = 0.10,
    fill_timeout: float = 60.0,
    poll: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> CheckResult:
    if client.environment != "demo":
        raise NotDemoError("The order check only runs against the Trading212 demo account.")
    if client.dry_run:
        raise ValueError("The order check needs a client with dry_run off; it would test nothing.")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if not 0 < offset < 1:
        raise ValueError("offset must be between 0 and 1")

    result = CheckResult()
    try:
        position = _find_position(client.positions(), ticker)
    except Exception as exc:
        result.add("read positions", False, str(exc))
        return result
    held = float(position.get("quantity", 0)) if position else 0.0
    current = position.get("currentPrice") if position else None
    result.add(
        "read positions",
        True,
        f"holding {held:g} {ticker}" + (f" at {current}" if current is not None else ""),
    )

    if fill:
        _market_round_trip(client, result, ticker, quantity, fill_timeout, poll, sleep)
        return result

    if buy_price is None and current is not None:
        buy_price = _round_price(float(current) * (1 - offset))
    if buy_price is None:
        result.add("place buy", False, f"no position in {ticker} to price from; pass --buy-price")
    else:
        _resting_order(client, result, "buy", ticker, quantity, buy_price)

    if held < quantity:
        result.add(
            "place sell",
            False,
            f"skipped: need {quantity:g} {ticker} on demo to test a sell, hold {held:g}"
            " (buy some on demo first, or use --fill)",
        )
        return result
    if sell_price is None and current is not None:
        sell_price = _round_price(float(current) * (1 + offset))
    if sell_price is None:
        result.add("place sell", False, "no current price; pass --sell-price")
    else:
        _resting_order(client, result, "sell", ticker, quantity, sell_price)
    return result
