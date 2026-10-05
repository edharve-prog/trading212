"""Thin client for the Trading212 public API (v0, beta).

Docs: https://docs.trading212.com/api

- Auth is HTTP Basic with API_KEY:API_SECRET.
- Sells are negative quantities.
- There is no bracket order and no price data; exits and prices are handled elsewhere.
- Rate limits are per account. Each endpoint is paced client-side, and HTTP 429 responses
  are retried after the time given in the x-ratelimit-reset header.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

log = logging.getLogger(__name__)

BASE_URLS = {
    "demo": "https://demo.trading212.com/api/v0",
    "live": "https://live.trading212.com/api/v0",
}

TimeValidity = Literal["DAY", "GOOD_TILL_CANCEL"]

# Called after every order placement or cancel (dry run included) with
# (kind, payload, result). Used to journal orders.
OrderListener = Callable[[str, dict[str, Any], Any], None]

# Minimum seconds between calls to each endpoint, taken from the documented limits.
# Order placement limits are not on the overview page; these are conservative guesses
# to be checked against the per-endpoint reference on the demo account.
MIN_INTERVAL: dict[str, float] = {
    "account_summary": 5.0,
    "positions": 1.0,
    "orders": 5.0,
    "order": 1.0,
    "cancel": 1.2,
    "market": 1.2,
    "limit": 2.0,
    "stop": 2.0,
    "stop_limit": 2.0,
    "instruments": 50.0,
    "exchanges": 30.0,
    "history": 3.0,
}


class T212Error(Exception):
    def __init__(self, status: int, body: str, path: str):
        super().__init__(f"Trading212 {path} returned HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body
        self.path = path


class LiveTradingDisabled(Exception):
    pass


@dataclass
class DryRunOrder:
    """Returned instead of a real order when dry_run is on."""

    kind: str
    payload: dict[str, Any]


class T212Client:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        environment: Literal["demo", "live"] = "demo",
        *,
        allow_live: bool = False,
        dry_run: bool = False,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        on_order: OrderListener | None = None,
    ):
        if environment not in BASE_URLS:
            raise ValueError(f"environment must be 'demo' or 'live', got {environment!r}")
        self.environment = environment
        self.allow_live = allow_live
        self.dry_run = dry_run
        self.max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._wall_clock = wall_clock
        self._on_order = on_order
        self._last_call: dict[str, float] = {}
        self._http = httpx.Client(
            base_url=BASE_URLS[environment],
            auth=httpx.BasicAuth(api_key, api_secret),
            timeout=20.0,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> T212Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- plumbing -------------------------------------------------------

    def _pace(self, key: str) -> None:
        interval = MIN_INTERVAL.get(key, 1.0)
        last = self._last_call.get(key)
        if last is not None:
            wait = interval - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_call[key] = self._clock()

    def _retry_after(self, response: httpx.Response, attempt: int) -> float:
        reset = response.headers.get("x-ratelimit-reset")
        if reset:
            try:
                # Documented as a Unix timestamp of when the window resets.
                wait = float(reset) - self._wall_clock()
                if 0 < wait < 120:
                    return wait + 0.25
            except ValueError:
                pass
        return float(2 ** (attempt + 1))

    def _request(self, method: str, path: str, key: str, **kwargs: Any) -> Any:
        for attempt in range(self.max_retries + 1):
            self._pace(key)
            response = self._http.request(method, path, **kwargs)
            if response.status_code == 429 and attempt < self.max_retries:
                wait = self._retry_after(response, attempt)
                log.warning("Rate limited on %s, waiting %.1fs", path, wait)
                self._sleep(wait)
                continue
            if response.status_code >= 500 and attempt < self.max_retries and method == "GET":
                self._sleep(float(2 ** (attempt + 1)))
                continue
            if response.is_error:
                raise T212Error(response.status_code, response.text, path)
            if not response.content:
                return None
            return response.json()
        raise AssertionError("unreachable")

    def _place(self, kind: str, payload: dict[str, Any]) -> Any:
        if self.environment == "live" and not self.allow_live:
            raise LiveTradingDisabled(
                "Refusing to place a live order: set broker.allow_live = true to enable."
            )
        if self.dry_run:
            log.info("DRY RUN %s order: %s", kind, payload)
            result: Any = DryRunOrder(kind=kind, payload=payload)
        else:
            result = self._request("POST", f"/equity/orders/{kind}", kind, json=payload)
        self._notify(kind, payload, result)
        return result

    def _notify(self, kind: str, payload: dict[str, Any], result: Any) -> None:
        if self._on_order is None:
            return
        try:
            self._on_order(kind, payload, result)
        except Exception:  # journaling must never break order handling
            log.exception("order listener failed for %s", kind)

    # ---- account ------------------------------------------------------

    def account_summary(self) -> dict[str, Any]:
        return self._request("GET", "/equity/account/summary", "account_summary")

    def positions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/equity/positions", "positions")

    # ---- orders -------------------------------------------------------

    def orders(self) -> list[dict[str, Any]]:
        return self._request("GET", "/equity/orders", "orders")

    def order(self, order_id: int) -> dict[str, Any]:
        return self._request("GET", f"/equity/orders/{order_id}", "order")

    def cancel_order(self, order_id: int) -> None:
        if self.dry_run:
            log.info("DRY RUN cancel order %s", order_id)
        else:
            self._request("DELETE", f"/equity/orders/{order_id}", "cancel")
        self._notify("cancel", {"orderId": order_id}, None)

    def place_market_order(
        self, ticker: str, quantity: float, *, extended_hours: bool = False
    ) -> Any:
        _check_quantity(quantity)
        payload: dict[str, Any] = {"ticker": ticker, "quantity": quantity}
        if extended_hours:
            payload["extendedHours"] = True
        return self._place("market", payload)

    def place_limit_order(
        self,
        ticker: str,
        quantity: float,
        limit_price: float,
        time_validity: TimeValidity = "DAY",
    ) -> Any:
        _check_quantity(quantity)
        _check_price(limit_price)
        return self._place(
            "limit",
            {
                "ticker": ticker,
                "quantity": quantity,
                "limitPrice": limit_price,
                "timeValidity": time_validity,
            },
        )

    def place_stop_order(
        self,
        ticker: str,
        quantity: float,
        stop_price: float,
        time_validity: TimeValidity = "GOOD_TILL_CANCEL",
    ) -> Any:
        _check_quantity(quantity)
        _check_price(stop_price)
        return self._place(
            "stop",
            {
                "ticker": ticker,
                "quantity": quantity,
                "stopPrice": stop_price,
                "timeValidity": time_validity,
            },
        )

    def place_stop_limit_order(
        self,
        ticker: str,
        quantity: float,
        stop_price: float,
        limit_price: float,
        time_validity: TimeValidity = "GOOD_TILL_CANCEL",
    ) -> Any:
        _check_quantity(quantity)
        _check_price(stop_price)
        _check_price(limit_price)
        return self._place(
            "stop_limit",
            {
                "ticker": ticker,
                "quantity": quantity,
                "stopPrice": stop_price,
                "limitPrice": limit_price,
                "timeValidity": time_validity,
            },
        )

    # ---- metadata and history ------------------------------------------

    def instruments(self) -> list[dict[str, Any]]:
        return self._request("GET", "/equity/metadata/instruments", "instruments")

    def exchanges(self) -> list[dict[str, Any]]:
        return self._request("GET", "/equity/metadata/exchanges", "exchanges")

    def order_history(self, cursor: str | None = None, limit: int = 50) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/equity/history/orders", "history", params=params)


def _check_quantity(quantity: float) -> None:
    if quantity == 0:
        raise ValueError("quantity must be non-zero (negative to sell)")


def _check_price(price: float) -> None:
    if not price > 0:
        raise ValueError(f"price must be positive, got {price}")
