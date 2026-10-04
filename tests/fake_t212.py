"""In-memory stand-in for the Trading212 demo API, served through httpx.MockTransport."""

import json

import httpx


class FakeT212:
    def __init__(self, positions=None, fill_market=True):
        self.positions = positions or []
        self.fill_market = fill_market
        self.pending = {}
        self.placed = []
        self.cancelled = []
        self.next_id = 100
        self.hosts = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.hosts.add(request.url.host)
        path = request.url.path.removeprefix("/api/v0")
        if request.method == "GET" and path == "/equity/positions":
            return httpx.Response(200, json=self.positions)
        if request.method == "GET" and path == "/equity/account/summary":
            return httpx.Response(200, json={"currency": "GBP", "cash": {"free": 1000.0}})
        if request.method == "GET" and path == "/equity/orders":
            return httpx.Response(200, json=list(self.pending.values()))
        if request.method == "POST" and path.startswith("/equity/orders/"):
            kind = path.rsplit("/", 1)[1]
            body = json.loads(request.content)
            self.next_id += 1
            order = {"id": self.next_id, "type": kind.upper(), "status": "NEW", **body}
            self.placed.append((kind, body))
            if not (kind == "market" and self.fill_market):
                self.pending[self.next_id] = order
            return httpx.Response(200, json=order)
        if request.method == "DELETE" and path.startswith("/equity/orders/"):
            order_id = int(path.rsplit("/", 1)[1])
            if order_id not in self.pending:
                return httpx.Response(404, text="not found")
            del self.pending[order_id]
            self.cancelled.append(order_id)
            return httpx.Response(200)
        return httpx.Response(404, text=f"unhandled {request.method} {path}")
