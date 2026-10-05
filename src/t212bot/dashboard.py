"""Read-only local web dashboard: mode, account, positions, open orders, recent runs.

It is a small stdlib HTTP server meant to run on the Pi. API keys stay in the server
process; the browser only receives rendered HTML and JSON. There are no forms and no
endpoints that change anything.

Binding is restricted to loopback or a Tailscale address, and requests are rejected
unless their Host header names this machine, which blocks DNS-rebinding pages in a
browser from reading the dashboard.
"""

from __future__ import annotations

import html
import ipaddress
import json
import logging
import socket
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .broker.t212_client import T212Client
from .config import Settings
from .journal import Journal

log = logging.getLogger(__name__)

TAILSCALE_V4 = ipaddress.ip_network("100.64.0.0/10")
TAILSCALE_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "[::1]"}

POSITION_COLUMNS = [
    "ticker",
    "instrument.ticker",
    "instrument.name",
    "quantity",
    "averagePrice",
    "averagePricePaid",
    "currentPrice",
    "ppl",
    "walletImpact.unrealizedProfitLoss",
    "fxPpl",
]
ORDER_COLUMNS = [
    "id",
    "ticker",
    "instrument.ticker",
    "type",
    "side",
    "status",
    "quantity",
    "filledQuantity",
    "limitPrice",
    "stopPrice",
    "timeValidity",
    "creationTime",
    "createdAt",
]


class BindError(Exception):
    pass


def check_bind_host(host: str) -> None:
    """Allow only loopback and Tailscale addresses."""
    if host == "localhost":
        return
    try:
        addr = ipaddress.ip_address(host)
    except ValueError as exc:
        raise BindError(f"--host must be an IP address or localhost, got {host!r}") from exc
    if addr.is_loopback or addr in TAILSCALE_V4 or addr in TAILSCALE_V6:
        return
    raise BindError(
        f"Refusing to bind {host}: use 127.0.0.1 (with `tailscale serve`) or the Pi's "
        "Tailscale address (100.x.y.z), so the dashboard is not exposed on the LAN."
    )


def host_allowed(host_header: str | None, allowed: set[str]) -> bool:
    if not host_header:
        return False
    host = host_header.strip().lower()
    if host.startswith("["):
        name = host.split("]")[0] + "]"
    else:
        name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return name in allowed or name.endswith(".ts.net")


def default_allowed_hosts(bind_host: str, extra: list[str] | None = None) -> set[str]:
    allowed = set(LOOPBACK_NAMES)
    allowed.add(bind_host.lower())
    hostname = socket.gethostname().lower()
    allowed.update({hostname, hostname.split(".")[0]})
    allowed.update(h.lower() for h in extra or [])
    return allowed


def _flatten(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def render_table(rows: list[dict[str, Any]], preferred: list[str] | None = None) -> str:
    if not rows:
        return '<p class="muted">None.</p>'
    flat = [_flatten(r) for r in rows]
    present = {k for r in flat for k in r}
    columns = [c for c in preferred or [] if c in present]
    if not columns:
        columns = sorted(present)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(r.get(c)))}</td>" for c in columns) + "</tr>"
        for r in flat
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


class DashboardData:
    """Collects what the page shows, caching broker calls to stay within rate limits."""

    def __init__(
        self,
        settings: Settings,
        client_factory: Callable[[], T212Client],
        journal: Journal,
        ttl: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.settings = settings
        self.client_factory = client_factory
        self.journal = journal
        self.ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._cached: dict[str, Any] | None = None
        self._cached_at = 0.0

    def _broker(self) -> dict[str, Any]:
        with self._lock:
            if self._cached is not None and self._clock() - self._cached_at < self.ttl:
                return self._cached
            data: dict[str, Any] = {"error": None}
            try:
                with self.client_factory() as client:
                    data["account"] = client.account_summary()
                    data["positions"] = client.positions()
                    data["orders"] = client.orders()
            except Exception as exc:
                log.warning("dashboard broker fetch failed: %s", exc)
                data["error"] = str(exc)
            data["fetched_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            self._cached, self._cached_at = data, self._clock()
            return data

    def snapshot(self) -> dict[str, Any]:
        broker = self.settings.broker
        try:
            runs = self.journal.recent_runs(20)
            journal_orders = self.journal.recent_orders(20)
        except Exception as exc:
            runs, journal_orders = [], []
            log.warning("dashboard journal read failed: %s", exc)
        return {
            "mode": {
                "environment": broker.environment,
                "dry_run": broker.dry_run,
                "allow_live": broker.allow_live,
            },
            "broker": self._broker(),
            "runs": runs,
            "journal_orders": journal_orders,
        }


def _mode_banner(mode: dict[str, Any]) -> str:
    env = mode["environment"]
    parts = [f'<span class="pill {"live" if env == "live" else "demo"}">{env.upper()}</span>']
    if mode["dry_run"]:
        parts.append('<span class="pill demo">DRY RUN: orders are logged, not sent</span>')
    else:
        parts.append('<span class="pill warn">ORDERS ARE SENT</span>')
    if env == "live":
        parts.append(
            '<span class="pill live">live orders allowed</span>'
            if mode["allow_live"]
            else '<span class="pill demo">live orders blocked</span>'
        )
    return " ".join(parts)


PAGE_CSS = """
:root{--bg:#fafafa;--fg:#1b1b1b;--muted:#6b6b6b;--line:#ddd;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--fg:#eaeaea;--muted:#9a9a9a;
--line:#333;--card:#1c1c1c}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
font:14px/1.4 system-ui,sans-serif}
h1{font-size:20px;margin:0 0 8px}h2{font-size:16px;margin:20px 0 8px}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px;margin-bottom:12px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600}.muted{color:var(--muted)}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-weight:600;font-size:12px}
.demo{background:#d8f0dc;color:#14532d}.live{background:#fbd5d5;color:#7f1d1d}
.warn{background:#fde68a;color:#78350f}.err{color:#b91c1c}
"""


def render_page(snapshot: dict[str, Any]) -> str:
    mode = snapshot["mode"]
    broker = snapshot["broker"]
    sections = [f"<section>{_mode_banner(mode)}"]
    sections.append(
        f'<p class="muted">Broker data fetched {html.escape(broker.get("fetched_at", ""))}'
        "; page refreshes every 60s.</p></section>"
    )
    if broker.get("error"):
        sections.append(
            f'<section><p class="err">Trading212: {html.escape(broker["error"])}</p></section>'
        )
    else:
        account = broker.get("account") or {}
        sections.append(
            "<section><h2>Account</h2>"
            + render_table([{"field": k, "value": v} for k, v in _flatten(account).items()])
            + "</section>"
        )
        sections.append(
            "<section><h2>Positions</h2>"
            + render_table(broker.get("positions") or [], POSITION_COLUMNS)
            + "</section>"
        )
        sections.append(
            "<section><h2>Open orders</h2>"
            + render_table(broker.get("orders") or [], ORDER_COLUMNS)
            + "</section>"
        )
    sections.append("<section><h2>Recent runs</h2>" + render_table(snapshot["runs"]) + "</section>")
    sections.append(
        "<section><h2>Orders placed by the bot</h2>"
        + render_table(snapshot["journal_orders"])
        + "</section>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="refresh" content="60">'
        f"<title>t212bot</title><style>{PAGE_CSS}</style></head><body>"
        "<h1>t212bot</h1>" + "".join(sections) + "</body></html>"
    )


def make_handler(data: DashboardData, allowed_hosts: set[str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "t212bot-dashboard"

        def _send(self, status: int, body: str, content_type: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            if not host_allowed(self.headers.get("Host"), allowed_hosts):
                self._send(421, "Unrecognised Host header\n", "text/plain; charset=utf-8")
                return
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                self._send(200, "ok\n", "text/plain; charset=utf-8")
            elif path == "/":
                self._send(200, render_page(data.snapshot()), "text/html; charset=utf-8")
            elif path == "/api/status":
                body = json.dumps(data.snapshot(), indent=2, default=str)
                self._send(200, body, "application/json")
            else:
                self._send(404, "Not found\n", "text/plain; charset=utf-8")

        def _read_only(self) -> None:
            self._send(405, "Read-only dashboard\n", "text/plain; charset=utf-8")

        do_POST = do_PUT = do_DELETE = do_PATCH = _read_only  # noqa: N815

        def log_message(self, format: str, *args: Any) -> None:
            log.info("%s %s", self.address_string(), format % args)

    return Handler


class _V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def serve(
    data: DashboardData,
    host: str = "127.0.0.1",
    port: int = 8212,
    extra_hosts: list[str] | None = None,
) -> None:
    check_bind_host(host)
    handler = make_handler(data, default_allowed_hosts(host, extra_hosts))
    server_cls = _V6Server if ":" in host else ThreadingHTTPServer
    with server_cls((host, port), handler) as server:
        log.info("Dashboard on http://%s:%d (read-only)", host, port)
        server.serve_forever()
