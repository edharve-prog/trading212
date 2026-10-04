import http.client
import threading

import httpx
import pytest

from t212bot.broker.t212_client import T212Client
from t212bot.config import Settings
from t212bot.dashboard import (
    BindError,
    DashboardData,
    check_bind_host,
    default_allowed_hosts,
    host_allowed,
    make_handler,
    render_page,
)
from t212bot.journal import Journal

from .fake_t212 import FakeT212


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "100.101.102.103"])
def test_bind_allows_loopback_and_tailscale(host):
    check_bind_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::", "pi.local"])
def test_bind_refuses_lan_and_wildcard(host):
    with pytest.raises(BindError):
        check_bind_host(host)


def test_host_header_check():
    allowed = default_allowed_hosts("127.0.0.1", ["mypi"])
    assert host_allowed("127.0.0.1:8212", allowed)
    assert host_allowed("localhost", allowed)
    assert host_allowed("[::1]:8212", allowed)
    assert host_allowed("mypi:8212", allowed)
    assert host_allowed("mypi.tail1234.ts.net", allowed)
    assert not host_allowed("evil.example.com:8212", allowed)
    assert not host_allowed(None, allowed)


def make_data(tmp_path, fake, clock=None):
    def factory():
        return T212Client(
            "k", "s", transport=httpx.MockTransport(fake.handler), sleep=lambda s: None
        )

    kwargs = {"clock": clock} if clock else {}
    return DashboardData(Settings(), factory, Journal(tmp_path / "j.sqlite"), ttl=30, **kwargs)


def test_snapshot_caches_broker_calls(tmp_path):
    fake = FakeT212(positions=[{"ticker": "AAPL_US_EQ", "quantity": 1}])
    calls = []
    real = fake.handler
    fake.handler = lambda r: calls.append(r.url.path) or real(r)
    now = [0.0]
    data = make_data(tmp_path, fake, clock=lambda: now[0])
    data.snapshot()
    data.snapshot()
    assert len(calls) == 3
    now[0] = 31
    data.snapshot()
    assert len(calls) == 6


def test_page_shows_mode_and_escapes_content(tmp_path):
    fake = FakeT212(positions=[{"ticker": "<script>alert(1)</script>", "quantity": 1}])
    page = render_page(make_data(tmp_path, fake).snapshot())
    assert "DRY RUN" in page and "DEMO" in page
    assert "<script>alert" not in page and "&lt;script&gt;" in page
    assert "<form" not in page


def test_broker_error_is_shown_not_raised(tmp_path):
    fake = FakeT212()
    fake.handler = lambda r: httpx.Response(401, text="bad key")
    snap = make_data(tmp_path, fake).snapshot()
    assert "401" in snap["broker"]["error"]
    assert "bad key" in render_page(snap)


@pytest.fixture
def server(tmp_path):
    from http.server import ThreadingHTTPServer

    data = make_data(tmp_path, FakeT212())
    srv = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(data, default_allowed_hosts("127.0.0.1"))
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def request(port, method, path, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest(method, path, skip_host=True)
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    conn.endheaders()
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    return resp, body


def test_http_routes(server):
    resp, body = request(server, "GET", "/")
    assert resp.status == 200 and "t212bot" in body
    assert "frame-ancestors 'none'" in resp.getheader("Content-Security-Policy")
    resp, body = request(server, "GET", "/api/status")
    assert resp.status == 200 and '"dry_run": true' in body
    assert request(server, "GET", "/healthz")[0].status == 200
    assert request(server, "GET", "/nope")[0].status == 404
    assert request(server, "POST", "/")[0].status == 405
    assert request(server, "GET", "/", host="attacker.example:80")[0].status == 421
