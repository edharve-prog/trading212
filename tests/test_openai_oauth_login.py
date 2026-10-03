"""Login, refresh and streaming tests against a real loopback callback server.

OpenAI's HTTP endpoints are mocked; the callback server and the browser's requests to it
are real, so these exercise the wait loop, stray requests and timeouts end to end.
"""

import contextlib
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from t212bot.llm.openai_oauth import (
    DYNAMIC_CLIENT_ID,
    ISSUER,
    LLMError,
    OAuthError,
    OpenAIOAuthClient,
    _Profile,
)

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
JWK = {**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(KEY.public_key())), "kid": "k1"}
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def id_token(client_id, nonce, sub="sub-1"):
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "sub": sub,
        "aud": client_id,
        "email": "user@example.com",
        "nonce": nonce,
        "iat": now,
        "exp": now + 3600,
    }
    return jwt.encode(claims, KEY, algorithm="RS256", headers={"kid": "k1"})


class FakeOpenAI:
    """Mocked token and JWKS endpoints that remember what the client sent."""

    def __init__(self, token_status=200, token_error=None):
        self.token_status = token_status
        self.token_error = token_error
        self.token_requests = []
        self.nonce = None
        self.client_id = None

    def __call__(self, request):
        if request.url.path.endswith("jwks.json"):
            return httpx.Response(200, json={"keys": [JWK]})
        if request.url.path.endswith("/oauth/token"):
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            self.token_requests.append(form)
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": self.token_error})
            return httpx.Response(
                200,
                json={
                    "access_token": "access-1",
                    "refresh_token": "refresh-1",
                    "id_token": id_token(form["client_id"], self.nonce),
                    "expires_in": 3600,
                    "scope": "openid offline_access chatgpt.tokens.use.direct",
                },
            )
        return httpx.Response(404)


def browser(fake, requests_to_send):
    """A browser_opener that fires the given callback requests from another thread.

    Each item maps the authorization URL's query to the extra query for one request,
    or is a bare path string for a stray request.
    """
    seen = {}

    def opener(url):
        query = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        seen.update(query)
        fake.nonce = query["nonce"]
        redirect = query["redirect_uri"]
        base = redirect.rsplit("/auth/callback", 1)[0]

        def run():
            for item in requests_to_send:
                target = (
                    base + item if isinstance(item, str) else f"{redirect}?{urlencode(item(query))}"
                )
                with contextlib.suppress(urllib.error.HTTPError):
                    NO_PROXY.open(target, timeout=5).read()

        threading.Thread(target=run, daemon=True).start()
        return True

    return opener, seen


def client(tmp_path, fake, **kwargs):
    return OpenAIOAuthClient(
        credential_path=tmp_path / "oauth.json",
        transport=httpx.MockTransport(fake),
        **kwargs,
    )


def good(client_id="oaiapp_new"):
    return lambda q: {"state": q["state"], "code": "code-1", "client_id": client_id}


def saved_profile(tmp_path, **overrides):
    values = dict(
        issuer=ISSUER,
        subject="sub-1",
        client_id="oaiapp_saved",
        ext_agent_host_id="urn:uuid:host",
        email="user@example.com",
        id_token="secret-id-token",
        access_token="access-old",
        refresh_token="refresh-old",
        expires_in=3600,
        scopes=["chatgpt.tokens.use.direct"],
        saved_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
    )
    values.update(overrides)
    return _Profile(**values)


def test_new_registration_survives_stray_and_wrong_state_requests(tmp_path):
    fake = FakeOpenAI()
    opener, seen = browser(
        fake,
        [
            "/favicon.ico",
            lambda q: {"state": "not-the-state", "code": "evil", "client_id": "oaiapp_evil"},
            good(),
        ],
    )
    c = client(tmp_path, fake, callback_timeout=10)
    status = c.login(callback_port=0, browser_opener=opener)

    assert status.signed_in and status.plan_enabled
    assert status.client_id == "oaiapp_new"
    assert seen["client_id"] == DYNAMIC_CLIENT_ID
    assert len(fake.token_requests) == 1
    assert fake.token_requests[0]["code"] == "code-1"
    stored = json.loads((tmp_path / "oauth.json").read_text())
    assert "pending_client_id" not in stored


def test_times_out_after_callback_timeout_not_http_timeout(tmp_path):
    fake = FakeOpenAI()
    opener, _ = browser(fake, ["/favicon.ico"])
    c = client(tmp_path, fake, callback_timeout=0.5, http_timeout=0.01)
    started = time.monotonic()
    with pytest.raises(OAuthError, match="Timed out"):
        c.login(callback_port=0, browser_opener=opener)
    assert time.monotonic() - started >= 0.5
    assert fake.token_requests == []


def test_access_denied_stops_without_code_exchange(tmp_path):
    fake = FakeOpenAI()
    opener, _ = browser(fake, [lambda q: {"state": q["state"], "error": "access_denied"}])
    with pytest.raises(OAuthError, match="access_denied"):
        client(tmp_path, fake, callback_timeout=10).login(callback_port=0, browser_opener=opener)
    assert fake.token_requests == []


def test_new_registration_requires_issued_client_id(tmp_path):
    fake = FakeOpenAI()
    opener, _ = browser(fake, [good(client_id=DYNAMIC_CLIENT_ID)])
    with pytest.raises(OAuthError, match="issued client_id"):
        client(tmp_path, fake, callback_timeout=10).login(callback_port=0, browser_opener=opener)


def test_returning_login_rejects_a_different_client_id(tmp_path):
    fake = FakeOpenAI()
    c = client(tmp_path, fake, callback_timeout=10)
    p = saved_profile(tmp_path)
    c._save_state(p.ext_agent_host_id, p)
    opener, seen = browser(fake, [good(client_id="oaiapp_other")])
    with pytest.raises(OAuthError, match="different client_id"):
        c.login(callback_port=0, browser_opener=opener)
    assert seen["client_id"] == "oaiapp_saved"
    assert seen["id_token_hint"] == "secret-id-token"  # straight to the browser only


def test_invalid_grant_keeps_issued_client_id_for_the_retry(tmp_path):
    fake = FakeOpenAI(token_status=400, token_error="invalid_grant")
    opener, _ = browser(fake, [good()])
    c = client(tmp_path, fake, callback_timeout=10)
    with pytest.raises(OAuthError, match="invalid_grant"):
        c.login(callback_port=0, browser_opener=opener)
    stored = json.loads((tmp_path / "oauth.json").read_text())
    assert stored["pending_client_id"] == "oaiapp_new"

    fake.token_status = 200
    opener, seen = browser(fake, [lambda q: {"state": q["state"], "code": "code-2"}])
    status = c.login(callback_port=0, browser_opener=opener)
    assert seen["client_id"] == "oaiapp_new"
    assert fake.token_requests[-1]["client_id"] == "oaiapp_new"
    assert status.client_id == "oaiapp_new"


def test_headless_url_never_contains_the_id_token(tmp_path):
    fake = FakeOpenAI()
    c = client(tmp_path, fake, callback_timeout=10)
    p = saved_profile(tmp_path)
    c._save_state(p.ext_agent_host_id, p)
    opener, _ = browser(fake, [good(client_id="oaiapp_saved")])
    shown = []

    def show(url):
        shown.append(url)
        opener(url)  # the user pastes the URL into a browser elsewhere

    status = c.login(callback_port=0, open_browser=False, url_callback=show)
    assert status.signed_in
    assert "id_token_hint" not in shown[0]
    assert "secret-id-token" not in shown[0]


# ---- refresh ---------------------------------------------------------------


def refresh_client(tmp_path, responses, **kwargs):
    calls = []
    sleeps = []

    def handler(request):
        calls.append(request)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    c = OpenAIOAuthClient(
        credential_path=tmp_path / "oauth.json",
        transport=httpx.MockTransport(handler),
        now=lambda: datetime(2026, 1, 1, 0, 59, tzinfo=UTC),
        sleep=sleeps.append,
        **kwargs,
    )
    p = saved_profile(tmp_path)
    c._save_state(p.ext_agent_host_id, p)
    return c, calls, sleeps


ROTATED = httpx.Response(
    200,
    json={
        "access_token": "access-new",
        "refresh_token": "refresh-new",
        "expires_in": 3600,
        "scope": "chatgpt.tokens.use.direct",
    },
)


def test_refresh_retries_transient_failures_with_backoff(tmp_path):
    c, calls, sleeps = refresh_client(
        tmp_path, [httpx.ConnectError("down"), httpx.Response(503), ROTATED]
    )
    assert c.access_token() == "access-new"
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_refresh_gives_up_on_transient_failures_but_keeps_tokens(tmp_path):
    c, calls, _ = refresh_client(tmp_path, [httpx.Response(503)] * 3)
    with pytest.raises(OAuthError) as err:
        c.access_token()
    assert err.value.transient
    stored = json.loads((tmp_path / "oauth.json").read_text())["profile"]
    assert stored["refresh_token"] == "refresh-old"


def test_terminal_refresh_error_clears_the_session(tmp_path):
    c, calls, sleeps = refresh_client(
        tmp_path, [httpx.Response(400, json={"error": "invalid_grant"})]
    )
    with pytest.raises(OAuthError, match="llm-login"):
        c.access_token()
    assert len(calls) == 1 and sleeps == []
    stored = json.loads((tmp_path / "oauth.json").read_text())["profile"]
    assert stored["refresh_token"] is None and stored["access_token"] is None
    assert stored["client_id"] == "oaiapp_saved"  # registration is kept
    assert not c.status().signed_in


def test_refresh_reuses_token_another_process_refreshed(tmp_path):
    c, calls, _ = refresh_client(tmp_path, [])
    original = c._load_profile
    first = {"done": False}

    def stale_then_fresh():
        profile = original()
        if not first["done"]:
            first["done"] = True
            # Another process refreshes while this one waits for the lock.
            fresh = saved_profile(
                tmp_path,
                access_token="from-other-process",
                refresh_token="refresh-other",
                saved_at=datetime(2026, 1, 1, 0, 58, tzinfo=UTC).isoformat(),
            )
            c._save_state(fresh.ext_agent_host_id, fresh)
        return profile

    c._load_profile = stale_then_fresh
    assert c.access_token() == "from-other-process"
    assert calls == []


# ---- streaming diagnostics -------------------------------------------------


def stream_client(tmp_path, handler):
    c = OpenAIOAuthClient(
        credential_path=tmp_path / "oauth.json",
        transport=httpx.MockTransport(handler),
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    p = saved_profile(tmp_path)
    c._save_state(p.ext_agent_host_id, p)
    return c


def test_response_failed_keeps_message_param_and_request_id(tmp_path):
    event = {
        "type": "response.failed",
        "response": {"error": {"code": "invalid_value", "message": "bad model", "param": "model"}},
    }
    c = stream_client(
        tmp_path,
        lambda r: httpx.Response(
            200, text=f"data: {json.dumps(event)}\n\n", headers={"x-request-id": "req_123"}
        ),
    )
    with pytest.raises(LLMError) as err:
        c.respond("hi", model="m")
    message = str(err.value)
    assert "invalid_value" in message and "bad model" in message
    assert "param=model" in message and "req_123" in message


def test_transport_and_malformed_stream_errors_become_llm_errors(tmp_path):
    def down(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMError, match="slow"):
        stream_client(tmp_path, down).respond("hi", model="m")

    c = stream_client(tmp_path, lambda r: httpx.Response(200, text="data: {not json\n\n"))
    with pytest.raises(LLMError, match="malformed"):
        c.respond("hi", model="m")
