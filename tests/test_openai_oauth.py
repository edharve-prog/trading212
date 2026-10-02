import json
import os
from datetime import UTC, datetime

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from t212bot.llm.openai_oauth import LLMError, OAuthError, OpenAIOAuthClient, _Profile


def profile(now: datetime, **overrides):
    values = dict(
        issuer="https://auth.openai.com",
        subject="sub-1",
        client_id="oaiapp_test",
        ext_agent_host_id="urn:uuid:11111111-1111-4111-8111-111111111111",
        email="user@example.com",
        id_token="id",
        access_token="access-old",
        refresh_token="refresh-old",
        token_type="Bearer",
        expires_in=3600,
        scopes=["chatgpt.tokens.use.direct", "offline_access", "openid"],
        saved_at=now.isoformat(),
    )
    values.update(overrides)
    return _Profile(**values)


def test_state_file_is_owner_only_and_host_id_is_stable(tmp_path):
    path = tmp_path / "oauth.json"
    client = OpenAIOAuthClient(credential_path=path)
    host1, saved = client._load_state()
    host2, saved2 = client._load_state()
    assert host1 == host2
    assert host1.startswith("urn:uuid:")
    assert saved is None and saved2 is None
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


def test_refresh_rotates_refresh_token_atomically(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    seen = []

    def handler(request):
        seen.append(request)
        assert request.url.path == "/api/accounts/oauth/token"
        body = request.content.decode()
        assert "grant_type=refresh_token" in body
        assert "refresh_token=refresh-old" in body
        return httpx.Response(
            200,
            json={
                "access_token": "access-new",
                "refresh_token": "refresh-new",
                "expires_in": 3600,
                "scope": "chatgpt.tokens.use.direct offline_access openid",
            },
        )

    path = tmp_path / "oauth.json"
    client = OpenAIOAuthClient(
        credential_path=path, transport=httpx.MockTransport(handler), now=lambda: now
    )
    client._save_state(profile(now).ext_agent_host_id, profile(now))
    updated = client.refresh()
    assert updated.access_token == "access-new"
    assert updated.refresh_token == "refresh-new"
    stored = json.loads(path.read_text())
    assert stored["profile"]["refresh_token"] == "refresh-new"
    assert len(seen) == 1


def test_access_token_refreshes_near_expiry(tmp_path):
    saved = datetime(2026, 1, 1, tzinfo=UTC)
    now = datetime(2026, 1, 1, 0, 59, tzinfo=UTC)

    def handler(request):
        return httpx.Response(
            200,
            json={
                "access_token": "fresh",
                "refresh_token": "rotated",
                "expires_in": 3600,
                "scope": "chatgpt.tokens.use.direct offline_access",
            },
        )

    client = OpenAIOAuthClient(
        credential_path=tmp_path / "oauth.json",
        transport=httpx.MockTransport(handler),
        now=lambda: now,
    )
    client._save_state(profile(saved).ext_agent_host_id, profile(saved))
    assert client.access_token() == "fresh"


def test_list_models_uses_account_catalog(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def handler(request):
        assert request.headers["Authorization"] == "Bearer access-old"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "gpt-a", "display_name": "GPT A", "visibility": "list"},
                    {"slug": "hidden", "display_name": "Hidden", "visibility": "hide"},
                ]
            },
        )

    client = OpenAIOAuthClient(
        credential_path=tmp_path / "oauth.json",
        transport=httpx.MockTransport(handler),
        now=lambda: now,
    )
    client._save_state(profile(now).ext_agent_host_id, profile(now))
    models = client.list_models()
    assert [(m.slug, m.display_name) for m in models] == [("gpt-a", "GPT A")]


def test_respond_requires_completed_stream_and_collects_text(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        body = (
            'data: {"type":"response.output_text.delta","delta":"hello "}\n\n'
            'data: {"type":"response.output_text.delta","delta":"world"}\n\n'
            'data: {"type":"response.completed","response":{"id":"r1"}}\n\n'
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = OpenAIOAuthClient(
        credential_path=tmp_path / "oauth.json",
        transport=httpx.MockTransport(handler),
        now=lambda: now,
    )
    client._save_state(profile(now).ext_agent_host_id, profile(now))
    assert client.respond("Hi", model="gpt-test", instructions="Be brief") == "hello world"
    assert seen["body"] == {
        "model": "gpt-test",
        "input": [{"role": "user", "content": "Hi"}],
        "store": False,
        "stream": True,
        "instructions": "Be brief",
    }


def test_respond_surfaces_stream_failure(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def handler(request):
        body = 'data: {"type":"response.failed","response":{"error":{"code":"limit"}}}\n\n'
        return httpx.Response(200, text=body)

    client = OpenAIOAuthClient(
        credential_path=tmp_path / "oauth.json",
        transport=httpx.MockTransport(handler),
        now=lambda: now,
    )
    client._save_state(profile(now).ext_agent_host_id, profile(now))
    with pytest.raises(LLMError, match="limit"):
        client.respond("Hi", model="gpt-test")


def test_verify_id_token_checks_signature_audience_issuer_and_nonce(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    public_jwk["kid"] = "kid-1"
    now = int(datetime.now(UTC).timestamp())
    token = jwt.encode(
        {
            "iss": "https://auth.openai.com",
            "sub": "subject",
            "aud": "oaiapp_test",
            "email": "user@example.com",
            "nonce": "nonce-1",
            "iat": now,
            "nbf": now - 1,
            "exp": now + 3600,
        },
        key,
        algorithm="RS256",
        headers={"kid": "kid-1"},
    )

    def handler(request):
        return httpx.Response(200, json={"keys": [public_jwk]})

    client = OpenAIOAuthClient(
        credential_path=tmp_path / "oauth.json", transport=httpx.MockTransport(handler)
    )
    claims = client._verify_id_token(token, "oaiapp_test", "nonce-1")
    assert claims["sub"] == "subject"
    with pytest.raises(OAuthError, match="nonce"):
        client._verify_id_token(token, "oaiapp_test", "wrong")


def test_plan_scope_is_required_for_inference(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    client = OpenAIOAuthClient(credential_path=tmp_path / "oauth.json", now=lambda: now)
    p = profile(now, scopes=["openid", "offline_access"])
    client._save_state(p.ext_agent_host_id, p)
    with pytest.raises(OAuthError, match="plan usage"):
        client.access_token()
