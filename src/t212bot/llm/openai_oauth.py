"""Sign in with ChatGPT OAuth and Responses API client.

This implements OpenAI's open-source token-sharing flow. Credentials are stored
outside the repository and are never read from Codex's private auth files.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
import tempfile
import threading
import time
import uuid
import webbrowser
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import jwt

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
API_BASE_URL = "https://api.openai.com/v1"
RESOURCE = API_BASE_URL
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
CALLBACK_PATH = "/auth/callback"


class OAuthError(RuntimeError):
    """OAuth registration, validation, refresh, or revocation failed."""


class LLMError(RuntimeError):
    """An OpenAI model discovery or Responses API request failed."""


@dataclass(frozen=True)
class AuthStatus:
    signed_in: bool
    email: str | None = None
    plan_enabled: bool = False
    client_id: str | None = None
    expires_at: datetime | None = None


@dataclass(frozen=True)
class ModelInfo:
    slug: str
    display_name: str


@dataclass
class _Profile:
    issuer: str
    subject: str
    client_id: str
    ext_agent_host_id: str
    email: str | None = None
    id_token: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    token_type: str = "Bearer"
    expires_in: int = 0
    scopes: list[str] | None = None
    saved_at: str | None = None
    earliest_refresh_at: Any = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> _Profile:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: value[k] for k in allowed if k in value})

    def as_dict(self) -> dict[str, Any]:
        return {
            "issuer": self.issuer,
            "subject": self.subject,
            "client_id": self.client_id,
            "ext_agent_host_id": self.ext_agent_host_id,
            "email": self.email,
            "id_token": self.id_token,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "scopes": self.scopes or [],
            "saved_at": self.saved_at,
            "earliest_refresh_at": self.earliest_refresh_at,
        }


class OpenAIOAuthClient:
    """OpenAI OAuth client for locally hosted/open-source applications."""

    def __init__(
        self,
        *,
        credential_path: Path | str | None = None,
        agent_name: str = "t212bot",
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        default = Path.home() / ".config" / "t212bot" / "openai_oauth.json"
        self.credential_path = Path(credential_path).expanduser() if credential_path else default
        self.agent_name = agent_name
        self.timeout = timeout
        self._now = now
        self._http = httpx.Client(timeout=timeout, transport=transport)
        self._thread_lock = threading.Lock()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OpenAIOAuthClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def status(self) -> AuthStatus:
        profile = self._load_profile()
        if profile is None or not profile.access_token:
            return AuthStatus(signed_in=False, email=profile.email if profile else None)
        return AuthStatus(
            signed_in=True,
            email=profile.email,
            plan_enabled=DIRECT_SCOPE in (profile.scopes or []),
            client_id=profile.client_id,
            expires_at=self._expires_at(profile),
        )

    def login(
        self,
        *,
        callback_port: int = 1455,
        open_browser: bool = True,
        browser_opener: Callable[[str], Any] = webbrowser.open,
        url_callback: Callable[[str], None] | None = None,
    ) -> AuthStatus:
        """Run a loopback OAuth flow and persist the validated credentials."""
        host_id, existing = self._load_state()
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
        callback: dict[str, str] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(handler_self) -> None:  # noqa: N802
                parsed = urlparse(handler_self.path)
                if parsed.path != CALLBACK_PATH:
                    handler_self.send_response(404)
                    handler_self.end_headers()
                    return
                values = parse_qs(parsed.query)
                for name, items in values.items():
                    if items:
                        callback[name] = items[0]
                handler_self.send_response(200)
                handler_self.send_header("Content-Type", "text/html; charset=utf-8")
                handler_self.end_headers()
                handler_self.wfile.write(
                    b"<html><body><h1>Connected</h1><p>You can close this window.</p></body></html>"
                )

            def log_message(self, format: str, *args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", callback_port), Handler)
        actual_port = server.server_address[1]
        redirect_uri = f"http://127.0.0.1:{actual_port}{CALLBACK_PATH}"
        pending_client_id = existing.client_id if existing else DYNAMIC_CLIENT_ID
        params = {
            "client_id": pending_client_id,
            "ext_agent_host_id": host_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
        if existing:
            if existing.id_token:
                params["id_token_hint"] = existing.id_token
            if existing.email:
                params["login_hint"] = existing.email
        else:
            params["agent_name_hint"] = self.agent_name

        authorization_url = f"{AUTHORIZE_URL}?{urlencode(params)}"
        if url_callback:
            url_callback(authorization_url)
        if open_browser and not browser_opener(authorization_url):
            server.server_close()
            raise OAuthError(
                "Could not open a browser; rerun with --no-browser and open the URL manually"
            )

        server.timeout = self.timeout
        try:
            server.handle_request()
        finally:
            server.server_close()
        if not callback:
            raise OAuthError("Timed out waiting for the OpenAI OAuth callback")
        if callback.get("state") != state:
            raise OAuthError("OAuth callback state did not match the pending sign-in")
        if callback.get("error"):
            raise OAuthError(f"OpenAI authorization failed: {callback['error']}")
        code = callback.get("code")
        if not code:
            raise OAuthError("OAuth callback did not include an authorization code")

        returned_client_id = callback.get("client_id")
        if existing:
            if returned_client_id and returned_client_id != existing.client_id:
                raise OAuthError("OAuth callback returned a different client_id for the saved account")
            issued_client_id = existing.client_id
        else:
            if not returned_client_id or returned_client_id == DYNAMIC_CLIENT_ID:
                raise OAuthError("OpenAI did not return an issued client_id for the new registration")
            issued_client_id = returned_client_id

        token_response = self._post_token(
            {
                "grant_type": "authorization_code",
                "client_id": issued_client_id,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "resource": RESOURCE,
            }
        )
        id_token = _require_string(token_response, "id_token")
        claims = self._verify_id_token(id_token, issued_client_id, nonce)
        if existing and (claims["iss"], claims["sub"]) != (existing.issuer, existing.subject):
            raise OAuthError("The authorized ChatGPT account does not match the saved registration")

        scopes = str(token_response.get("scope", "")).split()
        profile = _Profile(
            issuer=str(claims["iss"]),
            subject=str(claims["sub"]),
            client_id=issued_client_id,
            ext_agent_host_id=host_id,
            email=str(claims["email"]) if claims.get("email") else None,
            id_token=id_token,
            access_token=_require_string(token_response, "access_token"),
            refresh_token=_require_string(token_response, "refresh_token"),
            token_type=str(token_response.get("token_type", "Bearer")),
            expires_in=int(token_response.get("expires_in", 3600)),
            scopes=scopes,
            saved_at=self._now().isoformat(),
            earliest_refresh_at=token_response.get("earliest_refresh_at"),
        )
        self._save_state(host_id, profile)
        return self.status()

    def access_token(self) -> str:
        profile = self._load_profile()
        if profile is None or not profile.access_token or not profile.refresh_token:
            raise OAuthError("Not signed in. Run `t212bot llm-login` first.")
        if DIRECT_SCOPE not in (profile.scopes or []):
            raise OAuthError("ChatGPT plan usage was not authorized for this connection")
        if self._needs_refresh(profile):
            profile = self.refresh()
        if not profile.access_token:
            raise OAuthError("Saved OpenAI credentials do not contain an access token")
        return profile.access_token

    def refresh(self, *, force: bool = True) -> _Profile:
        """Refresh credentials, atomically preserving the rotating refresh token."""
        with self._thread_lock, self._refresh_lock():
            profile = self._load_profile()
            if profile is None or not profile.refresh_token:
                raise OAuthError("No renewable OpenAI session is stored")
            if not force and not self._needs_refresh(profile):
                return profile
            data = self._post_token(
                {
                    "grant_type": "refresh_token",
                    "client_id": profile.client_id,
                    "refresh_token": profile.refresh_token,
                    "resource": RESOURCE,
                }
            )
            new_access = _require_string(data, "access_token")
            new_refresh = _require_string(data, "refresh_token")
            profile.access_token = new_access
            profile.refresh_token = new_refresh
            profile.id_token = str(data.get("id_token") or profile.id_token or "") or None
            profile.token_type = str(data.get("token_type", profile.token_type))
            profile.expires_in = int(data.get("expires_in", 3600))
            if data.get("scope"):
                profile.scopes = str(data["scope"]).split()
            profile.saved_at = self._now().isoformat()
            profile.earliest_refresh_at = data.get("earliest_refresh_at")
            self._save_state(profile.ext_agent_host_id, profile)
            return profile

    def logout(self) -> bool:
        """Attempt remote refresh-token revocation and clear local tokens regardless."""
        host_id, profile = self._load_state()
        if profile is None:
            return True
        revoked = True
        if profile.refresh_token:
            try:
                discovery = self._http.get(DISCOVERY_URL)
                discovery.raise_for_status()
                endpoint = discovery.json().get("revocation_endpoint")
                if not endpoint:
                    raise OAuthError("OpenAI discovery document has no revocation_endpoint")
                response = self._http.post(
                    endpoint,
                    data={
                        "token": profile.refresh_token,
                        "token_type_hint": "refresh_token",
                        "client_id": profile.client_id,
                    },
                )
                revoked = response.status_code == 200
            except (httpx.HTTPError, OAuthError, ValueError):
                revoked = False
        profile.id_token = None
        profile.access_token = None
        profile.refresh_token = None
        profile.expires_in = 0
        profile.scopes = []
        profile.saved_at = None
        self._save_state(host_id, profile)
        return revoked

    def list_models(self) -> list[ModelInfo]:
        response = self._http.get(
            f"{API_BASE_URL}/models",
            headers={"Authorization": f"Bearer {self.access_token()}"},
        )
        if response.is_error:
            raise LLMError(_http_error("OpenAI model discovery failed", response))
        body = response.json()
        models = body.get("models", body.get("data", []))
        result = []
        for item in models:
            if item.get("visibility", "list") != "list":
                continue
            slug = item.get("slug") or item.get("id")
            if slug:
                result.append(ModelInfo(str(slug), str(item.get("display_name") or slug)))
        return result

    def respond(self, prompt: str, *, model: str, instructions: str | None = None) -> str:
        """Send one stateless, streamed Responses request and return its text output."""
        payload: dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "store": False,
            "stream": True,
        }
        if instructions:
            payload["instructions"] = instructions
        headers = {
            "Authorization": f"Bearer {self.access_token()}",
            "Content-Type": "application/json",
        }
        pieces: list[str] = []
        completed = False
        with self._http.stream(
            "POST", f"{API_BASE_URL}/responses", headers=headers, json=payload
        ) as response:
            if response.is_error:
                response.read()
                raise LLMError(_http_error("OpenAI Responses request failed", response))
            for event in _sse_events(response.iter_lines()):
                kind = event.get("type")
                if kind == "response.output_text.delta":
                    pieces.append(str(event.get("delta", "")))
                elif kind == "response.failed":
                    error = (event.get("response") or {}).get("error") or {}
                    code = error.get("code", "unknown_error")
                    raise LLMError(f"OpenAI response failed: {code}")
                elif kind == "response.incomplete":
                    raise LLMError("OpenAI response was incomplete")
                elif kind == "response.completed":
                    completed = True
        if not completed:
            raise LLMError("OpenAI response stream ended without response.completed")
        return "".join(pieces)

    def _post_token(self, data: dict[str, str]) -> dict[str, Any]:
        response = self._http.post(TOKEN_URL, data=data)
        if response.is_error:
            raise OAuthError(_http_error("OpenAI token exchange failed", response))
        body: dict[str, Any] = response.json()
        return body

    def _verify_id_token(self, token: str, client_id: str, nonce: str) -> dict[str, Any]:
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if not kid:
                raise OAuthError("OpenAI ID token did not contain a key id")
            response = self._http.get(JWKS_URL)
            response.raise_for_status()
            keys = response.json().get("keys", [])
            matching = next((item for item in keys if item.get("kid") == kid), None)
            if matching is None:
                raise OAuthError("OpenAI ID token signing key was not found")
            jwk = jwt.PyJWK.from_dict(matching)
            claims: dict[str, Any] = jwt.decode(
                token,
                jwk.key,
                algorithms=[jwk.algorithm_name],
                audience=client_id,
                issuer=ISSUER,
                options={"require": ["exp", "iss", "aud", "sub", "nonce"]},
            )
        except OAuthError:
            raise
        except (jwt.PyJWTError, httpx.HTTPError, ValueError, StopIteration) as exc:
            raise OAuthError(f"OpenAI ID token validation failed: {exc}") from exc
        if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
            raise OAuthError("OpenAI ID token nonce did not match the pending sign-in")
        return claims

    def _load_state(self) -> tuple[str, _Profile | None]:
        if not self.credential_path.exists():
            host_id = f"urn:uuid:{uuid.uuid4()}"
            self._save_state(host_id, None)
            return host_id, None
        try:
            raw = json.loads(self.credential_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OAuthError(f"Could not read OpenAI credential file: {exc}") from exc
        host_id = raw.get("ext_agent_host_id")
        if not isinstance(host_id, str) or not host_id:
            raise OAuthError("OpenAI credential file is missing ext_agent_host_id")
        profile_raw = raw.get("profile")
        profile = _Profile.from_dict(profile_raw) if isinstance(profile_raw, dict) else None
        return host_id, profile

    def _load_profile(self) -> _Profile | None:
        return self._load_state()[1]

    def _save_state(self, host_id: str, profile: _Profile | None) -> None:
        path = self.credential_path
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "version": 1,
            "ext_agent_host_id": host_id,
            "profile": profile.as_dict() if profile else None,
        }
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            if os.name != "nt":
                os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(body, handle, indent=2)
                handle.write("\n")
            os.replace(tmp, path)
            if os.name != "nt":
                path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @contextmanager
    def _refresh_lock(self) -> Iterator[None]:
        lock_path = self.credential_path.with_suffix(self.credential_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 30.0
        fd: int | None = None
        while fd is None:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise OAuthError("Timed out waiting for the OpenAI token refresh lock") from None
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > 120:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                time.sleep(0.05)
        try:
            os.close(fd)
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _expires_at(self, profile: _Profile) -> datetime | None:
        if not profile.saved_at or not profile.expires_in:
            return None
        try:
            saved = datetime.fromisoformat(profile.saved_at)
        except ValueError:
            return None
        if saved.tzinfo is None:
            saved = saved.replace(tzinfo=UTC)
        return saved + timedelta(seconds=profile.expires_in)

    def _needs_refresh(self, profile: _Profile) -> bool:
        expires = self._expires_at(profile)
        return expires is None or self._now() >= expires - timedelta(minutes=2)


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise OAuthError(f"OpenAI token response did not contain {key}")
    return value


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _http_error(prefix: str, response: httpx.Response) -> str:
    request_id = response.headers.get("openai-request-id") or response.headers.get("x-request-id")
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text[:500]
    suffix = f" (request id {request_id})" if request_id else ""
    return f"{prefix}: HTTP {response.status_code}: {body}{suffix}"


def _sse_events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                raw = "\n".join(data_lines)
                data_lines.clear()
                if raw != "[DONE]":
                    yield json.loads(raw)
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        raw = "\n".join(data_lines)
        if raw != "[DONE]":
            yield json.loads(raw)
