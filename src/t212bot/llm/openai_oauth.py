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
from contextlib import contextmanager, suppress
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


# Token-endpoint failures worth retrying: the request may succeed if sent again.
TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class OAuthError(RuntimeError):
    """OAuth registration, validation, refresh, or revocation failed."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        error: str | None = None,
        transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error = error
        self.transient = transient


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
        http_timeout: float = 30.0,
        callback_timeout: float = 300.0,
        refresh_attempts: int = 3,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        default = Path.home() / ".config" / "t212bot" / "openai_oauth.json"
        self.credential_path = Path(credential_path).expanduser() if credential_path else default
        self.agent_name = agent_name
        self.http_timeout = http_timeout
        self.callback_timeout = callback_timeout
        self.refresh_attempts = max(1, refresh_attempts)
        self._now = now
        self._sleep = sleep
        self._http = httpx.Client(timeout=http_timeout, transport=transport)
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
        """Run a loopback OAuth flow and persist the validated credentials.

        When the URL is shown to the user rather than opened directly (``--no-browser``),
        the saved ID token is never put in it: the account selector is shown instead.
        """
        host_id, existing, pending_client_id = self._load_full_state()
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
        callback: dict[str, str] = {}
        ignored: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(handler_self) -> None:  # noqa: N802
                parsed = urlparse(handler_self.path)
                values = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}
                if callback or parsed.path != CALLBACK_PATH:
                    handler_self.send_response(404)
                    handler_self.end_headers()
                    return
                if not secrets.compare_digest(values.get("state", ""), state):
                    # A stale tab or stray request: keep waiting for the real callback.
                    ignored.append("state mismatch")
                    handler_self.send_response(400)
                    handler_self.end_headers()
                    return
                callback.update(values)
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
        if existing:
            request_client_id = existing.client_id
        else:
            request_client_id = pending_client_id or DYNAMIC_CLIENT_ID
        params = {
            "client_id": request_client_id,
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
        url_is_shown = url_callback is not None or not open_browser
        if existing:
            # id_token_hint must never reach logs or a terminal; only send it when the URL
            # goes straight to the browser.
            if existing.id_token and not url_is_shown:
                params["id_token_hint"] = existing.id_token
            if existing.email:
                params["login_hint"] = existing.email
        else:
            params["agent_name_hint"] = self.agent_name

        authorization_url = f"{AUTHORIZE_URL}?{urlencode(params)}"
        try:
            if url_callback:
                url_callback(authorization_url)
            if open_browser and not browser_opener(authorization_url):
                raise OAuthError(
                    "Could not open a browser; rerun with --no-browser and open the URL manually"
                )
            deadline = time.monotonic() + self.callback_timeout
            while not callback:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                server.timeout = min(remaining, 1.0)
                server.handle_request()
        finally:
            server.server_close()
        if not callback:
            detail = f" (ignored: {', '.join(ignored)})" if ignored else ""
            raise OAuthError(
                f"Timed out after {self.callback_timeout:.0f}s waiting for the OpenAI "
                f"OAuth callback{detail}"
            )
        if callback.get("error"):
            raise OAuthError(f"OpenAI authorization failed: {callback['error']}")
        code = callback.get("code")
        if not code:
            raise OAuthError("OAuth callback did not include an authorization code")

        returned_client_id = callback.get("client_id")
        if existing or pending_client_id:
            expected = existing.client_id if existing else pending_client_id
            if returned_client_id and returned_client_id != expected:
                raise OAuthError(
                    "OAuth callback returned a different client_id for the saved registration"
                )
            issued_client_id = str(expected)
        else:
            if not returned_client_id or returned_client_id == DYNAMIC_CLIENT_ID:
                raise OAuthError(
                    "OpenAI did not return an issued client_id for the new registration"
                )
            issued_client_id = returned_client_id
            # Keep the issued ID before exchanging the code, so a retry after invalid_grant
            # restarts authorization with it instead of registering again.
            self._save_state(host_id, None, pending_client_id=issued_client_id)

        try:
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
        except OAuthError as exc:
            if exc.error == "invalid_grant":
                raise OAuthError(
                    "The authorization code was rejected (invalid_grant). "
                    "Run `t212bot llm-login` again; the issued client ID has been kept.",
                    status=exc.status,
                    error=exc.error,
                ) from exc
            raise
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
            # force=False: after taking the lock, reuse a token another process just refreshed.
            profile = self.refresh(force=False)
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
            data = self._refresh_with_retry(profile)
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

    def _refresh_with_retry(self, profile: _Profile) -> dict[str, Any]:
        """Retry transient failures with backoff; clear the session on terminal ones."""
        request = {
            "grant_type": "refresh_token",
            "client_id": profile.client_id,
            "refresh_token": str(profile.refresh_token),
            "resource": RESOURCE,
        }
        for attempt in range(self.refresh_attempts):
            try:
                return self._post_token(request)
            except OAuthError as exc:
                if exc.transient and attempt + 1 < self.refresh_attempts:
                    self._sleep(float(2**attempt))
                    continue
                if not exc.transient:
                    self._clear_tokens(profile)
                    raise OAuthError(
                        f"The OpenAI session can no longer be refreshed ({exc}). "
                        "Run `t212bot llm-login` to sign in again.",
                        status=exc.status,
                        error=exc.error,
                    ) from exc
                raise
        raise AssertionError("unreachable")

    def _clear_tokens(self, profile: _Profile) -> None:
        profile.id_token = None
        profile.access_token = None
        profile.refresh_token = None
        profile.expires_in = 0
        profile.scopes = []
        profile.saved_at = None
        self._save_state(profile.ext_agent_host_id, profile)

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
        self._clear_tokens(profile)
        return revoked

    def list_models(self) -> list[ModelInfo]:
        try:
            response = self._http.get(
                f"{API_BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.access_token()}"},
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI model discovery failed: {exc}") from exc
        if response.is_error:
            raise LLMError(_http_error("OpenAI model discovery failed", response))
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"OpenAI model discovery returned invalid JSON: {exc}") from exc
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
        request_id: str | None = None
        try:
            with self._http.stream(
                "POST", f"{API_BASE_URL}/responses", headers=headers, json=payload
            ) as response:
                request_id = _request_id(response)
                if response.is_error:
                    response.read()
                    raise LLMError(_http_error("OpenAI Responses request failed", response))
                for event in _sse_events(response.iter_lines()):
                    kind = event.get("type")
                    if kind == "response.output_text.delta":
                        pieces.append(str(event.get("delta", "")))
                    elif kind == "response.failed":
                        error = (event.get("response") or {}).get("error") or {}
                        raise LLMError(_describe_failure(error, request_id))
                    elif kind == "response.incomplete":
                        details = (event.get("response") or {}).get("incomplete_details") or {}
                        reason = details.get("reason", "unknown reason")
                        raise LLMError(
                            f"OpenAI response was incomplete: {reason}{_rid(request_id)}"
                        )
                    elif kind == "response.completed":
                        completed = True
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI Responses request failed: {exc}{_rid(request_id)}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"OpenAI Responses stream had a malformed event: {exc}{_rid(request_id)}"
            ) from exc
        if not completed:
            raise LLMError(
                f"OpenAI response stream ended without response.completed{_rid(request_id)}"
            )
        return "".join(pieces)

    def _post_token(self, data: dict[str, str]) -> dict[str, Any]:
        try:
            response = self._http.post(TOKEN_URL, data=data)
        except httpx.TransportError as exc:
            raise OAuthError(f"OpenAI token request failed: {exc}", transient=True) from exc
        if response.is_error:
            error: str | None = None
            with suppress(ValueError, AttributeError):
                error = response.json().get("error")
            raise OAuthError(
                _http_error("OpenAI token exchange failed", response),
                status=response.status_code,
                error=error if isinstance(error, str) else None,
                transient=response.status_code in TRANSIENT_STATUSES,
            )
        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise OAuthError(
                f"OpenAI token endpoint returned invalid JSON: {exc}", transient=True
            ) from exc
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
        host_id, profile, _ = self._load_full_state()
        return host_id, profile

    def _load_full_state(self) -> tuple[str, _Profile | None, str | None]:
        if not self.credential_path.exists():
            host_id = f"urn:uuid:{uuid.uuid4()}"
            self._save_state(host_id, None)
            return host_id, None, None
        try:
            raw = json.loads(self.credential_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OAuthError(f"Could not read OpenAI credential file: {exc}") from exc
        host_id = raw.get("ext_agent_host_id")
        if not isinstance(host_id, str) or not host_id:
            raise OAuthError("OpenAI credential file is missing ext_agent_host_id")
        profile_raw = raw.get("profile")
        profile = _Profile.from_dict(profile_raw) if isinstance(profile_raw, dict) else None
        pending = raw.get("pending_client_id")
        return host_id, profile, pending if isinstance(pending, str) and pending else None

    def _load_profile(self) -> _Profile | None:
        return self._load_state()[1]

    def _save_state(
        self, host_id: str, profile: _Profile | None, *, pending_client_id: str | None = None
    ) -> None:
        path = self.credential_path
        path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "version": 1,
            "ext_agent_host_id": host_id,
            "profile": profile.as_dict() if profile else None,
        }
        if pending_client_id:
            body["pending_client_id"] = pending_client_id
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            if os.name != "nt":
                os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
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
                    raise OAuthError(
                        "Timed out waiting for the OpenAI token refresh lock"
                    ) from None
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
            with suppress(FileNotFoundError):
                lock_path.unlink()

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


def _request_id(response: httpx.Response) -> str | None:
    return response.headers.get("openai-request-id") or response.headers.get("x-request-id")


def _rid(request_id: str | None) -> str:
    return f" (request id {request_id})" if request_id else ""


def _describe_failure(error: dict[str, Any], request_id: str | None) -> str:
    parts = [str(error.get("code") or "unknown_error")]
    if error.get("message"):
        parts.append(str(error["message"]))
    if error.get("param"):
        parts.append(f"param={error['param']}")
    return f"OpenAI response failed: {': '.join(parts)}{_rid(request_id)}"


def _http_error(prefix: str, response: httpx.Response) -> str:
    request_id = _request_id(response)
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
