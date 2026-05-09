"""Per-phone memory: networking goal + recent message history.

Two backends behind a single Memory protocol:
  - InMemoryMemory: thread-safe dict (default; always safe).
  - NiaMemory:      HTTP-backed via NIA_ENDPOINT, scoped by NIA_NAMESPACE.

Selection: if NIA_API_KEY and NIA_ENDPOINT are both set, NiaMemory is the
primary backend. Every call wraps Nia in try/except — on ANY error we degrade
to the in-memory backend, log a warning, and still return a sane value. The
demo must never crash on a memory hiccup.

Pane 1 imports module-level get_goal / set_goal / get_history / append_history
— those signatures are stable across H1/H3.
"""

from __future__ import annotations

import logging
import os
from threading import RLock
from typing import Any, Protocol

import httpx

logger = logging.getLogger("wingman.memory")

_HISTORY_CAP = 20
_NIA_TIMEOUT_S = float(os.environ.get("WINGMAN_NIA_TIMEOUT_S", "0.8"))


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Memory(Protocol):
    def get_goal(self, phone: str) -> str | None: ...
    def set_goal(self, phone: str, goal: str) -> None: ...
    def get_history(self, phone: str) -> list[dict[str, str]]: ...
    def append_history(
        self, phone: str, user_msg: str, assistant_msg: str
    ) -> None: ...


# ---------------------------------------------------------------------------
# In-memory backend (always available)
# ---------------------------------------------------------------------------


class InMemoryMemory:
    """Thread-safe dict-backed memory. The fallback for everything."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._goals: dict[str, str] = {}
        self._histories: dict[str, list[dict[str, str]]] = {}

    def get_goal(self, phone: str) -> str | None:
        with self._lock:
            return self._goals.get(phone)

    def set_goal(self, phone: str, goal: str) -> None:
        if not phone or not goal or not goal.strip():
            return
        with self._lock:
            self._goals[phone] = goal.strip()

    def get_history(self, phone: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._histories.get(phone, ()))

    def append_history(
        self, phone: str, user_msg: str, assistant_msg: str
    ) -> None:
        if not phone:
            return
        with self._lock:
            bucket = self._histories.setdefault(phone, [])
            if user_msg:
                bucket.append({"role": "user", "content": user_msg})
            if assistant_msg:
                bucket.append({"role": "assistant", "content": assistant_msg})
            overflow = len(bucket) - _HISTORY_CAP
            if overflow > 0:
                del bucket[:overflow]

    def reset(self) -> None:
        with self._lock:
            self._goals.clear()
            self._histories.clear()


# ---------------------------------------------------------------------------
# Nia backend
# ---------------------------------------------------------------------------


class NiaMemory:
    """Nia-backed memory adapter.

    Treats Nia as a generic key/value + append-only-list store, scoped by
    namespace. The exact REST shape isn't standardized in the hackathon
    sponsor docs, so we keep the surface conservative and fail fast on any
    non-2xx response. The orchestrator (`_dispatch`) catches our exceptions
    and falls back to the in-memory backend so the demo stays alive.

    Endpoints used (configurable via env):
      POST   {endpoint}/v1/memory/{namespace}/{phone}/goal     {"goal": ...}
      GET    {endpoint}/v1/memory/{namespace}/{phone}/goal
      POST   {endpoint}/v1/memory/{namespace}/{phone}/history  {"role":..,"content":..}
      GET    {endpoint}/v1/memory/{namespace}/{phone}/history?limit=N
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        namespace: str = "wingman_memory",
        timeout_s: float = _NIA_TIMEOUT_S,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.namespace = namespace
        self.timeout_s = timeout_s

    # ----- internal HTTP --------------------------------------------------

    def _url(self, *parts: str) -> str:
        suffix = "/".join(p.strip("/") for p in parts if p)
        return f"{self.endpoint}/v1/memory/{self.namespace}/{suffix}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, url: str, **kw: Any) -> httpx.Response:
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.request(method, url, headers=self._headers(), **kw)
        if resp.status_code >= 500:
            raise RuntimeError(f"nia.{resp.status_code} url={url}")
        return resp

    # ----- protocol -------------------------------------------------------

    def get_goal(self, phone: str) -> str | None:
        resp = self._request("GET", self._url(phone, "goal"))
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise RuntimeError(f"nia.get_goal.{resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RuntimeError("nia.get_goal.bad_json") from exc
        goal = payload.get("goal") if isinstance(payload, dict) else None
        return goal if isinstance(goal, str) and goal.strip() else None

    def set_goal(self, phone: str, goal: str) -> None:
        if not phone or not goal or not goal.strip():
            return
        resp = self._request(
            "POST",
            self._url(phone, "goal"),
            json={"goal": goal.strip()},
        )
        if resp.status_code not in (200, 201, 204):
            raise RuntimeError(f"nia.set_goal.{resp.status_code}")

    def get_history(self, phone: str) -> list[dict[str, str]]:
        resp = self._request(
            "GET",
            self._url(phone, "history"),
            params={"limit": _HISTORY_CAP},
        )
        if resp.status_code == 404:
            return []
        if resp.status_code != 200:
            raise RuntimeError(f"nia.get_history.{resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RuntimeError("nia.get_history.bad_json") from exc
        items = payload.get("history") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return []
        return [
            {"role": str(it.get("role", "")), "content": str(it.get("content", ""))}
            for it in items
            if isinstance(it, dict)
        ]

    def append_history(
        self, phone: str, user_msg: str, assistant_msg: str
    ) -> None:
        if not phone:
            return
        # Two POSTs (user then assistant) keep the shape simple; Nia returns
        # the canonical ordered history on read.
        for role, content in (("user", user_msg), ("assistant", assistant_msg)):
            if not content:
                continue
            resp = self._request(
                "POST",
                self._url(phone, "history"),
                json={"role": role, "content": content},
            )
            if resp.status_code not in (200, 201, 204):
                raise RuntimeError(f"nia.append_history.{resp.status_code}")


# ---------------------------------------------------------------------------
# Backend selection + dispatch with safe fallback
# ---------------------------------------------------------------------------


_in_memory: InMemoryMemory = InMemoryMemory()
_primary: Memory | None = None
_primary_name: str = "in_memory"


def _select_primary() -> tuple[Memory, str]:
    """Choose the primary backend at import time. Re-callable for tests."""
    nia_key = os.environ.get("NIA_API_KEY", "").strip()
    nia_endpoint = os.environ.get("NIA_ENDPOINT", "").strip()
    namespace = os.environ.get("NIA_NAMESPACE", "wingman_memory").strip() or "wingman_memory"

    if nia_key and nia_endpoint:
        return (
            NiaMemory(endpoint=nia_endpoint, api_key=nia_key, namespace=namespace),
            "nia",
        )
    return (_in_memory, "in_memory")


_primary, _primary_name = _select_primary()
logger.info("memory.backend=%s", _primary_name)


def _dispatch(method: str, *args: Any, _default: Any = None) -> Any:
    """Call the primary backend's `method`; on error, fall back to in-memory."""
    fn = getattr(_primary, method, None)
    if callable(fn):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 — degrade, don't crash
            logger.warning(
                "memory.%s primary=%s error=%s; falling back to in_memory",
                method,
                _primary_name,
                exc,
            )

    fallback = getattr(_in_memory, method, None)
    if callable(fallback):
        return fallback(*args)
    return _default


# ---------------------------------------------------------------------------
# Public API (Pane 1 contract — DO NOT change signatures)
# ---------------------------------------------------------------------------


def get_goal(phone: str) -> str | None:
    return _dispatch("get_goal", phone, _default=None)


def set_goal(phone: str, goal: str) -> None:
    _dispatch("set_goal", phone, goal)


def get_history(phone: str) -> list[dict[str, str]]:
    result = _dispatch("get_history", phone, _default=[])
    return result if isinstance(result, list) else []


def append_history(phone: str, user_msg: str, assistant_msg: str) -> None:
    _dispatch("append_history", phone, user_msg, assistant_msg)


def _reset_for_tests() -> None:
    """Test-only helper. Not part of the public contract."""
    _in_memory.reset()


def _backend_name() -> str:
    return _primary_name


__all__ = [
    "get_goal",
    "set_goal",
    "get_history",
    "append_history",
]


# ---------------------------------------------------------------------------
# Smoke test (run as `python -m app.memory`)
# ---------------------------------------------------------------------------


def _smoke() -> None:
    """Round-trip set/get for goal + history. Prints OK or the failure."""
    phone = "+15555550199"
    _reset_for_tests()
    set_goal(phone, "raising a seed for med-tech AI")
    assert get_goal(phone) == "raising a seed for med-tech AI", (
        f"goal round-trip failed: {get_goal(phone)!r}"
    )

    append_history(phone, "raising a seed", "Top 3 tonight: ...")
    append_history(phone, "tell me about Marcus", "Marcus is...")
    hist = get_history(phone)
    assert len(hist) == 4, f"expected 4 history entries, got {len(hist)}: {hist}"
    assert hist[0] == {"role": "user", "content": "raising a seed"}
    assert hist[-1]["role"] == "assistant"

    print(f"memory.smoke OK backend={_backend_name()} hist_len={len(hist)}")


if __name__ == "__main__":
    _smoke()
