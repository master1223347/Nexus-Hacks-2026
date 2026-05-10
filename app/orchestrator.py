"""SMS pipeline orchestrator.

Composes the downstream contracts (retrieval / llm / memory) with timeouts
and a single fallback path. Owned by Pane 1; the downstream modules are
owned by other panes — we never edit them, only wrap them.

Contract (assumed; we shim missing bits locally so an empty module file
during the hackathon doesn't crash the webhook):

    retrieval.find_candidates(goal: str, k: int = 10) -> list[dict]
    llm.rank_and_riff(goal, candidates, message, history) -> str
    memory.get_goal(phone) / set_goal(phone, goal)
    memory.get_history(phone) / append_history(phone, user_msg, assistant_msg)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

def _safe_import(name: str) -> Any:
    """Import a sibling module if present, else return None.

    During parallel hackathon dev the downstream modules (llm, memory,
    retrieval) may not be on disk on every branch. _resolve() below handles
    None modules transparently.
    """
    try:
        from importlib import import_module

        return import_module(name)
    except ImportError:
        return None


_llm_module: Any = _safe_import("app.llm")
_memory_module: Any = _safe_import("app.memory")
_retrieval_module: Any = _safe_import("app.retrieval")

logger = logging.getLogger("wingman.orchestrator")

RETRIEVAL_TIMEOUT_S = 5.0
LLM_TIMEOUT_S = 1.5
FALLBACK_REPLY = "Still loading context — give me one more clear nudge."
DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------------
# Contract shim — fallbacks for empty/missing downstream modules
# ---------------------------------------------------------------------------

# In-memory fallback memory store. Used only when app.memory is empty during
# H2 dev; once Pane 3 ships memory.py, those callables shadow these.
_fallback_goals: dict[str, str] = {}
_fallback_history: dict[str, list[tuple[str, str]]] = {}


def _fallback_get_goal(phone: str) -> str | None:
    return _fallback_goals.get(phone)


def _fallback_set_goal(phone: str, goal: str) -> None:
    _fallback_goals[phone] = goal


def _fallback_get_history(phone: str) -> list[tuple[str, str]]:
    return list(_fallback_history.get(phone, []))


def _fallback_append_history(phone: str, user_msg: str, assistant_msg: str) -> None:
    _fallback_history.setdefault(phone, []).append((user_msg, assistant_msg))


def _fallback_find_candidates(goal: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    return []


def _fallback_rank_and_riff(
    goal: str | None,
    candidates: list[dict],
    message: str,
    history: list[tuple[str, str]],
) -> str:
    if not goal:
        return "Got it — what are you trying to get out of tonight? (e.g. 'raising seed for med-tech')"
    if not candidates:
        return f"Locked in your goal: \"{goal}\". Loading the room — text me 'who?' in a sec."
    return f"Top match for {goal}: " + ", ".join(
        c.get("name", "?") for c in candidates[:3]
    )


def _resolve(module: Any, name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    """Return the module's attribute if callable, else fall back."""
    if module is None:
        return fallback
    fn = getattr(module, name, None)
    return fn if callable(fn) else fallback


find_candidates = _resolve(_retrieval_module, "find_candidates", _fallback_find_candidates)
rank_and_riff = _resolve(_llm_module, "rank_and_riff", _fallback_rank_and_riff)
get_goal = _resolve(_memory_module, "get_goal", _fallback_get_goal)
set_goal = _resolve(_memory_module, "set_goal", _fallback_set_goal)
get_history = _resolve(_memory_module, "get_history", _fallback_get_history)
append_history = _resolve(_memory_module, "append_history", _fallback_append_history)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hash_phone(phone: str) -> str:
    """sha256 first 8 hex of phone — safe to log, untraceable to PII."""
    return hashlib.sha256(phone.encode("utf-8")).hexdigest()[:8]


def extract_goal(body: str) -> str:
    """First-message-becomes-goal heuristic. Trim + collapse whitespace."""
    return " ".join((body or "").split()).strip()


async def _run_with_timeout(
    fn: Callable[..., Any],
    args: tuple,
    timeout_s: float,
    label: str,
) -> tuple[Any, bool]:
    """Run a sync callable in a thread under a wall-clock timeout.

    Returns (result, ok). ok=False signals timeout or exception — the caller
    decides the fallback path.
    """
    try:
        coro: Awaitable[Any] = asyncio.to_thread(fn, *args)
        result = await asyncio.wait_for(coro, timeout=timeout_s)
        return result, True
    except asyncio.TimeoutError:
        logger.warning("%s.timeout after=%.2fs", label, timeout_s)
        return None, False
    except Exception as exc:  # noqa: BLE001 — fallback path, log + degrade
        logger.exception("%s.error err=%s", label, exc)
        return None, False


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmsTurnResult:
    reply: str
    goal: str | None
    n_candidates: int
    elapsed_ms: int


async def handle_sms_turn(from_phone: str, body: str) -> SmsTurnResult:
    """Run one inbound-SMS turn.

    Always returns a reply string — never raises, since Twilio expects 200 + TwiML.
    """
    started = time.perf_counter()
    body = (body or "").strip()
    phone_hash = hash_phone(from_phone)

    history = get_history(from_phone) or []
    goal = get_goal(from_phone)
    if not goal and body:
        goal = extract_goal(body)
        if goal:
            set_goal(from_phone, goal)

    # Retrieval — 5s budget. Skip entirely when goal is empty.
    candidates: list[dict] = []
    if goal:
        result, ok = await _run_with_timeout(
            find_candidates,
            (goal, DEFAULT_TOP_K),
            RETRIEVAL_TIMEOUT_S,
            "retrieval",
        )
        if ok and isinstance(result, list):
            candidates = result

    # LLM — 1.5s budget. On miss, hand back the fallback reply.
    reply: str
    result, ok = await _run_with_timeout(
        rank_and_riff,
        (goal, candidates, body, history),
        LLM_TIMEOUT_S,
        "llm",
    )
    if ok and isinstance(result, str) and result.strip():
        reply = result.strip()
    else:
        reply = FALLBACK_REPLY

    # Persist conversation regardless of which path produced the reply.
    try:
        append_history(from_phone, body, reply)
    except Exception:  # noqa: BLE001
        logger.exception("memory.append_history failed (non-fatal)")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "sms_turn phone_hash=%s body_len=%d goal=%r n_cands=%d reply_len=%d ms=%d",
        phone_hash,
        len(body),
        goal,
        len(candidates),
        len(reply),
        elapsed_ms,
    )

    return SmsTurnResult(
        reply=reply,
        goal=goal,
        n_candidates=len(candidates),
        elapsed_ms=elapsed_ms,
    )
