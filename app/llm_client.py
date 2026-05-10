"""Gemini-backed LLM client.

Stack:
  - SDK: google-genai (already in .venv)
  - Models:
      ranking + drill-in + intent  -> gemini-2.5-flash-lite
      rapport (the demo-killer)    -> gemini-2.5-flash
  - Auth: GEMINI_API_KEY (NOT OpenAI/Anthropic — those are out of scope)

Public surface (signatures stable; app.llm doesn't change):
    chat(messages, *, model=..., temperature=..., max_tokens=..., timeout_s=...)
        -> str | None      # generic text completion
    chat_structured(messages, *, schema, model=..., ...) -> Any | None
                              # JSON output validated against a Pydantic schema

Both return None on missing key, timeout, HTTP error, or empty content. The
caller (app.llm.rank_and_riff) treats None as "fall back to the deterministic
H1 renderer" — the demo never crashes from a model hiccup.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger("wingman.llm.client")

# Per-mode default models. app.llm picks the right one per request.
# NOTE: rapport defaulted to flash-lite (high free-tier quota = 1000/day).
# Set WINGMAN_RAPPORT_MODEL=gemini-2.5-flash to escalate when quota allows.
# Set WINGMAN_RAPPORT_MODEL=gemini-2.5-pro only if eval drops below 9/10.
MODEL_RANK = os.environ.get("WINGMAN_RANK_MODEL", "gemini-2.5-flash-lite")
MODEL_RAPPORT = os.environ.get("WINGMAN_RAPPORT_MODEL", "gemini-2.5-flash-lite")

# Pane 1 budget is 1.5s — flash-lite is well under 1s in practice; flash is
# ~1-2s. We rely on the SDK's internal connection timeout.
DEFAULT_TIMEOUT_S = float(os.environ.get("WINGMAN_LLM_TIMEOUT_S", "4.0"))


def _api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key or None


def _split_system_and_contents(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Map OpenAI-style messages to Gemini's (system_instruction, contents).

    - All system messages are concatenated into one system_instruction.
    - Remaining user/assistant messages map to role user/model in the
      contents array.
    """
    system_chunks: list[str] = []
    contents: list[dict[str, Any]] = []

    for msg in messages:
        role = (msg.get("role") or "").lower()
        text = msg.get("content") or ""
        if not text:
            continue
        if role == "system":
            system_chunks.append(text)
            continue
        gem_role = "model" if role == "assistant" else "user"
        contents.append({"role": gem_role, "parts": [{"text": text}]})

    system_instruction = "\n\n".join(system_chunks).strip() or None
    return system_instruction, contents


def _client() -> Any | None:
    """Lazy import + lazy client construction. Returns None when no key."""
    key = _api_key()
    if not key:
        return None
    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai not installed; LLM disabled")
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:  # noqa: BLE001
        logger.exception("genai.Client construction failed")
        return None


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = MODEL_RAPPORT,
    temperature: float = 0.5,
    max_tokens: int = 400,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str | None:
    """Generic text completion. Returns assistant text or None on any failure."""
    client = _client()
    if client is None:
        return None

    system_instruction, contents = _split_system_and_contents(messages)
    if not contents:
        return None

    try:
        from google.genai import types
    except ImportError:
        logger.warning("google.genai.types unavailable")
        return None

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_instruction,
    )

    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=model, contents=contents, config=config
        )
    except Exception as exc:  # noqa: BLE001 — demo safety net
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("gemini.error model=%s ms=%d err=%s", model, elapsed_ms, exc)
        return None

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    text = (getattr(resp, "text", None) or "").strip()
    usage = getattr(resp, "usage_metadata", None)
    in_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    out_tokens = getattr(usage, "candidates_token_count", None) if usage else None
    logger.info(
        "gemini.ok model=%s in=%s out=%s ms=%d",
        model,
        in_tokens,
        out_tokens,
        elapsed_ms,
    )

    return text or None


def chat_structured(
    messages: list[dict[str, str]],
    *,
    schema: Any,
    model: str = MODEL_RANK,
    temperature: float = 0.3,
    max_tokens: int = 400,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Any | None:
    """JSON-output completion validated against a Pydantic schema.

    Returns the parsed Python object (Pydantic model instance, list, or dict)
    or None on failure. If the SDK can't auto-parse, falls back to manual
    json.loads on the raw text.
    """
    client = _client()
    if client is None:
        return None

    system_instruction, contents = _split_system_and_contents(messages)
    if not contents:
        return None

    try:
        from google.genai import types
    except ImportError:
        return None

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=schema,
    )

    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=model, contents=contents, config=config
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "gemini.struct.error model=%s ms=%d err=%s", model, elapsed_ms, exc
        )
        return None

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    parsed = getattr(resp, "parsed", None)
    if parsed is not None:
        logger.info("gemini.struct.ok model=%s ms=%d parsed=1", model, elapsed_ms)
        return parsed

    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        logger.info("gemini.struct.empty model=%s ms=%d", model, elapsed_ms)
        return None
    try:
        import json

        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "gemini.struct.bad_json model=%s text=%r", model, text[:200]
        )
        return None


def warm(
    *,
    model: str | None = None,
    timeout_s: float = 5.0,
) -> bool:
    """Cold-start the Gemini connection so the first real user call is warm.

    Pane 1 calls this from FastAPI's startup hook (app/main.py). Cold
    Gemini latency was observed at 1.5-5.7s on the very first call;
    pre-warming pulls that hit out of the user-facing request path.

    Cost: one prompt token + one output token (~free).
    Behavior: never raises. Returns True on a clean ping, False otherwise.
    Safe to call multiple times — idempotent.

    Suggested wiring in app/main.py (do this from Pane 1, not here):

        from app.llm import warm_llm

        @app.on_event("startup")
        async def _startup_warm() -> None:
            # Sync call is fine — startup is not in the request path.
            warm_llm()

        # OR, with FastAPI lifespan:
        # @asynccontextmanager
        # async def lifespan(app):
        #     warm_llm()
        #     yield
    """
    if not _api_key():
        logger.info("gemini.warm.skipped reason=no_api_key")
        return False

    client = _client()
    if client is None:
        logger.info("gemini.warm.skipped reason=client_unavailable")
        return False

    try:
        from google.genai import types
    except ImportError:
        logger.warning("gemini.warm.skipped reason=types_unavailable")
        return False

    target_model = model or MODEL_RANK
    config = types.GenerateContentConfig(max_output_tokens=1, temperature=0.0)

    started = time.perf_counter()
    try:
        client.models.generate_content(
            model=target_model,
            contents="ping",
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 — startup must never fail
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "gemini.warm.failed model=%s ms=%d err=%s",
            target_model,
            elapsed_ms,
            exc,
        )
        return False

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info("gemini.warmed_up model=%s ms=%d", target_model, elapsed_ms)
    return True


__all__ = [
    "chat",
    "chat_structured",
    "warm",
    "MODEL_RANK",
    "MODEL_RAPPORT",
    "DEFAULT_TIMEOUT_S",
]
