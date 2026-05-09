"""Minimal OpenAI Chat Completions client.

Sync httpx (httpx is already a project dep — avoids pulling in `openai`).
Pane 1 wraps rank_and_riff() in a 1.5s wall-clock budget; we set our own
client timeout slightly tighter so we can fall back to the H1 renderer
before Pane 1's outer timeout fires.

If OPENAI_API_KEY is unset we return None — callers must treat that as a
"skip the LLM, render deterministically" signal.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger("wingman.llm.client")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

DEFAULT_MODEL = os.environ.get("WINGMAN_LLM_MODEL", "gpt-4o-mini")
# Pane 1 budget is 1.5s — keep our HTTP timeout below that so we can fall
# back gracefully without Pane 1's outer wait_for() getting cancelled mid-call.
DEFAULT_HTTP_TIMEOUT_S = float(os.environ.get("WINGMAN_LLM_TIMEOUT_S", "1.2"))


class LLMUnavailable(Exception):
    """Raised (or returned via None) when no LLM call should be attempted."""


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 400,
    timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
) -> str | None:
    """Call OpenAI Chat Completions. Return assistant text or None.

    None means: API key missing, network/HTTP error, or empty content. The
    caller should fall back to the deterministic renderer — never raise to
    Pane 1, since the demo must always reply with something.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if resp.status_code != 200:
            logger.warning(
                "openai.non200 status=%d ms=%d body=%s",
                resp.status_code,
                elapsed_ms,
                resp.text[:200],
            )
            return None

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content", "")
        usage = data.get("usage") or {}
        logger.info(
            "openai.ok model=%s in=%s out=%s ms=%d",
            model,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            elapsed_ms,
        )
        return (content or "").strip() or None
    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("openai.timeout ms=%d budget=%.0fms", elapsed_ms, timeout_s * 1000)
        return None
    except httpx.HTTPError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("openai.http_error ms=%d err=%s", elapsed_ms, exc)
        return None
    except Exception:  # noqa: BLE001 — this is the demo safety net
        logger.exception("openai.unexpected")
        return None


__all__ = ["chat", "DEFAULT_MODEL", "DEFAULT_HTTP_TIMEOUT_S"]
