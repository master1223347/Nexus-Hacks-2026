"""Minimal LLM client with OpenAI + Gemini support.

Sync httpx (httpx is already a project dep — avoids pulling in `openai`).
Pane 1 wraps rank_and_riff() in a 1.5s wall-clock budget; we set our own
client timeout slightly tighter so we can fall back to the H1 renderer
before Pane 1's outer timeout fires.

If no provider API key is configured we return None — callers must treat that
as a "skip the LLM, render deterministically" signal.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger("wingman.llm.client")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_MODEL = os.environ.get("WINGMAN_LLM_MODEL", "gpt-4o-mini")
# Pane 1 budget is 1.5s — keep our HTTP timeout below that so we can fall
# back gracefully without Pane 1's outer wait_for() getting cancelled mid-call.
DEFAULT_HTTP_TIMEOUT_S = float(os.environ.get("WINGMAN_LLM_TIMEOUT_S", "1.2"))


class LLMUnavailable(Exception):
    """Raised (or returned via None) when no LLM call should be attempted."""


def is_configured() -> bool:
    """True when at least one supported provider has a usable API key."""
    return bool(
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 400,
    timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
) -> str | None:
    """Call configured LLM provider. Return assistant text or None.

    None means: API key missing, network/HTTP error, or empty content. The
    caller should fall back to the deterministic renderer — never raise to
    Pane 1, since the demo must always reply with something.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # Keep existing behavior if OpenAI is configured; otherwise use Gemini.
    if openai_key:
        return _chat_openai(
            messages=messages,
            api_key=openai_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    if gemini_key:
        gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
        return _chat_gemini(
            messages=messages,
            api_key=gemini_key,
            model=gemini_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    return None


def _chat_openai(
    *,
    messages: list[dict[str, str]],
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> str | None:
    """Call OpenAI Chat Completions. Return assistant text or None."""

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


def _chat_gemini(
    *,
    messages: list[dict[str, str]],
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> str | None:
    """Call Gemini generateContent. Return assistant text or None."""
    system_text = ""
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        text = (msg.get("content") or "").strip()
        if not text:
            continue
        if role == "system":
            system_text = f"{system_text}\n\n{text}".strip() if system_text else text
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            }
        )

    if not contents:
        return None

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_text:
        payload["system_instruction"] = {"parts": [{"text": system_text}]}

    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                GEMINI_URL_TMPL.format(model=model),
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if resp.status_code != 200:
            logger.warning(
                "gemini.non200 status=%d model=%s ms=%d body=%s",
                resp.status_code,
                model,
                elapsed_ms,
                resp.text[:200],
            )
            return None

        data = resp.json()
        candidates = data.get("candidates") or []
        parts = (((candidates[0] if candidates else {}).get("content") or {}).get("parts")) or []
        text_out = "\n".join((p.get("text") or "").strip() for p in parts if p.get("text"))
        logger.info("gemini.ok model=%s ms=%d", model, elapsed_ms)
        return text_out.strip() or None
    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("gemini.timeout model=%s ms=%d budget=%.0fms", model, elapsed_ms, timeout_s * 1000)
        return None
    except httpx.HTTPError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("gemini.http_error model=%s ms=%d err=%s", model, elapsed_ms, exc)
        return None
    except Exception:  # noqa: BLE001 — this is the demo safety net
        logger.exception("gemini.unexpected model=%s", model)
        return None


__all__ = ["chat", "is_configured", "DEFAULT_MODEL", "DEFAULT_HTTP_TIMEOUT_S"]
