"""Attendee retrieval. Frozen contract — other panes import find_candidates.

Strategy:
    1. Semantic search via Hyperspell (when env is configured + responsive).
    2. Fallback: lowercase keyword scoring over the same concat text used at indexing
       time. Deterministic, no network, runs in milliseconds.

The fallback is intentionally simple: even if Hyperspell never wakes up, the demo
still ranks attendees against the goal in a sensible way.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger("wingman.retrieval")

REPO_ROOT = Path(__file__).resolve().parent.parent
ATTENDEES_PATH = REPO_ROOT / "data" / "attendees.json"

CONTRACT_KEYS: tuple[str, ...] = (
    "name",
    "headline",
    "company",
    "recent_posts",
    "interests",
    "one_liner",
)

HYPERSPELL_TIMEOUT_S = 1.5  # leave headroom inside the 2s find_candidates budget


@lru_cache(maxsize=1)
def _load_attendees_cached() -> tuple[dict[str, Any], ...]:
    """Read attendees.json once per process and freeze a tuple of dicts."""
    if not ATTENDEES_PATH.exists():
        logger.warning("attendees.json not found at %s", ATTENDEES_PATH)
        return tuple()
    try:
        with ATTENDEES_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("failed to load attendees.json: %s", exc)
        return tuple()
    if not isinstance(raw, list):
        logger.error("attendees.json is not a JSON array")
        return tuple()
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if not all(k in item for k in CONTRACT_KEYS):
            continue
        cleaned.append(_shape(item))
    return tuple(cleaned)


def _shape(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce raw dict into the frozen contract shape — exactly the 6 keys."""
    return {
        "name": str(raw.get("name", "")).strip(),
        "headline": str(raw.get("headline", "")).strip(),
        "company": str(raw.get("company", "")).strip(),
        "recent_posts": [str(p).strip() for p in raw.get("recent_posts", []) if str(p).strip()],
        "interests": [str(s).strip() for s in raw.get("interests", []) if str(s).strip()],
        "one_liner": str(raw.get("one_liner", "")).strip(),
    }


def _embedding_text(att: dict[str, Any]) -> str:
    return "\n".join(
        [
            att["headline"],
            att["one_liner"],
            " | ".join(att["interests"]),
            "\n".join(att["recent_posts"]),
        ]
    )


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# Words that match almost everything and just add noise to the keyword score.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "i", "in", "is", "it", "of", "on", "or", "the", "to", "with",
        "me", "my", "we", "our", "you", "your", "this", "that", "any", "some",
        "find", "looking", "want", "need", "who", "someone", "people", "person",
    }
)


def _keyword_score(goal_tokens: list[str], doc_text: str) -> float:
    """Cheap lexical score: fraction of distinct goal tokens that appear in the doc.

    Bonuses for substring matches in the headline (heavily weighted) and interests.
    Returns a score in roughly [0, 1+] — used only for fallback ranking.
    """
    if not goal_tokens:
        return 0.0
    doc_lower = doc_text.lower()
    distinct = {t for t in goal_tokens if t not in _STOPWORDS and len(t) > 1}
    if not distinct:
        return 0.0
    hits = sum(1 for t in distinct if t in doc_lower)
    return hits / len(distinct)


def _fallback_search(goal: str, k: int) -> list[dict[str, Any]]:
    attendees = _load_attendees_cached()
    if not attendees:
        return []

    if not goal or not goal.strip():
        sorted_all = sorted(attendees, key=lambda a: a["name"].lower())
        return [_shape(a) for a in sorted_all[:k]]

    goal_tokens = _tokenize(goal)
    distinct = {t for t in goal_tokens if t not in _STOPWORDS and len(t) > 1}

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for att in attendees:
        text = _embedding_text(att)
        text_lower = text.lower()
        score = _keyword_score(goal_tokens, text)

        # Field-weighted bonuses: headline + interests are intent-heavy signals.
        headline_lower = att["headline"].lower()
        interests_lower = " | ".join(att["interests"]).lower()
        for tok in distinct:
            if tok in headline_lower:
                score += 0.5
            if tok in interests_lower:
                score += 0.25
            if tok in text_lower:
                # Tiny extra weight for any match — breaks ties toward fuller coverage.
                score += 0.05

        scored.append((score, att["name"].lower(), att))

    # Sort by score desc, then name asc for stable output.
    scored.sort(key=lambda t: (-t[0], t[1]))

    # If goal had usable tokens but nothing matched, return all alphabetically as a last resort.
    top = [att for score, _, att in scored if score > 0][:k]
    if not top:
        sorted_all = sorted(attendees, key=lambda a: a["name"].lower())
        return [_shape(a) for a in sorted_all[:k]]
    return [_shape(a) for a in top]


def _hyperspell_search(goal: str, k: int) -> list[dict[str, Any]] | None:
    """Try Hyperspell. Return None on any error so caller falls back."""
    api_key = os.environ.get("HYPERSPELL_API_KEY", "").strip()
    endpoint = os.environ.get("HYPERSPELL_ENDPOINT", "").strip()
    index = os.environ.get("HYPERSPELL_INDEX", "wingman_attendees").strip()
    if not api_key or not endpoint:
        return None

    try:
        import httpx
    except ImportError:
        logger.warning("httpx not available, skipping Hyperspell")
        return None

    base = endpoint.rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"query": goal, "index": index, "top_k": k}

    try:
        with httpx.Client(timeout=HYPERSPELL_TIMEOUT_S) as client:
            resp = client.post(f"{base}/search", headers=headers, json=payload)
            if resp.status_code >= 400:
                logger.warning("Hyperspell %s on /search: %s", resp.status_code, resp.text[:200])
                return None
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Hyperspell error: %s — falling back to local", exc)
        return None

    results = _extract_metadata(data)
    if not results:
        return None
    return results[:k]


def _extract_metadata(data: Any) -> list[dict[str, Any]]:
    """Hyperspell shape varies; pull metadata dicts that match our contract."""
    candidates: Iterable[Any]
    if isinstance(data, dict):
        candidates = data.get("results") or data.get("documents") or data.get("hits") or []
    elif isinstance(data, list):
        candidates = data
    else:
        return []

    out: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else item
        if not isinstance(meta, dict):
            continue
        if not all(k in meta for k in CONTRACT_KEYS):
            continue
        out.append(_shape(meta))
    return out


def find_candidates(goal: str, k: int = 10) -> list[dict[str, Any]]:
    """Return up to k attendees most relevant to `goal`, conforming to the frozen contract."""
    started = time.monotonic()
    if k <= 0:
        return []

    attendees = _load_attendees_cached()
    if not attendees:
        return []

    # Empty goal → alphabetical, capped at k. No remote call needed.
    if not goal or not goal.strip():
        sorted_all = sorted(attendees, key=lambda a: a["name"].lower())
        return [_shape(a) for a in sorted_all[: min(k, len(sorted_all))]]

    effective_k = min(k, len(attendees))

    semantic = _hyperspell_search(goal, effective_k)
    if semantic:
        elapsed_ms = (time.monotonic() - started) * 1000
        logger.info("hyperspell hit: goal=%r returned=%d (%.0fms)", goal, len(semantic), elapsed_ms)
        return semantic[:effective_k]

    fallback = _fallback_search(goal, effective_k)
    elapsed_ms = (time.monotonic() - started) * 1000
    logger.info("local fallback: goal=%r returned=%d (%.0fms)", goal, len(fallback), elapsed_ms)
    return fallback


__all__ = ["find_candidates", "CONTRACT_KEYS"]
