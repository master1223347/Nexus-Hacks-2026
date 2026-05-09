"""Wingman LLM orchestrator.

Public surface (Pane 1 imports this directly):
    extract_goal(message)            -> str | None
    rank_and_riff(goal, candidates, message, history) -> str

H1 (this file): no network calls. Branches on simple keyword routing and
returns hand-shaped demo replies. Pane 2 may pass an empty candidate list while
their pipeline boots; we still return something demo-grade so Pane 1 can wire
the SMS round-trip immediately.

H2 will replace _route() and the per-mode renderers with a single LLM call
that uses the system prompts in app/prompts/*. The signatures and routing
keywords are stable across H1/H2 so swapping is mechanical.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.prompts.rank import FALLBACK_REPLY

logger = logging.getLogger("wingman.llm")

# SMS budgets (verified by scripts/eval_rapport.py in H2).
MAX_LIST_CHARS = 320
MAX_DRILL_CHARS = 480
MAX_RAPPORT_CHARS = 320

# Routing keywords. Lowercased, whole-word match where order matters.
_GOAL_VERBS = (
    "raising",
    "raise",
    "seed",
    "series",
    "looking for",
    "want to meet",
    "want to find",
    "trying to meet",
    "trying to find",
    "find me",
    "find ",
    "meet ",
    "hire",
    "hiring",
    "investor",
    "investors",
    "vc",
    "vcs",
    "cofounder",
    "co-founder",
    "engineer",
    "engineers",
    "designer",
    "designers",
)

_RAPPORT_KEYWORDS = (
    "fun",
    "drink",
    "drinks",
    "coffee",
    "chill",
    "casual",
    "interesting",
    "cool",
    "vibe",
    "vibey",
    "hang",
    "hangout",
    "boba",
)

_DRILL_VERBS = (
    "tell me about",
    "tell me more about",
    "more on",
    "more about",
    "who is",
    "what about",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_goal(message: str) -> str | None:
    """Return a networking goal string from a message, or None if unclear.

    H1 heuristic: if any goal-verb appears, we treat the whole message as the
    goal (lightly cleaned). H2 may upgrade this to a tiny LLM call, but most
    real demo phrasings ("I'm raising a seed for med-tech AI") satisfy the
    heuristic and don't need a model round-trip.
    """
    if not message or not message.strip():
        return None

    text = message.strip()
    lowered = text.lower()

    if not any(kw in lowered for kw in _GOAL_VERBS):
        return None

    cleaned = re.sub(r"\s+", " ", text).strip(" .!?…")
    return cleaned or None


def rank_and_riff(
    goal: str,
    candidates: list[dict[str, Any]],
    message: str,
    history: list[dict[str, str]],
) -> str:
    """Produce an SMS reply text for one of three branches.

    Branches:
      - initial:  user just gave a goal -> top-3 list
      - drill-in: user named an attendee -> bio + opener with a quoted post
      - rapport:  user asked for fun/casual -> one person, verbatim post quote

    On any internal error, return FALLBACK_REPLY rather than a stack trace.
    """
    try:
        all_cands = list(candidates or [])
        mode = _route(message=message or "", candidates=all_cands)
        logger.info(
            "llm_route mode=%s goal_set=%s n_candidates=%d hist=%d",
            mode,
            bool(goal),
            len(all_cands),
            len(history or []),
        )

        # Last-6 history slice + top-5 candidate slice, per H2 architecture.
        # Done here so H1 callers already exercise the same shape H2 expects.
        top5 = all_cands[:5]
        recent_history = list(history or [])[-6:]
        del recent_history  # unused in H1; kept for symmetry

        if mode == "rapport":
            return _h1_rapport(goal=goal, candidates=top5)
        if mode == "drill_in":
            # Drill-in searches the full list — users can name anyone, not
            # just the top 5. Token-budget truncation only matters for the
            # H2 LLM prompt, not for H1's deterministic renderer.
            return _h1_drill_in(message=message, candidates=all_cands)
        return _h1_initial(goal=goal, candidates=top5)
    except Exception:  # pragma: no cover — defensive; demo must never 500
        logger.exception("rank_and_riff failed; returning fallback")
        return FALLBACK_REPLY


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route(message: str, candidates: list[dict[str, Any]]) -> str:
    """Return one of: 'initial' | 'drill_in' | 'rapport'.

    Order matters: drill-in beats rapport beats initial because a user can
    legitimately say "tell me about marcus, who's fun?" and we want the
    direct-name signal to win.
    """
    lowered = message.lower()

    if any(verb in lowered for verb in _DRILL_VERBS):
        return "drill_in"
    if _match_candidate(message=message, candidates=candidates) is not None:
        return "drill_in"

    if any(re.search(rf"\b{re.escape(kw)}\b", lowered) for kw in _RAPPORT_KEYWORDS):
        return "rapport"

    return "initial"


# ---------------------------------------------------------------------------
# H1 hardcoded renderers
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[: max(1, limit - 1)].rstrip()
    return cut + "…"


_DEMO_TOP3 = (
    "Top 3 tonight:\n"
    "1) Sarah Chen — GP at Bessemer, leads health AI\n"
    "2) Marcus Patel — ex-surgeon, AI advisor at a16z\n"
    "3) Priya Shah — founder Medvana, recent bridge"
)

_DEMO_DRILL_MARCUS = (
    "Marcus Patel — 20yrs ortho surgeon, pivoted to AI in 2022. "
    "Posted yesterday about Whoop biometrics for clinical trials. "
    "Open with: \"saw your Whoop thread — what wearable data are you most "
    "bullish on for clinical use?\""
)

_DEMO_RAPPORT_PRIYA = (
    "Priya. She's been \"live-tweeting my boba shop tier list\" all week "
    "and just posted about Laufey at the Greek. Open with boba spots in "
    "the area."
)


def _h1_initial(goal: str, candidates: list[dict[str, Any]]) -> str:
    """Top-3 list. Uses real candidate data when available; falls back to demo."""
    if not candidates:
        return _truncate(_DEMO_TOP3, MAX_LIST_CHARS)

    lines: list[str] = []
    for i, cand in enumerate(candidates[:3], start=1):
        name = (cand.get("name") or "").strip() or "Unknown"
        one_liner = (cand.get("one_liner") or cand.get("headline") or "").strip()
        one_liner = re.sub(r"\s+", " ", one_liner)[:80]
        if not one_liner:
            one_liner = "in the room tonight"
        lines.append(f"{i}) {name} — {one_liner}")

    # If there are fewer than 3 real candidates, pad from the demo set so
    # Pane 1 can always show a 3-line list during the H1 wiring phase.
    while len(lines) < 3:
        idx = len(lines)
        fallback = (
            "Sarah Chen — GP at Bessemer, leads health AI",
            "Marcus Patel — ex-surgeon, AI advisor at a16z",
            "Priya Shah — founder Medvana, recent bridge",
        )[idx]
        lines.append(f"{idx + 1}) {fallback}")

    body = "Top 3 tonight:\n" + "\n".join(lines)
    return _truncate(body, MAX_LIST_CHARS)


def _h1_drill_in(message: str, candidates: list[dict[str, Any]]) -> str:
    target = _match_candidate(message=message, candidates=candidates)

    if target is None:
        return _truncate(_DEMO_DRILL_MARCUS, MAX_DRILL_CHARS)

    name = (target.get("name") or "Unknown").strip()
    headline = (target.get("headline") or target.get("one_liner") or "").strip()
    posts = list(target.get("recent_posts") or [])
    quote = posts[0] if posts else ""
    quote = re.sub(r"\s+", " ", quote).strip().strip("\"'")
    if quote and len(quote) > 100:
        quote = quote[:97].rstrip() + "…"

    bio = f"{name} — {headline}." if headline else f"{name}."
    if quote:
        body = (
            f"{bio} Recent post: \"{quote}\". "
            f"Open with: \"saw your post — tell me more about that?\""
        )
    else:
        body = (
            f"{bio} Worth grabbing 5 min. "
            f"Open with: \"what are you working on right now?\""
        )

    return _truncate(body, MAX_DRILL_CHARS)


def _h1_rapport(goal: str, candidates: list[dict[str, Any]]) -> str:
    pick: dict[str, Any] | None = None
    for cand in candidates:
        if cand.get("recent_posts"):
            pick = cand
            break

    if pick is None:
        return _truncate(_DEMO_RAPPORT_PRIYA, MAX_RAPPORT_CHARS)

    full_name = (pick.get("name") or "Unknown").strip()
    first = full_name.split()[0] if full_name else "Unknown"
    # If everyone shares the first token (e.g. placeholder "Attendee N"),
    # the first name alone is meaningless — use the full name instead.
    others_share_first = sum(
        1 for c in candidates if (c.get("name") or "").split()[:1] == [first]
    )
    name = full_name if others_share_first > 1 else first
    posts = list(pick.get("recent_posts") or [])
    quote = posts[0] if posts else ""
    quote = re.sub(r"\s+", " ", quote).strip().strip("\"'")
    # Trim to ~10 words for the SMS quote — H2's eval enforces ≥15 chars.
    words = quote.split()
    if len(words) > 12:
        quote = " ".join(words[:12])

    body = (
        f"{name}. Recently posted \"{quote}\". "
        f"Open with that — it's a real interest, not work."
    )
    return _truncate(body, MAX_RAPPORT_CHARS)


def _match_candidate(
    message: str, candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find a candidate whose name uniquely appears in the message.

    Tries a unique discriminator first — the token that distinguishes this
    candidate from others in the list — then falls back to any name token
    that's >=3 chars. This handles both real first names ("Marcus") and
    placeholder names like "Attendee 7" where the digit is the unique part.
    """
    lowered = message.lower()
    if not candidates:
        return None

    name_tokens: list[list[str]] = []
    for cand in candidates:
        name = (cand.get("name") or "").strip().lower()
        tokens = [t for t in re.split(r"\s+", name) if t]
        name_tokens.append(tokens)

    for cand, tokens in zip(candidates, name_tokens):
        if not tokens:
            continue
        unique_tokens = [
            t
            for t in tokens
            if sum(1 for other in name_tokens if t in other) == 1
        ]
        for tok in unique_tokens:
            if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", lowered):
                return cand

    for cand, tokens in zip(candidates, name_tokens):
        for tok in tokens:
            if len(tok) < 3:
                continue
            if re.search(rf"\b{re.escape(tok)}\b", lowered):
                return cand
    return None


__all__ = ["extract_goal", "rank_and_riff"]
