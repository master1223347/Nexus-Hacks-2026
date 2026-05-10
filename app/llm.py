"""Wingman LLM orchestrator.

Public surface (Pane 1 imports this directly):
    extract_goal(message)            -> str | None
    rank_and_riff(goal, candidates, message, history) -> str

Strategy:
  - Deterministic router (`_route`) picks the mode from message + candidates.
  - When an LLM provider key is set: assemble a single composite-prompt LLM
    call via app.llm_client.chat(). The system prompt covers all three modes
    (initial / drill-in / rapport). The user message carries the mode hint,
    so the LLM has both a strong prior and the freedom to override.
  - When the LLM is unavailable, returns NEED_MORE_DATA, or the response
    fails a hard quality check, fall back to the H1 deterministic renderer
    so the demo always replies with something on-brand.
  - On any internal error, return FALLBACK_REPLY rather than crash.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app import llm_client
from app.prompts.rank import (
    FALLBACK_REPLY,
    NEED_MORE_DATA,
    SYSTEM_DRILL_IN,
    SYSTEM_INITIAL,
)
from app.prompts.rapport import RAPPORT_FEW_SHOT, SYSTEM_RAPPORT

logger = logging.getLogger("wingman.llm")

# SMS budgets (verified by scripts/eval_rapport.py).
MAX_LIST_CHARS = 320
MAX_DRILL_CHARS = 480
MAX_RAPPORT_CHARS = 320

# Cap how many candidates we hand the LLM (token cost). Pane 2 returns up to 10.
MAX_CANDIDATES_FOR_LLM = 5
# Cap how many history entries we hand the LLM.
MAX_HISTORY_FOR_LLM = 6

# Generic phrases the LLM must avoid. Detected post-call; on hit we re-render.
_FILLER = (
    "works in tech",
    "passionate about",
    "interested in",
    "loves innovation",
    "in the space of",
    "thought leader",
    "excited about",
)

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

    Heuristic-only: if any goal-verb appears, treat the whole (cleaned) message
    as the goal. Most demo phrasings ("I'm raising a seed for med-tech AI")
    satisfy this; an LLM call would just add latency and cost.
    """
    if not message or not message.strip():
        return None

    text = message.strip()
    lowered = text.lower()

    if not any(kw in lowered for kw in _GOAL_VERBS):
        return None

    cleaned = re.sub(r"\s+", " ", text).strip(" .!?…")
    return cleaned or None


# ---------------------------------------------------------------------------
# Special-intent constants and patterns (small_talk + inappropriate)
# ---------------------------------------------------------------------------

SMALL_TALK_REPLY = (
    "hey! tell me what you're hunting and i'll point you somewhere — try "
    "'find me ML engineers', 'anyone from CMU', or 'anyone fun to grab a "
    "drink with'."
)

INAPPROPRIATE_REPLY = (
    "i don't filter on appearance — but tell me what you actually want from "
    "the room and i'll find them. e.g. 'i'm raising a seed' or 'i need a "
    "technical cofounder'."
)

_SMALL_TALK_PATTERNS = (
    re.compile(
        r"^\s*(hi|hello|hey|yo|sup|wassup|wsp|howdy|hiya|aloha|"
        r"hi\s+\w+|hello\s+\w+|hey\s+\w+)[\s!?.,]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(what'?s\s+up|good\s+(morning|afternoon|evening|night))[\s!?.,]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(thanks|thank\s+you|ty|tysm|cool|nice|got\s+it|"
        r"ok|okay|k|kk|lol|haha|lmao|nvm|never\s*mind)[\s!?.,]*$",
        re.IGNORECASE,
    ),
)

_INAPPROPRIATE_PATTERNS = (
    re.compile(r"\bbaddies?\b", re.IGNORECASE),
    re.compile(r"\bcuties?\b", re.IGNORECASE),
    re.compile(r"\bhotties?\b", re.IGNORECASE),
    re.compile(
        r"\bhot\s+(?:girls?|guys?|men|women|chicks?|babes?|people|ones?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bhook[- ]?up\b", re.IGNORECASE),
    re.compile(r"\btinder\b", re.IGNORECASE),
)


def _is_small_talk(message: str) -> bool:
    if not message or not message.strip():
        return True
    return any(rgx.search(message) for rgx in _SMALL_TALK_PATTERNS)


def _is_inappropriate(message: str) -> bool:
    if not message:
        return False
    return any(rgx.search(message) for rgx in _INAPPROPRIATE_PATTERNS)


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

    Falls back to a safe on-topic string on any internal error.
    """
    try:
        all_cands = list(candidates or [])
        recent_history = list(history or [])[-MAX_HISTORY_FOR_LLM:]

        # Special intents short-circuit before retrieval/LLM.
        if _is_inappropriate(message or ""):
            return INAPPROPRIATE_REPLY
        if _is_small_talk(message or ""):
            return SMALL_TALK_REPLY

        mode = _route(message=message or "", candidates=all_cands)

        # Drill-in needs the full list so users can name anyone, not just the
        # top 5. Other modes operate on the LLM-budget slice.
        candidates_for_pick = (
            all_cands if mode == "drill_in" else all_cands[:MAX_CANDIDATES_FOR_LLM]
        )

        logger.info(
            "llm_route mode=%s goal_set=%s n_candidates=%d hist=%d",
            mode,
            bool(goal),
            len(all_cands),
            len(recent_history),
        )

        # Try the LLM. _try_llm() returns None when the call is unavailable
        # or the output fails a hard quality check.
        llm_reply = _try_llm(
            mode=mode,
            goal=goal or "",
            candidates=candidates_for_pick,
            message=message or "",
            history=recent_history,
        )
        if llm_reply is not None:
            return _truncate_for_mode(llm_reply, mode)

        # H1 deterministic fallback path.
        return _h1_render(
            mode=mode, goal=goal or "", candidates=all_cands, message=message or ""
        )
    except Exception:  # pragma: no cover — defensive; demo must never 500
        logger.exception("rank_and_riff failed; returning FALLBACK_REPLY")
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
# LLM path
# ---------------------------------------------------------------------------


def _try_llm(
    mode: str,
    goal: str,
    candidates: list[dict[str, Any]],
    message: str,
    history: list[dict[str, str]],
) -> str | None:
    """Single-call LLM dispatch. Returns SMS text or None on any failure."""
    if not llm_client.is_configured():
        return None

    system_prompt = _build_system_prompt()
    user_payload = _build_user_payload(
        mode=mode,
        goal=goal,
        candidates=candidates,
        message=message,
        history=history,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]

    # Rapport demands the most discipline (verbatim quote). Drop temp slightly
    # to bias the model toward copying from recent_posts rather than improvising.
    temperature = 0.4 if mode == "rapport" else 0.5

    reply = llm_client.chat(messages=messages, temperature=temperature, max_tokens=400)
    if reply is None:
        return None

    reply = reply.strip()
    if not reply or NEED_MORE_DATA in reply:
        logger.info("llm_reply.need_more_data mode=%s", mode)
        return None

    # Reject filler-laden replies; rerender deterministically rather than ship.
    lowered = reply.lower()
    for phrase in _FILLER:
        if phrase in lowered:
            logger.info("llm_reply.filler mode=%s phrase=%r", mode, phrase)
            return None

    # Rapport: enforce the verbatim-quote rule at runtime. If the LLM didn't
    # actually quote a recent_post, the H1 renderer (which always quotes) is
    # strictly safer — even if its prose is plainer.
    if mode == "rapport" and not _has_verbatim_quote(reply, candidates):
        logger.info("llm_reply.no_verbatim_quote mode=%s", mode)
        return None

    return reply


def _has_verbatim_quote(reply: str, candidates: list[dict[str, Any]]) -> bool:
    """True if `reply` contains a >=15-char verbatim slice from any candidate's
    recent_posts. Mirrors the bar enforced by scripts/eval_rapport.py."""
    if not reply:
        return False
    for cand in candidates:
        for post in cand.get("recent_posts") or []:
            post = (post or "").strip()
            if len(post) < 15:
                continue
            for i in range(0, len(post) - 14):
                if post[i : i + 15] in reply:
                    return True
    return False


def _build_system_prompt() -> str:
    """Compose the three per-mode rule blocks plus a routing block."""
    routing = (
        "Routing:\n"
        "- If the user names a specific attendee from CANDIDATES, or starts "
        "with phrases like 'tell me about <name>' / 'who is <name>' / 'what "
        "about <name>' -> mode = DRILL_IN.\n"
        "- Else if the user asks for someone fun / casual / interesting / "
        "chill / cool / good for coffee or drinks -> mode = RAPPORT.\n"
        "- Otherwise -> mode = INITIAL_RANK.\n"
        "The user message includes a 'mode_hint' line — trust it unless the "
        "message clearly contradicts it.\n"
    )
    style = (
        "Voice and style:\n"
        "- Sound like a sharp, human friend over text.\n"
        "- Use light slang sparingly (0-2 casual terms).\n"
        "- You may use one playful jab max (light roast energy), but keep it "
        "respectful and non-abusive.\n"
        "- Never insult identity, appearance, health, or protected traits.\n"
        "- Keep replies in sentence/paragraph format. No bullet points or "
        "numbered lists.\n"
    )

    return "\n\n".join(
        [
            "You are WingmanAI, a real-time networking copilot delivered over SMS.",
            "You operate in exactly one of three modes per request.",
            routing,
            style,
            SYSTEM_INITIAL,
            SYSTEM_DRILL_IN,
            SYSTEM_RAPPORT,
            "Few-shot examples for RAPPORT (study these, then write yours):\n"
            + RAPPORT_FEW_SHOT,
            "Output ONLY the final SMS text — no JSON, no preamble, no commentary "
            "about which mode you chose.",
        ]
    )


def _build_user_payload(
    mode: str,
    goal: str,
    candidates: list[dict[str, Any]],
    message: str,
    history: list[dict[str, str]],
) -> str:
    """Assemble the per-request user message: goal, candidates, history, msg.

    Recent_posts come first inside each candidate block — recency bias is real
    and we want the LLM scanning quotable material before rolling up to bio.
    """
    parts: list[str] = []
    parts.append(f"goal: {goal or '(unknown)'}")
    parts.append(f"mode_hint: {mode.upper()}")

    parts.append("\nCANDIDATES (top-{}):".format(len(candidates)))
    for i, cand in enumerate(candidates, start=1):
        name = (cand.get("name") or "Unknown").strip()
        headline = (cand.get("headline") or "").strip()
        company = (cand.get("company") or "").strip()
        one_liner = (cand.get("one_liner") or "").strip()
        interests = cand.get("interests") or []
        recent_posts = [p for p in (cand.get("recent_posts") or []) if p]

        parts.append(f"\n[{i}] name: {name}")
        if recent_posts:
            parts.append("  recent_posts:")
            for post in recent_posts[:5]:
                parts.append(f'    - "{post}"')
        if headline:
            parts.append(f"  headline: {headline}")
        if company:
            parts.append(f"  company: {company}")
        if one_liner:
            parts.append(f"  one_liner: {one_liner}")
        if interests:
            parts.append(f"  interests: {', '.join(str(x) for x in interests[:6])}")

    if history:
        parts.append("\nRECENT HISTORY:")
        for entry in history[-MAX_HISTORY_FOR_LLM:]:
            role = entry.get("role", "?")
            content = (entry.get("content") or "").strip()
            if content:
                parts.append(f"  {role}: {content}")

    parts.append(f"\nCURRENT MESSAGE: {message}")
    parts.append(
        "\nWrite the SMS reply now. Follow the rules of the mode_hint. If "
        "RAPPORT and no recent_post has a quotable specific phrase, output "
        f"only: {NEED_MORE_DATA}"
    )

    return "\n".join(parts)


def _truncate_for_mode(text: str, mode: str) -> str:
    if mode == "drill_in":
        return _truncate(text, MAX_DRILL_CHARS)
    if mode == "rapport":
        return _truncate(text, MAX_RAPPORT_CHARS)
    return _truncate(text, MAX_LIST_CHARS)


# ---------------------------------------------------------------------------
# H1 deterministic fallback renderers
# ---------------------------------------------------------------------------


def _h1_render(
    mode: str,
    goal: str,
    candidates: list[dict[str, Any]],
    message: str,
) -> str:
    if mode == "rapport":
        return _h1_rapport(goal=goal, candidates=candidates[:MAX_CANDIDATES_FOR_LLM])
    if mode == "drill_in":
        return _h1_drill_in(message=message, candidates=candidates)
    return _h1_initial(
        goal=goal, candidates=candidates[:MAX_CANDIDATES_FOR_LLM]
    )


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[: max(1, limit - 1)].rstrip()
    return cut + "…"

def _h1_initial(goal: str, candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        ask = goal or "your goal"
        return _truncate(
            f"I checked the room and don't have strong matches for \"{ask}\" yet. "
            "You're being a little vague, so give me role + stage + sector and "
            "I'll rerank fast.",
            MAX_LIST_CHARS,
        )

    picks: list[str] = []
    for cand in candidates[:3]:
        name = (cand.get("name") or "").strip() or "Unknown"
        one_liner = (cand.get("one_liner") or cand.get("headline") or "").strip()
        one_liner = re.sub(r"\s+", " ", one_liner)[:80]
        if not one_liner:
            one_liner = "in the room tonight"
        picks.append(f"{name} looks strong: {one_liner}.")

    body = " ".join(picks)
    if len(picks) < 3:
        body = (
            f"{body} I'm short on high-signal options right now, so send one "
            "more filter and I'll tighten this up."
        )

    return _truncate(body, MAX_LIST_CHARS)


def _h1_drill_in(message: str, candidates: list[dict[str, Any]]) -> str:
    target = _match_candidate(message=message, candidates=candidates)

    if target is None:
        if candidates:
            names = ", ".join(
                (c.get("name") or "Unknown").strip() or "Unknown"
                for c in candidates[:3]
            )
            return _truncate(
                f"I can't tell who you mean, and I'm not a mind reader. "
                f"Pick one name exactly: {names}.",
                MAX_DRILL_CHARS,
            )
        return _truncate(
            "I don't have attendee data loaded yet, so I can't drill in on a "
            "person. Send your goal again and I'll refresh the list.",
            MAX_DRILL_CHARS,
        )

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


_PERSONAL_SIGNAL = (
    "boba",
    "matcha",
    "coffee",
    "concert",
    "music",
    "weekend",
    "saturday",
    "sunday",
    "evening",
    "tonight",
    "ergodox",
    "keyboard",
    "sourdough",
    "hike",
    "hiking",
    "running",
    "soccer",
    "drank",
    "café",
    "cafe",
    "stonemill",
    "tea",
    "ramen",
    "dinner",
    "lunch",
)


def _rapport_score(cand: dict[str, Any]) -> tuple[int, int]:
    """Higher = more "fun to grab a drink with" signal.

    Returns (personal_signal_hits, post_count). The first wins ties; the
    second is the tiebreaker (more posts = more material).
    """
    posts = list(cand.get("recent_posts") or [])
    if not posts:
        return (0, 0)
    blob = " ".join(p for p in posts if p).lower()
    interests_blob = " ".join(str(x) for x in (cand.get("interests") or [])).lower()
    haystack = f"{blob} {interests_blob}"
    hits = sum(1 for kw in _PERSONAL_SIGNAL if kw in haystack)
    return (hits, len(posts))


def _pick_rapport_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the candidate with the strongest personal/casual signal.

    Falls back to "first with recent_posts" so we always return something
    when no candidate has personal signal — better a plain quote than nothing.
    """
    best: dict[str, Any] | None = None
    best_score = (-1, -1)
    for cand in candidates:
        if not cand.get("recent_posts"):
            continue
        score = _rapport_score(cand)
        if score > best_score:
            best_score = score
            best = cand
    return best


def _pick_rapport_post(posts: list[str]) -> str:
    """Pick the recent_post that's most personal — concrete activity beats
    work hot-takes, even when both are quotable."""
    best = posts[0] if posts else ""
    best_hits = -1
    for post in posts:
        text = (post or "").lower()
        hits = sum(1 for kw in _PERSONAL_SIGNAL if kw in text)
        if hits > best_hits:
            best_hits = hits
            best = post
    return best or ""


def _h1_rapport(goal: str, candidates: list[dict[str, Any]]) -> str:
    pick = _pick_rapport_candidate(candidates)

    if pick is None:
        if candidates:
            first = (candidates[0].get("name") or "Unknown").strip() or "Unknown"
            return _truncate(
                f"{first} could work, but I don't have enough personal signal to "
                "make it fun yet. Give me another vibe word and I'll find a better hit.",
                MAX_RAPPORT_CHARS,
            )
        return _truncate(
            "No social signal to work with yet. Tell me the vibe you want and "
            "I'll try again with fresh matches.",
            MAX_RAPPORT_CHARS,
        )

    full_name = (pick.get("name") or "Unknown").strip()
    first = full_name.split()[0] if full_name else "Unknown"
    others_share_first = sum(
        1 for c in candidates if (c.get("name") or "").split()[:1] == [first]
    )
    name = full_name if others_share_first > 1 else first
    posts = list(pick.get("recent_posts") or [])
    quote = _pick_rapport_post(posts)
    quote = re.sub(r"\s+", " ", quote).strip().strip("\"'")
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
