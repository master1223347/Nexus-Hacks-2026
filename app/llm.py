"""Wingman LLM orchestrator.

Public surface (Pane 1 imports this directly):
    extract_goal(message)            -> str | None
    rank_and_riff(goal, candidates, message, history) -> str

Strategy:
  - Deterministic router (`_route`) picks the mode from message + candidates.
  - When OPENAI_API_KEY is set: assemble a single composite-prompt LLM call
    via app.llm_client.chat(). The system prompt covers all three modes
    (initial / drill-in / rapport). The user message carries the mode hint,
    so the LLM has both a strong prior and the freedom to override.
  - When the LLM is unavailable, returns NEED_MORE_DATA, or the response
    fails a hard quality check, fall back to the H1 deterministic renderer
    so the demo always replies with something on-brand.
  - On any internal error, return FALLBACK_REPLY rather than crash.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from pydantic import BaseModel

from app import llm_client
from app.prompts.rank import (
    FALLBACK_REPLY,
    LIMITED_MATCHES_PREAMBLE,
    NEED_MORE_DATA,
    SYSTEM_DRILL_IN,
    SYSTEM_INITIAL,
)
from app.prompts.rapport import RAPPORT_FEW_SHOT, SYSTEM_RAPPORT

logger = logging.getLogger("wingman.llm")

# Bio one-liner cap. The system prompt asks the model to keep one_liners
# under 100 chars; we trim defensively at the same boundary, on a word edge,
# to keep "Cosmas Mandikonza — ..., an" truncations off the SMS.
ONE_LINER_CAP = 100

# Detects "Open with: \"<some text>\"" anywhere in the reply, with at least
# 5 chars inside the quotes. Used to enforce the literal-opener rule for
# rapport + drill-in replies.
_OPENER_RE = re.compile(r'Open with:\s*"([^"\n]{5,})"', re.IGNORECASE)


class _RankItem(BaseModel):
    name: str
    one_liner: str


class _RankResponse(BaseModel):
    top_3: list[_RankItem]
    under_filled: bool = False

# SMS budgets (verified by scripts/eval_rapport.py).
# MAX_LIST_CHARS bumped to 480 to fit 3 × 100-char bios + the optional
# "Limited goal-aligned matches…" preamble. Twilio segment-merging handles
# 3-segment messages cleanly.
MAX_LIST_CHARS = 480
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
        mode = _route(message=message or "", candidates=all_cands)

        # Per-mode candidate window:
        #   drill_in -> JUST the named target. Cross-pollination across
        #               attendee posts hallucinates badly otherwise.
        #               Falls back to top-5 if we couldn't match a name.
        #   rapport  -> wider window — rapport pick should not be constrained
        #               to the goal-ranked top 5; matcha/boba people may rank
        #               low for the user's stated goal but high for rapport.
        #   initial  -> top-5 (goal-fit is the right signal here).
        if mode == "drill_in":
            target = _match_candidate(message=message or "", candidates=all_cands)
            candidates_for_pick = (
                [target] if target is not None else all_cands[:MAX_CANDIDATES_FOR_LLM]
            )
        elif mode == "rapport":
            candidates_for_pick = all_cands[:10]
        else:
            candidates_for_pick = all_cands[:MAX_CANDIDATES_FOR_LLM]

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
    """Single-call LLM dispatch. Returns SMS text or None on any failure.

    Model selection (per user instruction):
      initial / drill_in -> gemini-2.5-flash-lite
      rapport            -> gemini-2.5-flash  (escalate to gemini-2.5-pro
                                               only if eval drops below 9/10)
    """
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return None

    # Initial mode: structured JSON via flash-lite, formatted into SMS.
    if mode == "initial":
        return _try_llm_initial(
            goal=goal,
            candidates=candidates,
            message=message,
            history=history,
        )

    # Drill-in & rapport: free-text generation.
    system_prompt = _build_system_prompt(mode=mode)
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

    # Rapport: lower temp + tight token budget. Output is ≤320 chars (~80
    # tokens); 200 leaves slack but cuts wall-clock vs the old 400.
    # Drill-in: ≤480 chars (~120 tokens); cap at 250 for the same reason.
    if mode == "rapport":
        model = llm_client.MODEL_RAPPORT
        temperature = 0.4
        max_tokens = 200
    else:
        # drill_in: bumped 250 -> 350 so the model has room to finish
        # the bio AND emit the literal `Open with: "..."` opener line
        # before max_output_tokens cuts it off mid-bio.
        model = llm_client.MODEL_RANK
        temperature = 0.5
        max_tokens = 350

    reply = llm_client.chat(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
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
    # strictly safer.
    if mode == "rapport" and not _has_verbatim_quote(reply, candidates):
        logger.info("llm_reply.no_verbatim_quote mode=%s", mode)
        return None

    # Drill-in + rapport: enforce the literal "Open with: \"…\"" opener.
    # Topic suggestions like "Open with AI tools" don't land — the user
    # needs a sentence they can speak. We salvage by appending a generic
    # quoted opener if the LLM forgot, rather than discarding the reply,
    # because the bio is usually still good.
    if mode in ("rapport", "drill_in"):
        cap = MAX_RAPPORT_CHARS if mode == "rapport" else MAX_DRILL_CHARS
        reply = _ensure_quoted_opener(
            reply=reply, candidates=candidates, total_cap=cap
        )
        # Cut anything the model emitted AFTER the quoted opener — it
        # tends to add a second bio block that gets clipped mid-word and
        # looks broken on the SMS. The opener is the contract; what
        # follows is noise.
        reply = _truncate_at_opener(reply)

    return reply


def _truncate_at_opener(reply: str) -> str:
    """Cut the reply right after the FIRST `Open with: "…"` line."""
    m = _OPENER_RE.search(reply)
    if not m:
        return reply
    end = m.end()
    # Allow trailing whitespace/newlines but drop everything after the next
    # newline that follows the closing quote.
    trailing = reply[end:]
    nl = trailing.find("\n")
    if nl == -1:
        return reply  # opener is the last line — nothing to trim
    return reply[: end + nl].rstrip()


def _ensure_quoted_opener(
    reply: str,
    candidates: list[dict[str, Any]],
    total_cap: int,
) -> str:
    """Return a reply guaranteed to end with `Open with: "…"` and fit in `total_cap`.

    If the LLM already produced a quoted opener (≥5 chars in the quotes),
    leave it. Otherwise:
      1. Strip any unquoted opener line ("Open with: AI tools").
      2. Build a salvaged quoted opener from the most recently quoted
         snippet in the reply, or the first recent_post we can find.
      3. Trim the bio first so bio + opener fits inside total_cap —
         otherwise the outer `_truncate_for_mode` would clip the opener
         off and we'd ship "..." with no opener at all (the regression
         this code is here to prevent).
    """
    if _OPENER_RE.search(reply):
        return reply

    logger.info("llm_reply.opener.salvaged")

    # Strip any unquoted "Open with: …" line, plus trailing punctuation /
    # ellipsis the model may have left.
    cleaned_bio = re.sub(
        r"\s*Open with:[^\n]*$", "", reply.rstrip(), flags=re.IGNORECASE
    ).rstrip(" .…—")
    if not cleaned_bio:
        cleaned_bio = "Worth grabbing 5 min"

    quoted = re.search(r'"([^"]{15,})"', reply)
    raw_hook = quoted.group(1) if quoted else _first_post_snippet(candidates)
    # Tight hook (~6 words) so the salvaged opener stays conversational.
    hook = " ".join((raw_hook or "").split()[:6]).rstrip(",.;:")

    if hook:
        opener = (
            f'Open with: "Saw your post about {hook} — what got you onto that?"'
        )
    else:
        opener = (
            'Open with: "What\'s the most interesting thing you\'ve been '
            'working on this week?"'
        )

    # Reserve room for the opener (+ ".\n" joiner) inside the cap. Without
    # this, the outer truncate eats the opener and the user sees "...".
    join_overhead = 2  # for ".\n"
    bio_budget = max(40, total_cap - len(opener) - join_overhead)
    if len(cleaned_bio) > bio_budget:
        cleaned_bio = _word_boundary_trim(cleaned_bio, bio_budget)

    return f"{cleaned_bio}.\n{opener}"


def _first_post_snippet(candidates: list[dict[str, Any]]) -> str:
    """Pull a short usable snippet from the first candidate with recent_posts."""
    for cand in candidates:
        for post in cand.get("recent_posts") or []:
            post = (post or "").strip()
            if len(post) >= 20:
                # Trim to ~10 words for the salvaged opener.
                words = post.split()[:10]
                snippet = " ".join(words).rstrip(",.;:")
                return snippet
    return ""


def _try_llm_initial(
    goal: str,
    candidates: list[dict[str, Any]],
    message: str,
    history: list[dict[str, str]],
) -> str | None:
    """Initial-rank mode: structured JSON output, formatted into SMS.

    Returns the formatted SMS reply, or None on any failure (caller falls
    back to the H1 deterministic renderer).
    """
    user_payload = _build_user_payload(
        mode="initial",
        goal=goal,
        candidates=candidates,
        message=message,
        history=history,
    )
    messages = [
        {"role": "system", "content": SYSTEM_INITIAL},
        {"role": "user", "content": user_payload},
    ]

    parsed = llm_client.chat_structured(
        messages=messages,
        schema=_RankResponse,
        model=llm_client.MODEL_RANK,
        temperature=0.3,
        max_tokens=400,
    )
    if parsed is None:
        return None

    # The SDK may hand us a parsed Pydantic model OR a raw dict.
    raw_items = (
        parsed.top_3
        if hasattr(parsed, "top_3")
        else (parsed.get("top_3") if isinstance(parsed, dict) else None)
    )
    under_filled = bool(
        getattr(parsed, "under_filled", False)
        if not isinstance(parsed, dict)
        else parsed.get("under_filled", False)
    )
    if not raw_items:
        return None

    items: list[dict[str, str]] = []
    for it in raw_items:
        name = (
            getattr(it, "name", None)
            if not isinstance(it, dict)
            else it.get("name")
        )
        one_liner = (
            getattr(it, "one_liner", None)
            if not isinstance(it, dict)
            else it.get("one_liner")
        )
        if not name or not one_liner:
            continue
        items.append({"name": str(name).strip(), "one_liner": str(one_liner).strip()})

    if len(items) < 3:
        logger.info("llm_initial.short_list n=%d", len(items))
        return None

    # Reject filler-laden one_liners across the whole reply.
    blob = " ".join(it["one_liner"].lower() for it in items)
    for phrase in _FILLER:
        if phrase in blob:
            logger.info("llm_initial.filler phrase=%r", phrase)
            return None

    header = LIMITED_MATCHES_PREAMBLE if under_filled else ""
    lines = [(header + "Top 3 tonight:").rstrip()]
    for i, it in enumerate(items[:3], start=1):
        one_liner = _word_boundary_trim(
            re.sub(r"\s+", " ", it["one_liner"]), ONE_LINER_CAP
        )
        lines.append(f"{i}) {it['name']} — {one_liner}")

    body = "\n".join(lines)
    return _truncate(body, MAX_LIST_CHARS)


def _word_boundary_trim(text: str, limit: int) -> str:
    """Trim `text` to <=limit chars on a word/sentence boundary.

    Never returns mid-word ("Cosmas Mandikonza — ..., an") and never returns
    text ending on filler joiners ("and", "or", "but"). Falls back to a
    last-space cut if no good boundary exists.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        # Still strip trailing junk in case the model itself overshot
        # complete-sentence discipline.
        return _strip_trailing_joiners(text)

    window = text[:limit]
    # Prefer a sentence-ish boundary first.
    for sep in (". ", "; ", " — ", " - "):
        idx = window.rfind(sep)
        if idx >= int(limit * 0.5):
            return _strip_trailing_joiners(text[:idx].rstrip())

    # Otherwise the last word boundary.
    space = window.rfind(" ")
    if space >= int(limit * 0.5):
        return _strip_trailing_joiners(text[:space].rstrip())

    return _strip_trailing_joiners(window.rstrip())


_TRAILING_JOINERS = {
    "and",
    "or",
    "but",
    "an",
    "a",
    "the",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "with",
}


def _strip_trailing_joiners(text: str) -> str:
    """Drop trailing connector words that signal an abrupt cut."""
    cleaned = text.rstrip(" ,;:.…—-")
    while True:
        parts = cleaned.rsplit(" ", 1)
        if len(parts) < 2:
            break
        if parts[-1].lower() in _TRAILING_JOINERS:
            cleaned = parts[0].rstrip(" ,;:.…—-")
            continue
        break
    return cleaned


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


def _build_system_prompt(mode: str = "rapport") -> str:
    """Build a mode-specific system prompt.

    Single-mode prompts cut input tokens ~2/3 vs the composite, which matters
    for free-tier flash-lite latency. Each mode is self-contained — the
    deterministic router (`_route`) has already chosen the mode by the time
    we hit the LLM, so we don't need the LLM to also do routing.
    """
    if mode == "drill_in":
        return SYSTEM_DRILL_IN
    if mode == "rapport":
        return (
            SYSTEM_RAPPORT
            + "\n\nFew-shot examples (study these, then write yours):\n"
            + RAPPORT_FEW_SHOT
            + "\nOutput ONLY the final SMS text — no JSON, no preamble."
        )
    return SYSTEM_INITIAL


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

    # Drill-in: if we deterministically matched a candidate from the message,
    # tell the LLM exactly who. Without this hint flash-lite often picks the
    # most goal-relevant attendee instead of the one the user named.
    if mode == "drill_in":
        target = _match_candidate(message=message, candidates=candidates)
        if target is not None:
            parts.append(f"drill_target: {target.get('name', 'Unknown').strip()}")

    parts.append("\nCANDIDATES (top-{}):".format(len(candidates)))
    for i, cand in enumerate(candidates[:MAX_CANDIDATES_FOR_LLM], start=1):
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
    """Cap reply length per mode, but never clip the `Open with: "..."` line.

    Drill-in + rapport replies must end in a complete quoted opener. A naive
    char cap eats the closing quote when the bio runs long; we instead
    trim the BIO portion to make room and stitch the opener back on intact.
    """
    if mode == "drill_in":
        return _truncate_protecting_opener(text, MAX_DRILL_CHARS)
    if mode == "rapport":
        return _truncate_protecting_opener(text, MAX_RAPPORT_CHARS)
    return _truncate(text, MAX_LIST_CHARS)


def _truncate_protecting_opener(text: str, cap: int) -> str:
    """Cap to `cap` chars while keeping the `Open with: "..."` line intact."""
    text = (text or "").strip()
    if len(text) <= cap:
        return text

    m = _OPENER_RE.search(text)
    if not m:
        return _truncate(text, cap)

    opener_start = m.start()
    opener_line = text[opener_start : m.end()]
    bio = text[:opener_start].rstrip()

    join = "\n"
    bio_budget = max(40, cap - len(opener_line) - len(join))
    if len(bio) > bio_budget:
        bio = _word_boundary_trim(bio, bio_budget)

    return f"{bio}{join}{opener_line}"


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
    'Priya. She\'s been "live-tweeting my boba shop tier list" all week '
    'and just posted about Laufey at the Greek.\n'
    'Open with: "Saw your boba tier list — what\'s the bar you\'re judging '
    'on, ice or chew?"'
)


def _h1_initial(goal: str, candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return _truncate(_DEMO_TOP3, MAX_LIST_CHARS)

    lines: list[str] = []
    for i, cand in enumerate(candidates[:3], start=1):
        name = (cand.get("name") or "").strip() or "Unknown"
        one_liner = (cand.get("one_liner") or cand.get("headline") or "").strip()
        one_liner = _word_boundary_trim(
            re.sub(r"\s+", " ", one_liner), ONE_LINER_CAP
        )
        if not one_liner:
            one_liner = "in the room tonight"
        lines.append(f"{i}) {name} — {one_liner}")

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

    # Tighten headline so the bio line stays demo-grade.
    headline = _word_boundary_trim(
        re.sub(r"\s+", " ", headline), ONE_LINER_CAP
    )
    bio = f"{name} — {headline}." if headline else f"{name}."
    if quote:
        # Take the first ~6 words of the quote for a tight opener hook.
        hook_words = quote.split()[:6]
        hook = " ".join(hook_words).rstrip(",.;:")
        body = (
            f"{bio} Recent post: \"{quote}\".\n"
            f'Open with: "Saw your post about {hook} — what got you onto that?"'
        )
    else:
        body = (
            f"{bio} Worth grabbing 5 min.\n"
            f'Open with: "What are you actually working on this week?"'
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
        return _truncate(_DEMO_RAPPORT_PRIYA, MAX_RAPPORT_CHARS)

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

    # Tight hook (~6 words) so the salvaged opener stays conversational.
    hook = " ".join(quote.split()[:6]).rstrip(",.;:")
    body = (
        f'{name}. Recently posted "{quote}".\n'
        f'Open with: "Saw your post about {hook} — what got you onto that?"'
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


def warm_llm() -> bool:
    """Pre-warm the Gemini connection. Call from FastAPI startup hook.

    Wraps app.llm_client.warm(). Returns True on a clean ping. Never raises
    — startup must not fail because the model is briefly unreachable.
    See app/llm_client.py::warm for wiring example in app/main.py.
    """
    return llm_client.warm()


__all__ = ["extract_goal", "rank_and_riff", "warm_llm"]
