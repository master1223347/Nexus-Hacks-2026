#!/usr/bin/env python3
"""Programmatic quality bar for the rapport reply.

Runs rank_and_riff() against every attendee in data/attendees.json with a
canned rapport-style query and asserts:
  (a) Reply contains a substring of length >=15 chars from one of the
      candidate's recent_posts (verbatim quote).
  (b) Reply does NOT match any phrase from the FILLER blocklist.
  (c) Reply length <= MAX (320 rapport / 480 drill-in).
  (d) Reply does not contain the NEED_MORE_DATA sentinel.

Per-attendee failure prints attendee name + reply + which check failed.

H1 status: this file is fully wired so once data/attendees.json has real
candidates, you can run it without touching code. With the H1 hardcoded
renderer, expect (a) to pass for the verbatim slice taken from recent_posts
and (c) to pass; (b)/(d) only matter once H2's LLM lands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running as `python scripts/eval_rapport.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm import (  # noqa: E402
    MAX_DRILL_CHARS,
    MAX_RAPPORT_CHARS,
    rank_and_riff,
)
from app.prompts.rank import NEED_MORE_DATA  # noqa: E402

ATTENDEES_PATH = ROOT / "data" / "attendees.json"

FILLER = (
    "works in tech",
    "passionate about",
    "interested in",
    "loves innovation",
    "in the space of",
    "thought leader",
    "passionate",
    "excited about",
)

# Rapport-shaped prompts the eval rotates through. Each attendee is run
# once per query so per-prompt regressions are visible in the output.
RAPPORT_QUERIES = (
    "anyone fun to grab a drink with?",
    "who should I grab coffee with?",
    "anyone chill in the room?",
)


def _verbatim_quote_present(reply: str, recent_posts: list[str]) -> bool:
    """True if `reply` contains a >=15-char verbatim slice from any post."""
    if not reply or not recent_posts:
        return False
    for post in recent_posts:
        post = (post or "").strip()
        if len(post) < 15:
            continue
        # Slide a 15-char window. Stop early on the first match.
        for i in range(0, len(post) - 14):
            window = post[i : i + 15]
            if window in reply:
                return True
    return False


def _filler_present(reply: str) -> str | None:
    lowered = (reply or "").lower()
    for phrase in FILLER:
        if phrase in lowered:
            return phrase
    return None


def _load_attendees() -> list[dict]:
    if not ATTENDEES_PATH.exists() or ATTENDEES_PATH.stat().st_size == 0:
        return []
    raw = json.loads(ATTENDEES_PATH.read_text())
    if isinstance(raw, dict) and "attendees" in raw:
        return list(raw["attendees"])
    if isinstance(raw, list):
        return list(raw)
    raise SystemExit(f"unrecognized attendees.json shape at {ATTENDEES_PATH}")


def evaluate(attendees: list[dict], goal: str, verbose: bool) -> tuple[int, int]:
    """Return (n_pass, n_total)."""
    if not attendees:
        print(
            "[skip] data/attendees.json is empty — populate it from Pane 2 "
            "before running eval_rapport"
        )
        return (0, 0)

    n_pass = 0
    n_total = 0

    for cand in attendees:
        name = cand.get("name", "?")
        recent_posts = list(cand.get("recent_posts") or [])

        for query in RAPPORT_QUERIES:
            n_total += 1

            # Put the target candidate first so the H1 renderer picks them.
            shuffled = [cand] + [c for c in attendees if c is not cand]

            reply = rank_and_riff(
                goal=goal,
                candidates=shuffled,
                message=query,
                history=[],
            )

            failures: list[str] = []

            if NEED_MORE_DATA in reply:
                failures.append("NEED_MORE_DATA sentinel present")
            if len(reply) > MAX_RAPPORT_CHARS:
                failures.append(
                    f"length {len(reply)} > {MAX_RAPPORT_CHARS}"
                )
            filler = _filler_present(reply)
            if filler:
                failures.append(f"filler phrase: {filler!r}")
            if not _verbatim_quote_present(reply, recent_posts):
                failures.append("no >=15-char verbatim quote from recent_posts")

            if failures:
                print(f"FAIL {name} :: query={query!r}")
                print(f"  reply: {reply!r}")
                for f in failures:
                    print(f"   - {f}")
            else:
                n_pass += 1
                if verbose:
                    print(f"PASS {name} :: query={query!r}")

    return (n_pass, n_total)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--goal",
        default="raising a seed for med-tech AI",
        help="Networking goal passed to rank_and_riff()",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    attendees = _load_attendees()
    n_pass, n_total = evaluate(attendees, goal=args.goal, verbose=args.verbose)

    if n_total == 0:
        return 0  # nothing to evaluate yet (H1)

    pct = (n_pass / n_total) * 100
    print(f"\n{n_pass}/{n_total} passed ({pct:.0f}%)")

    # Per H2 milestone: bar is 9/10 attendees passing rapport. Use a
    # proportional threshold so the eval scales with attendee count and
    # with the number of rapport queries.
    threshold = 0.9
    return 0 if (n_pass / n_total) >= threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
