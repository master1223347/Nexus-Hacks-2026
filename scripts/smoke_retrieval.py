"""Smoke test for find_candidates.

Runs the 5 canonical demo goals, prints top-3 names + a coarse score so we can
eyeball whether the retrieval feels right. If 4 of 5 don't look sensible, edit
attendees.json (richer one_liner / interests) before tuning the retrieval weights.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the repo root importable when invoked as `python scripts/smoke_retrieval.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval import CONTRACT_KEYS, find_candidates  # noqa: E402

GOALS: tuple[str, ...] = (
    "seed med-tech investor",
    "AI engineer",
    "designer",
    "co-founder for fintech",
    "just fun",
)


def _format_row(idx: int, attendee: dict, score: float | None) -> str:
    name = attendee["name"]
    headline = attendee["headline"]
    score_str = f"{score:.2f}" if score is not None else " n/a"
    return f"  {idx}. [{score_str}] {name} — {headline[:90]}"


def _coarse_score(goal: str, attendee: dict) -> float:
    """Crude lexical overlap, just for the printout — not used by retrieval."""
    if not goal.strip():
        return 0.0
    text = " ".join(
        [
            attendee.get("headline", ""),
            attendee.get("one_liner", ""),
            " ".join(attendee.get("interests", [])),
            " ".join(attendee.get("recent_posts", [])),
        ]
    ).lower()
    tokens = {t for t in goal.lower().split() if len(t) > 2}
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in text)
    return hits / len(tokens)


def main() -> int:
    print(f"smoke_retrieval: {len(GOALS)} goals\n")
    failures: list[str] = []
    for goal in GOALS:
        started = time.monotonic()
        results = find_candidates(goal, 3)
        elapsed_ms = (time.monotonic() - started) * 1000

        print(f"goal: {goal!r}  ({elapsed_ms:.0f}ms, {len(results)} results)")
        if not results:
            failures.append(goal)
            print("  (no results)\n")
            continue

        for i, attendee in enumerate(results, 1):
            if set(attendee.keys()) != set(CONTRACT_KEYS):
                failures.append(f"{goal}: contract violation on {attendee.get('name')}")
            print(_format_row(i, attendee, _coarse_score(goal, attendee)))
        print()

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("smoke_retrieval: OK (eyeball the rankings — 4/5 should feel right)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
