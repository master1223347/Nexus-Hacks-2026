"""Quick check on the state of data/attendees.json.

  python3 scripts/attendee_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "attendees.json"


def main() -> int:
    if not PATH.exists():
        print(f"MISSING: {PATH}")
        return 1

    raw = PATH.read_text()
    if not raw.strip():
        print(f"EMPTY: {PATH}")
        return 1

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"INVALID JSON: {e}")
        return 1

    if isinstance(data, list):
        attendees = data
    elif isinstance(data, dict):
        attendees = data.get("attendees", [])
    else:
        print(f"UNKNOWN SHAPE: {type(data).__name__}")
        return 1
    n = len(attendees)
    print(f"{n} attendees in {PATH.name}")
    print()

    required = ("name", "headline", "company", "recent_posts", "interests", "one_liner")
    for i, a in enumerate(attendees, 1):
        name = a.get("name") or "(no name)"
        headline = (a.get("headline") or "")[:80]
        posts_count = len(a.get("recent_posts") or [])
        missing = [k for k in required if not a.get(k)]
        flag = "OK" if not missing and posts_count >= 3 else "WARN"
        print(f"  [{flag}] {i}. {name}")
        print(f"        headline: {headline}")
        print(f"        recent_posts: {posts_count}")
        if missing:
            print(f"        missing: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
