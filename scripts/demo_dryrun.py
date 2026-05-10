#!/usr/bin/env python3
"""End-to-end demo dry-run for the 3-beat script.

Runs through Pane 1's orchestrator (handle_sms_turn) which composes
retrieval (Pane 2), llm + memory (Pane 3). 5 iterations of:
  1) "I'm raising a seed for med-tech AI."          -> top-3 list
  2) "Tell me about <name from #1>."                 -> drill-in
  3) "Anyone fun to grab a drink with?"              -> rapport

Logs per-message latency, prints p50/p95 at the end. Any reply >7s is
flagged (see Pane 1 build plan risk register).
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.orchestrator import handle_sms_turn  # noqa: E402

# Each demo run uses a unique phone so the in-memory store doesn't leak goals
# between iterations — that's how Twilio would actually segment conversations.
BASE_PHONE = "+15555550{:03d}"

BEAT_1 = "I'm raising a seed for med-tech AI."
BEAT_3 = "Anyone fun to grab a drink with?"

# Try to extract the first attendee name from beat 1's reply. Supports both
# old numbered format ("1) Name — ...") and current sentence format
# ("Name looks strong: ...").
NAME_NUMBERED_RE = re.compile(r"^\s*1\)\s*(?P<name>[^—\-\n]+?)\s*[—\-]", re.MULTILINE)
NAME_SENTENCE_RE = re.compile(
    r"(?P<name>[A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+){0,2})\s+looks\s+strong:",
    re.MULTILINE,
)


def _extract_first_name(reply_text: str) -> str:
    text = reply_text or ""
    m = NAME_NUMBERED_RE.search(text)
    if m:
        return m.group("name").strip()
    m = NAME_SENTENCE_RE.search(text)
    if m:
        return m.group("name").strip()
    # Fallback to demo-shaped reply with "Sarah Chen" / "Marcus Patel" / etc.
    if reply_text and ("Marcus" in reply_text):
        return "Marcus"
    if reply_text and ("Attendee" in reply_text):
        # Pull "Attendee N" from the first line if present.
        m2 = re.search(r"Attendee\s+\d+", reply_text)
        if m2:
            return m2.group(0)
    return "Marcus"


async def _run_once(phone: str) -> list[dict]:
    """Run one 3-beat session. Return per-beat metrics."""
    out: list[dict] = []

    t0 = time.perf_counter()
    r1 = await handle_sms_turn(phone, BEAT_1)
    out.append(
        {
            "beat": 1,
            "msg": BEAT_1,
            "reply": r1.reply,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "n_candidates": r1.n_candidates,
            "goal": r1.goal,
        }
    )

    name = _extract_first_name(r1.reply)
    beat_2 = f"Tell me about {name}."

    t0 = time.perf_counter()
    r2 = await handle_sms_turn(phone, beat_2)
    out.append(
        {
            "beat": 2,
            "msg": beat_2,
            "reply": r2.reply,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "n_candidates": r2.n_candidates,
            "goal": r2.goal,
        }
    )

    t0 = time.perf_counter()
    r3 = await handle_sms_turn(phone, BEAT_3)
    out.append(
        {
            "beat": 3,
            "msg": BEAT_3,
            "reply": r3.reply,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "n_candidates": r3.n_candidates,
            "goal": r3.goal,
        }
    )
    return out


async def _main(iterations: int, verbose: bool) -> int:
    all_results: list[dict] = []

    for i in range(iterations):
        phone = BASE_PHONE.format(100 + i)
        session = await _run_once(phone)
        for entry in session:
            entry["iter"] = i + 1
            entry["phone"] = phone
            all_results.append(entry)
            tag = "OK" if entry["elapsed_ms"] <= 7000 else "SLOW"
            print(
                f"[iter {i+1} beat {entry['beat']} {tag}] "
                f"{entry['elapsed_ms']:>5d}ms  "
                f"cands={entry['n_candidates']:>2d}  "
                f"goal={entry['goal']!r}"
            )
            if verbose:
                preview = entry["reply"].replace("\n", " | ")[:240]
                print(f"           reply: {preview}")

    print()
    if not all_results:
        print("no results")
        return 1

    by_beat: dict[int, list[int]] = {1: [], 2: [], 3: []}
    for entry in all_results:
        by_beat[entry["beat"]].append(entry["elapsed_ms"])

    all_ms = [e["elapsed_ms"] for e in all_results]

    print("--- latency (ms) ---")
    for beat in (1, 2, 3):
        vals = by_beat[beat]
        if not vals:
            continue
        p50 = statistics.median(vals)
        p95 = _percentile(vals, 95)
        print(
            f"beat {beat}: n={len(vals)}  p50={int(p50):>5d}  "
            f"p95={int(p95):>5d}  max={max(vals):>5d}"
        )

    p50 = statistics.median(all_ms)
    p95 = _percentile(all_ms, 95)
    print(
        f"overall: n={len(all_ms)}  p50={int(p50):>5d}  "
        f"p95={int(p95):>5d}  max={max(all_ms):>5d}"
    )

    slow = [e for e in all_results if e["elapsed_ms"] > 7000]
    if slow:
        print(f"\n{len(slow)} reply(ies) >7s — investigate:")
        for s in slow:
            print(f"  iter={s['iter']} beat={s['beat']} ms={s['elapsed_ms']}")
        return 1

    print("\nall replies <=7s — demo timing OK")
    return 0


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--iterations", type=int, default=5)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    return asyncio.run(_main(iterations=args.iterations, verbose=args.verbose))


if __name__ == "__main__":
    raise SystemExit(main())
