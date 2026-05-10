"""Quick Gemini smoke + quota check.

Usage:
  . .venv/bin/activate && python scripts/gemini_check.py
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
load_dotenv(".env")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import llm_client  # noqa: E402


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("FAIL: GEMINI_API_KEY not set in env")
        return 1

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    os.environ["OPENAI_API_KEY"] = ""

    t0 = time.time()
    out = llm_client.chat(
        messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
        model=model,
        temperature=0.0,
        max_tokens=8,
        timeout_s=4.0,
    )
    ms = int((time.time() - t0) * 1000)
    if not out:
        print(f"FAIL after {ms}ms")
        print("  message: empty/failed response (check logs for HTTP status/quota).")
        print("  diagnosis: key/model/quota issue or transient network error.")
        return 2

    print(f"OK after {ms}ms")
    print(f"  model:    {model}")
    print(f"  reply:    {out[:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
