"""Quick Gemini smoke + quota check.

  python3 scripts/gemini_check.py
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import errors


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("FAIL: GEMINI_API_KEY not set in env")
        return 1

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash-lite"

    t0 = time.time()
    try:
        r = client.models.generate_content(
            model=model,
            contents="Reply with exactly the word: pong",
        )
    except errors.APIError as e:
        ms = int((time.time() - t0) * 1000)
        print(f"FAIL after {ms}ms")
        print(f"  type:    {type(e).__name__}")
        print(f"  status:  {getattr(e, 'code', 'unknown')}")
        print(f"  message: {e}")
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            print()
            print("DIAGNOSIS: free tier quota exhausted.")
            print("FIX: upgrade to pay-as-you-go at https://aistudio.google.com/apikey")
        return 2

    ms = int((time.time() - t0) * 1000)
    print(f"OK after {ms}ms")
    print(f"  model:    {model}")
    print(f"  reply:    {r.text[:80]!r}")
    print(f"  prompt_tokens: {r.usage_metadata.prompt_token_count if r.usage_metadata else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
