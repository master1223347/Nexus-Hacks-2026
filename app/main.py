"""WingmanAI FastAPI entrypoint.

Routes:
  POST /sms     — Twilio inbound SMS webhook (TwiML response)
  GET  /health  — readiness probe for InsForge + ngrok smoke test
  GET  /version — build metadata to verify the live deploy
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="WingmanAI", version="0.1.0")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/version")
def version() -> dict[str, str]:
    return {
        "app_version": app.version,
        "git_sha": os.environ.get("GIT_SHA", "").strip(),
        "source_version": os.environ.get("SOURCE_VERSION", "").strip(),
        "build_ts": os.environ.get("WINGMAN_BUILD_TS", "").strip(),
    }


from app.sms_webhook import router as sms_router  # noqa: E402

app.include_router(sms_router)
