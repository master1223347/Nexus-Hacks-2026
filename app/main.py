"""WingmanAI FastAPI entrypoint.

Routes:
  POST /sms     — Twilio inbound SMS webhook (TwiML response)
  GET  /health  — readiness probe for InsForge + ngrok smoke test
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


from app.sms_webhook import router as sms_router  # noqa: E402

app.include_router(sms_router)
