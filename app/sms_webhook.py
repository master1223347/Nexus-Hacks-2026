"""POST /sms — Twilio inbound webhook.

Pipeline:
  1. Twilio form payload -> From, Body
  2. orchestrator.handle_sms_turn() runs memory + retrieval + llm under
     wall-clock timeouts and emits one structured log line per call.
  3. TwiML XML response.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Form, Response

from app.orchestrator import FALLBACK_REPLY, handle_sms_turn
from app.twilio_client import build_twiml

logger = logging.getLogger("wingman.sms")

router = APIRouter()


@router.post("/sms")
async def sms_webhook(
    From: str = Form(...),
    Body: str = Form(""),
) -> Response:
    """Twilio inbound webhook. Returns TwiML XML."""
    try:
        result = await handle_sms_turn(From, Body)
        reply = result.reply or FALLBACK_REPLY
    except Exception:  # noqa: BLE001 — never break Twilio's 200 contract
        logger.exception("sms_webhook unhandled")
        reply = FALLBACK_REPLY

    return Response(content=build_twiml(reply), media_type="application/xml")
