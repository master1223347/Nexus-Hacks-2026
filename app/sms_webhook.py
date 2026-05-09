"""POST /sms — Twilio inbound webhook (works for both SMS and WhatsApp).

Pipeline:
  1. Twilio form payload -> From, Body
     (WhatsApp arrives as `From=whatsapp:+1209…` — we strip the channel
     prefix so the orchestrator/memory keys stay bare E.164. The TwiML
     <Response><Message> reply Twilio routes back to the original sender,
     so we don't need to re-add the prefix on the reply path.)
  2. orchestrator.handle_sms_turn() runs memory + retrieval + llm under
     wall-clock timeouts and emits one structured log line per call.
  3. TwiML XML response.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Form, Response

from app.orchestrator import FALLBACK_REPLY, handle_sms_turn
from app.twilio_client import build_twiml, strip_channel_prefix
from app.twilio_security import validate_twilio_signature

logger = logging.getLogger("wingman.sms")

router = APIRouter()


@router.post("/sms", dependencies=[Depends(validate_twilio_signature)])
async def sms_webhook(
    From: str = Form(...),
    Body: str = Form(""),
) -> Response:
    """Twilio inbound webhook. Returns TwiML XML."""
    phone = strip_channel_prefix(From)
    try:
        result = await handle_sms_turn(phone, Body)
        reply = result.reply or FALLBACK_REPLY
    except Exception:  # noqa: BLE001 — never break Twilio's 200 contract
        logger.exception("sms_webhook unhandled")
        reply = FALLBACK_REPLY

    return Response(content=build_twiml(reply), media_type="application/xml")
