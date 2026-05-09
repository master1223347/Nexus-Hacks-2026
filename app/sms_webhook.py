"""POST /sms — Twilio inbound webhook orchestrator.

H1 milestone: hardcoded reply only. H2 wires retrieval/llm/memory.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Form, Response

from app.twilio_client import build_twiml

logger = logging.getLogger("wingman.sms")

router = APIRouter()


@router.post("/sms")
async def sms_webhook(
    From: str = Form(...),
    Body: str = Form(...),
) -> Response:
    """Twilio inbound webhook. Returns TwiML XML."""
    logger.info("sms_in from=%s body_len=%d", From[-4:], len(Body or ""))
    twiml = build_twiml("hello")
    return Response(content=twiml, media_type="application/xml")
