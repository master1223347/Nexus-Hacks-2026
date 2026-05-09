"""Twilio webhook signature validation.

Twilio signs every inbound webhook with HMAC-SHA1 over the full request URL
plus a sort+concat of the form fields, keyed by your auth token. The official
twilio-python package implements the algorithm — we just plumb the request
through it and reject mismatches with 403.

Toggleable via TWILIO_VALIDATE_SIGNATURE so local ngrok dev (where the
account SID/token may not be loaded) still works. Production deploys MUST
set this to "true".

Also handles ngrok / InsForge proxies by respecting X-Forwarded-Proto and
the proxy-rewritten Host header so the URL we hash matches the one Twilio
hashed on its side.
"""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Request, status
from twilio.request_validator import RequestValidator

logger = logging.getLogger("wingman.twilio")

SIGNATURE_HEADER = "X-Twilio-Signature"


def _flag_enabled() -> bool:
    return os.environ.get("TWILIO_VALIDATE_SIGNATURE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _reconstruct_url(request: Request) -> str:
    """Rebuild the URL Twilio called.

    FastAPI's request.url already includes scheme/host/path/query, but if we
    sit behind a proxy it may say 'http://internal:8000/...' instead of the
    public 'https://xxx.ngrok.app/sms' Twilio actually hit. Twilio hashes the
    public one, so prefer X-Forwarded-* + Host headers when present.
    """
    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if forwarded_proto and forwarded_host:
        path = request.url.path
        query = f"?{request.url.query}" if request.url.query else ""
        return f"{forwarded_proto}://{forwarded_host}{path}{query}"
    return str(request.url)


async def validate_twilio_signature(request: Request) -> None:
    """FastAPI dependency. Raises 403 on bad/missing signature when enabled."""
    if not _flag_enabled():
        return

    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    if not auth_token:
        logger.error("TWILIO_VALIDATE_SIGNATURE=true but TWILIO_AUTH_TOKEN unset")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="server signature config missing",
        )

    signature = request.headers.get(SIGNATURE_HEADER)
    if not signature:
        logger.warning("twilio.signature missing header")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="missing signature",
        )

    url = _reconstruct_url(request)
    form = await request.form()
    params = {k: v for k, v in form.multi_items()}

    validator = RequestValidator(auth_token)
    if not validator.validate(url, params, signature):
        logger.warning("twilio.signature invalid url=%s", url)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid signature",
        )
