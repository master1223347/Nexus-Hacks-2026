"""TwiML response helpers for outbound SMS replies.

We do not initiate outbound REST calls in the MVP — Twilio's inbound webhook
expects an XML <Response> body, which is a thousand times cheaper than POSTing
back to the Twilio REST API on every reply.
"""

from __future__ import annotations

from twilio.twiml.messaging_response import MessagingResponse

# SMS hard limit per segment is 160 GSM-7 chars, 70 UCS-2 chars. 1-2 segments
# = ~300 chars is the comfortable target the build plan calls for.
MAX_REPLY_CHARS = 300
ELLIPSIS = "…"


def truncate_reply(text: str, limit: int = MAX_REPLY_CHARS) -> str:
    """Shorten a reply to <=limit chars, preferring a sentence boundary.

    Falls back to a hard cut + ellipsis when no boundary exists in the budget.
    """
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text

    budget = max(1, limit - len(ELLIPSIS))
    window = text[:budget]

    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut >= int(budget * 0.5):
        return text[: cut + 1].rstrip() + ELLIPSIS

    space = window.rfind(" ")
    if space >= int(budget * 0.5):
        return text[:space].rstrip() + ELLIPSIS

    return window.rstrip() + ELLIPSIS


def build_twiml(reply_text: str) -> str:
    """Wrap a reply string in valid Twilio MessagingResponse XML."""
    body = truncate_reply(reply_text)
    response = MessagingResponse()
    response.message(body)
    return str(response)
