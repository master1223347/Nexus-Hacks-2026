"""TwiML + outbound-REST helpers for SMS / WhatsApp replies.

Inbound webhook replies use TwiML (Twilio routes the response back to the
original sender automatically — no recipient required, no REST call).

Outbound REST calls (e.g. proactive messages from a job) DO need a recipient.
On Twilio, SMS senders look like '+14155551234' but WhatsApp senders look
like 'whatsapp:+14155238886'. Twilio rejects the API call if the channel
prefix is on one side and not the other. `normalize_recipient` makes any
caller robust to that.
"""

from __future__ import annotations

import os

from twilio.twiml.messaging_response import MessagingResponse

WHATSAPP_PREFIX = "whatsapp:"

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


def strip_channel_prefix(phone: str) -> str:
    """Drop the 'whatsapp:' channel prefix so memory keys stay E.164.

    Twilio webhooks deliver `From=whatsapp:+12092378030`. We persist by the
    bare E.164 number so the same identity works whether the user texts SMS
    or WhatsApp later. Outbound replies must re-add the prefix — see
    `normalize_recipient`.
    """
    if not phone:
        return phone
    if phone.startswith(WHATSAPP_PREFIX):
        return phone[len(WHATSAPP_PREFIX) :]
    return phone


def normalize_recipient(to: str, sender: str | None = None) -> str:
    """Make `to` agree with the sender's channel.

    If `sender` (defaults to TWILIO_PHONE_NUMBER env) is a WhatsApp address,
    ensure `to` carries the 'whatsapp:' prefix. Otherwise leave `to` as-is.
    Idempotent — safe to call on already-prefixed values.
    """
    if not to:
        return to
    if sender is None:
        sender = os.environ.get("TWILIO_PHONE_NUMBER", "")
    sender_is_whatsapp = sender.startswith(WHATSAPP_PREFIX)
    if sender_is_whatsapp and not to.startswith(WHATSAPP_PREFIX):
        return f"{WHATSAPP_PREFIX}{to}"
    return to
