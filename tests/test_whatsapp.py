"""WhatsApp Sandbox prefix handling — inbound strip, outbound normalize."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from app import orchestrator
from app.main import app
from app.twilio_client import (
    WHATSAPP_PREFIX,
    normalize_recipient,
    strip_channel_prefix,
)


def test_strip_channel_prefix_removes_whatsapp() -> None:
    assert strip_channel_prefix("whatsapp:+12092378030") == "+12092378030"


def test_strip_channel_prefix_idempotent_on_bare_e164() -> None:
    assert strip_channel_prefix("+12092378030") == "+12092378030"


def test_strip_channel_prefix_handles_empty() -> None:
    assert strip_channel_prefix("") == ""


def test_normalize_recipient_adds_prefix_when_sender_is_whatsapp() -> None:
    out = normalize_recipient("+12092378030", sender="whatsapp:+14155238886")
    assert out == "whatsapp:+12092378030"


def test_normalize_recipient_passthrough_when_sender_is_sms() -> None:
    assert normalize_recipient("+12092378030", sender="+14155238886") == "+12092378030"


def test_normalize_recipient_idempotent() -> None:
    out = normalize_recipient("whatsapp:+12092378030", sender="whatsapp:+14155238886")
    assert out == "whatsapp:+12092378030"


def test_normalize_recipient_uses_env_when_sender_omitted(monkeypatch) -> None:
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "whatsapp:+14155238886")
    assert normalize_recipient("+12092378030") == "whatsapp:+12092378030"


def test_inbound_whatsapp_webhook_uses_bare_phone_for_memory(monkeypatch) -> None:
    """The orchestrator/memory layer should never see the 'whatsapp:' prefix."""
    captured: dict[str, str] = {}

    def fake_set_goal(phone: str, goal: str) -> None:
        captured["phone"] = phone

    monkeypatch.setattr(orchestrator, "set_goal", fake_set_goal)
    monkeypatch.setattr(orchestrator, "get_goal", lambda phone: None)
    monkeypatch.setattr(orchestrator, "get_history", lambda phone: [])
    monkeypatch.setattr(orchestrator, "append_history", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "find_candidates", lambda goal, k=10: [])
    monkeypatch.setattr(orchestrator, "rank_and_riff", lambda *a, **k: "ok")

    monkeypatch.setenv("TWILIO_VALIDATE_SIGNATURE", "false")
    client = TestClient(app)
    resp = client.post(
        "/sms",
        data={"From": "whatsapp:+12092378030", "Body": "raising seed"},
    )
    assert resp.status_code == 200
    assert captured["phone"] == "+12092378030", (
        f"orchestrator saw {captured['phone']!r}; expected bare E.164"
    )


def test_signed_whatsapp_webhook_validates_with_full_prefixed_from(monkeypatch) -> None:
    """Signature is computed over the wire payload (with prefix). Server must
    not strip before validating, only after."""
    auth_token = "test_token_xyz"
    monkeypatch.setenv("TWILIO_VALIDATE_SIGNATURE", "true")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", auth_token)
    monkeypatch.setattr(orchestrator, "get_goal", lambda phone: None)
    monkeypatch.setattr(orchestrator, "get_history", lambda phone: [])
    monkeypatch.setattr(orchestrator, "set_goal", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "append_history", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "find_candidates", lambda goal, k=10: [])
    monkeypatch.setattr(orchestrator, "rank_and_riff", lambda *a, **k: "ok")

    client = TestClient(app)
    params = {"From": "whatsapp:+12092378030", "Body": "hi"}
    url = "http://testserver/sms"
    sig = RequestValidator(auth_token).compute_signature(url, params)
    resp = client.post("/sms", data=params, headers={"X-Twilio-Signature": sig})
    assert resp.status_code == 200


def test_whatsapp_prefix_constant_is_lowercase() -> None:
    assert WHATSAPP_PREFIX == "whatsapp:"
