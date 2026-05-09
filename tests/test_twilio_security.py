"""Twilio signature validation tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from app.main import app

AUTH_TOKEN = "test_token_do_not_use_in_prod"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TWILIO_VALIDATE_SIGNATURE", "true")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", AUTH_TOKEN)
    return TestClient(app)


def _sign(url: str, params: dict[str, str]) -> str:
    return RequestValidator(AUTH_TOKEN).compute_signature(url, params)


def test_unsigned_request_is_rejected_403(client) -> None:
    resp = client.post("/sms", data={"From": "+15555550100", "Body": "hi"})
    assert resp.status_code == 403


def test_bogus_signature_is_rejected_403(client) -> None:
    resp = client.post(
        "/sms",
        data={"From": "+15555550100", "Body": "hi"},
        headers={"X-Twilio-Signature": "totally-fake-base64=="},
    )
    assert resp.status_code == 403


def test_valid_signature_passes(client) -> None:
    params = {"From": "+15555550100", "Body": "raising seed for med-tech"}
    url = "http://testserver/sms"
    sig = _sign(url, params)
    resp = client.post("/sms", data=params, headers={"X-Twilio-Signature": sig})
    assert resp.status_code == 200
    assert "<Response>" in resp.text


def test_signature_skipped_when_flag_off(monkeypatch) -> None:
    monkeypatch.setenv("TWILIO_VALIDATE_SIGNATURE", "false")
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    c = TestClient(app)
    resp = c.post("/sms", data={"From": "+15555550100", "Body": "hi"})
    assert resp.status_code == 200
