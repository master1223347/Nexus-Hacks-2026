"""Smoke tests for the SMS orchestrator.

Hackathon scope — focused on the Pane 1 contract: timeouts, fallback,
structured behavior. No external services touched.
"""

from __future__ import annotations

import asyncio

import pytest

from app import orchestrator
from app.orchestrator import (
    FALLBACK_REPLY,
    LLM_TIMEOUT_S,
    extract_goal,
    handle_sms_turn,
    hash_phone,
)
from app.twilio_client import build_twiml, truncate_reply


def test_hash_phone_is_deterministic_and_short() -> None:
    a = hash_phone("+15555550100")
    b = hash_phone("+15555550100")
    c = hash_phone("+15555550101")
    assert a == b
    assert a != c
    assert len(a) == 8


def test_extract_goal_collapses_whitespace() -> None:
    assert extract_goal("  raising  seed   for  med-tech  ") == "raising seed for med-tech"


def test_truncate_reply_prefers_sentence_boundary() -> None:
    long_text = "First sentence. Second sentence. Third sentence. " * 10
    out = truncate_reply(long_text, limit=80)
    assert len(out) <= 80
    assert out.endswith("…")


def test_build_twiml_wraps_reply() -> None:
    xml = build_twiml("hi there")
    assert "<Response>" in xml and "<Message>hi there</Message>" in xml


@pytest.mark.asyncio
async def test_handle_sms_turn_uses_fallback_on_llm_timeout(monkeypatch) -> None:
    """When the llm call exceeds its budget, the fallback reply ships."""

    def slow_llm(goal, candidates, message, history):
        import time as _t

        _t.sleep(LLM_TIMEOUT_S + 0.5)
        return "should not be returned"

    monkeypatch.setattr(orchestrator, "rank_and_riff", slow_llm)
    monkeypatch.setattr(
        orchestrator, "find_candidates", lambda g, k=10: [{"name": "x"}]
    )

    result = await handle_sms_turn("+15555550100", "raising seed for med-tech")
    assert result.reply == FALLBACK_REPLY
    assert result.goal == "raising seed for med-tech"


@pytest.mark.asyncio
async def test_handle_sms_turn_passes_through_llm_reply(monkeypatch) -> None:
    monkeypatch.setattr(
        orchestrator, "find_candidates", lambda g, k=10: [{"name": "Sarah"}]
    )
    monkeypatch.setattr(
        orchestrator,
        "rank_and_riff",
        lambda goal, cands, msg, hist: f"Top: {cands[0]['name']} for {goal}",
    )

    result = await handle_sms_turn("+15555550101", "Find VCs in fintech")
    assert "Sarah" in result.reply
    assert result.goal == "Find VCs in fintech"
    assert result.n_candidates == 1


@pytest.mark.asyncio
async def test_handle_sms_turn_recovers_when_retrieval_explodes(monkeypatch) -> None:
    def boom(goal, k=10):
        raise RuntimeError("hyperspell on fire")

    captured: dict[str, object] = {}

    def fake_llm(goal, cands, msg, hist):
        captured["n_cands"] = len(cands)
        return "fallback-but-llm-still-talks"

    monkeypatch.setattr(orchestrator, "find_candidates", boom)
    monkeypatch.setattr(orchestrator, "rank_and_riff", fake_llm)

    result = await handle_sms_turn("+15555550102", "find engineers")
    assert captured["n_cands"] == 0
    assert result.reply == "fallback-but-llm-still-talks"
