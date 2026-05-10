from __future__ import annotations

from app import llm


def test_build_user_payload_includes_all_passed_candidates() -> None:
    candidates = [
        {
            "name": f"Attendee {i}",
            "headline": "ML engineer",
            "company": "Acme",
            "recent_posts": [f"post {i}"],
            "interests": ["ml"],
            "one_liner": "strong ML background",
        }
        for i in range(1, 8)
    ]

    payload = llm._build_user_payload(  # noqa: SLF001 - explicit unit coverage for helper
        mode="drill_in",
        goal="find ML engineers",
        candidates=candidates,
        message="tell me about attendee 7",
        history=[],
    )

    assert "[7] name: Attendee 7" in payload


def test_rank_and_riff_uses_deterministic_fallback_when_llm_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(llm.llm_client, "is_configured", lambda: False)

    candidates = [
        {
            "name": "Buvani Pai",
            "headline": "AI/ML engineer",
            "company": "XDC Global",
            "recent_posts": ["Shipping a RAG eval harness this weekend"],
            "interests": ["ml", "agents"],
            "one_liner": "AI/ML engineer with production RAG work",
        }
    ]

    reply = llm.rank_and_riff(
        goal="people with strong ML backgrounds",
        candidates=candidates,
        message="people with strong ML backgrounds",
        history=[],
    )

    assert "Buvani Pai" in reply
    assert "give me a sec — try again" not in reply


def test_low_signal_post_filter_detects_placeholder_text() -> None:
    assert llm._is_low_signal_post(  # noqa: SLF001
        "No clearly public recent LinkedIn posts found from the available results."
    )
    assert not llm._is_low_signal_post("Shipped a new eval harness for RAG pipelines.")  # noqa: SLF001
