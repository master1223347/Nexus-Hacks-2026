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
