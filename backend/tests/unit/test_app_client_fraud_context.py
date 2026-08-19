from datetime import UTC, datetime

from app.modules.app_client.service import _evidence_frames, _fraud_scene


def test_visitor_presence_marks_home_visit_even_with_phone_context() -> None:
    evidence = {
        "evidence_chain": [
            {"kind": "phone_call_active", "source": "video"},
            {"kind": "visitor_presence", "source": "video"},
        ]
    }

    assert _fraud_scene(evidence) == "home_visit"


def test_phone_context_marks_telecom_fraud() -> None:
    evidence = {"evidence_chain": [{"kind": "phone_call_active", "source": "video"}]}

    assert _fraud_scene(evidence) == "telecom"


def test_people_count_without_phone_marks_home_visit() -> None:
    evidence = {
        "evidence_chain": [{"kind": "people_count_context", "source": "video", "people_count": 2}]
    }

    assert _fraud_scene(evidence) == "home_visit"


def test_speech_only_fraud_marks_telecom() -> None:
    evidence = {"evidence_chain": [{"kind": "identity_claim", "source": "speech"}]}

    assert _fraud_scene(evidence) == "telecom"


def test_evidence_frames_keep_image_timestamp_pairs_in_order() -> None:
    occurred_at = datetime(2026, 8, 5, 2, 10, tzinfo=UTC)
    evidence = {
        "evidence_frames": [
            {"timestamp": "2026-08-05T02:10:15Z", "image_url": "https://cdn/15.jpg"},
            {"timestamp": "2026-08-05T02:10:00Z", "image_url": "https://cdn/00.jpg"},
        ]
    }

    frames = _evidence_frames(evidence, occurred_at)

    assert [frame.image_url for frame in frames] == [
        "https://cdn/00.jpg",
        "https://cdn/15.jpg",
    ]
    assert frames[1].captured_at.second == 15


def test_evidence_frames_fall_back_to_direct_event_image() -> None:
    occurred_at = datetime(2026, 8, 5, 2, 10, tzinfo=UTC)

    frames = _evidence_frames({"image_url": "https://cdn/event.jpg"}, occurred_at)

    assert len(frames) == 1
    assert frames[0].captured_at == occurred_at
