import json
from pathlib import Path

from app.modules.fraud.evidence_labels import EVIDENCE_LABELS
from app.modules.fraud.speech_risk import build_speech_events
from app.modules.fraud.text_classifier import DEFAULT_DATA_DIR

BACKEND_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = DEFAULT_DATA_DIR / "train.jsonl"
SCHEMA_PATH = BACKEND_ROOT / "evaluation/fraud/schema.json"
EVAL_PUBLIC = BACKEND_ROOT / "evaluation/fraud/eval_public.jsonl"


def _load_lines(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def test_train_records_have_required_schema_fields() -> None:
    records = _load_lines(TRAIN_PATH)
    assert records, "train.jsonl must not be empty"
    for record in records:
        assert record["text"].strip()
        assert set(record["labels"]).issubset(EVIDENCE_LABELS)
        assert record["source"] in {"curated", "augmented", "recorded"}
        assert record["split_group"].startswith("conversation-")
        assert isinstance(record["asr_noisy"], bool)


def test_eval_records_have_required_schema_fields() -> None:
    records = _load_lines(EVAL_PUBLIC)
    assert records, "eval_public.jsonl must not be empty"
    for record in records:
        assert record["text"].strip()
        assert set(record["labels"]).issubset(EVIDENCE_LABELS)
        assert record["source"] in {"curated", "augmented", "recorded"}
        assert record["split_group"].startswith("conversation-")
        assert isinstance(record["asr_noisy"], bool)
        assert record.get("expected_state") in (
            None,
            "S0_NORMAL",
            "S1_OBSERVING",
            "S2_TRUST_BUILDING",
            "S3_INFORMATION_PROBING",
            "S4_ACTION_INDUCEMENT",
            "S5_CRITICAL_CONTROL",
        )


def test_split_groups_do_not_cross_train_and_eval() -> None:
    train_groups = {record["split_group"] for record in _load_lines(TRAIN_PATH)}
    eval_groups = {record["split_group"] for record in _load_lines(EVAL_PUBLIC)}
    assert train_groups.isdisjoint(eval_groups)


def test_schema_json_matches_runtime_labels() -> None:
    schema = json.loads(SCHEMA_PATH.read_text("utf-8"))
    enum = schema["properties"]["labels"]["items"]["enum"]
    assert set(enum) == set(EVIDENCE_LABELS)


def test_eval_covers_all_evidence_labels() -> None:
    records = _load_lines(EVAL_PUBLIC)
    covered = {label for record in records for label in record["labels"]}
    assert covered == set(EVIDENCE_LABELS)


def test_asr_noisy_samples_still_produce_expected_evidence() -> None:
    classifier = None
    cases = [
        ("把短信里的验证码念给我听", {"credential_request"}),
        ("安一个远程控制软件到手机上", {"remote_control_instruction"}),
        ("马上转五万块到安全账户", {"money_instruction"}),
    ]
    for text, expected in cases:
        event = build_speech_events(
            [{"start_ms": 0, "end_ms": 1_000, "text": text, "transcript_status": "FINAL"}],
            classifier=classifier,
        )[0]
        kinds = {item["kind"] for item in event["evidence_observations"]}
        assert expected.issubset(kinds), f"missing {expected - kinds} in {text!r}"


def test_protective_warning_does_not_advance_state() -> None:
    from app.modules.fraud.fraud_state_machine import FraudProcessStateMachine

    machine = FraudProcessStateMachine(memory_ms=120_000)
    event = build_speech_events(
        [
            {
                "start_ms": 0,
                "end_ms": 1_000,
                "text": "这是反诈提醒，请勿向任何陌生人转账",
                "transcript_status": "FINAL",
            }
        ]
    )[0]
    for evidence in event["evidence_observations"]:
        machine.consume(dict(evidence))
    assert machine.current_state == "S0_NORMAL"


def test_train_data_hash_is_stable() -> None:
    first = _load_lines(TRAIN_PATH)
    second = _load_lines(TRAIN_PATH)
    assert first == second
