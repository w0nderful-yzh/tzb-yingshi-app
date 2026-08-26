from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from app.modules.fall.multimodal_engine.tools.fusion_shadow_compare import analyze


class FusionShadowCompareTest(unittest.TestCase):
    def test_normalizes_camera_medium_and_reports_three_layers(self) -> None:
        rows = [
            {
                "logged_at": "2026-08-14T09:00:00+00:00",
                "risk_state": {
                    "camera": "MEDIUM",
                    "radar": "SUPPRESSED_RECOVERY",
                    "fusion": "WATCH",
                },
                "sync_delta_ms": 20.0,
                "degraded_mode": "NONE",
                "dynamic_risk_score": 0.4,
                "dynamic_risk_level": "WATCH",
                "dynamic_risk_reasons": [{"code": "POSTURE_ANOMALY"}],
                "short_term_fall_score": 0.3,
                "fall_event_status": "NO_EVENT",
                "camera_quality": 0.8,
                "radar_quality": 0.7,
                "camera_processing_latency_ms": 100.0,
                "radar_processing_latency_ms": 10.0,
                "camera_evidence_age_ms": 200.0,
                "radar_evidence_age_ms": 30.0,
                "alignment": {
                    "association_state": "MATCHED",
                    "eligible_for_temporal_association": True,
                    "sync_delta_ms": 15.0,
                    "association_confidence": 0.68,
                    "reason_codes": ["TARGET_ASSOCIATION_MATCHED"],
                },
                "associated_risk_augmentation": {
                    "associated_risk_state": "WATCH",
                    "associated_evidence_state": "CORROBORATED_WATCH",
                    "radar_motion_evidence_strength": "WEAK",
                    "associated_short_term_fall_score": 0.3,
                    "base_camera_score": 0.3,
                    "reason_codes": ["CAMERA_WATCH_RADAR_DESCENT_CONSISTENT"],
                },
            },
            {
                "logged_at": "2026-08-14T09:00:01+00:00",
                "risk_state": {
                    "camera": "LOW",
                    "radar": "WATCH",
                    "fusion": "NORMAL",
                },
                "sync_delta_ms": 40.0,
                "degraded_mode": "RADAR_ONLY",
                "dynamic_risk_score": None,
                "dynamic_risk_level": "UNKNOWN",
                "short_term_fall_score": 0.2,
                "fall_event_status": "UNKNOWN",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            report = analyze(
                path,
                start=datetime(2026, 8, 14, 8, 59, tzinfo=timezone.utc),
                end=None,
                expected_risk="UNLABELLED",
                label="unit",
            )

        self.assertEqual(report["schema_version"], "fusion_shadow_system_audit_v4")
        self.assertEqual(
            report["paths"]["camera_only"]["state_counts"],
            {"WATCH": 1, "NORMAL": 1},
        )
        self.assertEqual(
            report["paths"]["radar_only"]["state_counts"],
            {"NORMAL": 1, "WATCH": 1},
        )
        self.assertEqual(report["sync_delta_ms"]["p50"], 30.0)
        self.assertEqual(
            report["three_layer_risk"]["dynamic_risk"]["available_ratio"],
            0.5,
        )
        self.assertEqual(
            report["three_layer_risk"]["fall_event"]["status_counts"],
            {"NO_EVENT": 1, "UNKNOWN": 1},
        )
        self.assertEqual(
            report["camera_radar_alignment"]["state_counts"],
            {"MATCHED": 1},
        )
        self.assertEqual(report["camera_radar_alignment"]["coverage_ratio"], 0.5)
        self.assertEqual(report["camera_radar_alignment"]["eligible_ratio"], 1.0)
        self.assertEqual(report["camera_radar_alignment"]["sync_delta_ms"]["p50"], 15.0)
        self.assertEqual(report["associated_evidence_coverage_ratio"], 0.5)
        self.assertEqual(
            report["camera_led_associated_evidence"][
                "camera_score_invariant_mismatch_count"
            ],
            0,
        )
        self.assertEqual(
            report["paths"]["alignment_camera_led_radar_evidence"]["state_counts"],
            {"WATCH": 1, "UNKNOWN": 1},
        )


if __name__ == "__main__":
    unittest.main()
