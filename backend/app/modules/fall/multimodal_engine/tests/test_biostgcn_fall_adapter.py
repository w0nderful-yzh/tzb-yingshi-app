import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from app.modules.fall.multimodal_engine.algorithm_runtime import AdapterContext
from app.modules.fall.multimodal_engine.algorithm_runtime.adapters.biostgcn_fall import BioSTGCNFileAdapter


BACKEND_DIR = Path(__file__).resolve().parents[1]


class BioSTGCNFallAdapterTest(unittest.TestCase):
    def test_frozen_report_becomes_real_algorithm_finding(self) -> None:
        runtime_dir = BACKEND_DIR / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=runtime_dir) as directory:
            root = Path(directory)
            project = root / "project"
            python = root / "python.exe"
            video = root / "input.avi"
            job_dir = root / "job"
            python.write_bytes(b"test")
            video.write_bytes(b"video")
            for fold in range(1, 7):
                checkpoint = (
                    project
                    / "checkpoints"
                    / f"unified_fold{fold:02d}"
                    / "stage2_best.pt"
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"checkpoint")

            def fake_runner(command, **kwargs):
                del kwargs
                report_path = Path(command[command.index("--output") + 1])
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(
                        {
                            "model_version": "biostgcn-stage2-unified-v1-ensemble6",
                            "input": {
                                "shape": [157, 133, 3],
                                "fps_for_event_timing": 25.0,
                                "pose_extraction": {
                                    "detected_frames": 157,
                                    "missing_ratio": 0.0,
                                },
                            },
                            "window_count": 5,
                            "peak": {
                                "start_frame": 15,
                                "end_frame": 105,
                                "risk_score": 0.81,
                                "risk_level": "HIGH",
                                "alert": True,
                                "positive_votes": 5,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            adapter = BioSTGCNFileAdapter(
                project_dir=project,
                python_executable=python,
                job_dir=job_dir,
                process_runner=fake_runner,
            )
            adapter.load()
            adapter.start(AdapterContext(session_id="session", device_id="camera"))
            finding = adapter.consume(video)

            self.assertEqual(finding.event_type, "PRE_FALL_RISK")
            self.assertEqual(finding.risk_level.value, "HIGH")
            self.assertAlmostEqual(finding.risk_score, 0.81)
            self.assertEqual(finding.model_version, adapter.model_version)
            self.assertEqual(adapter.last_report["input"]["shape"], [157, 133, 3])
            evidence = {item.code: item.value for item in finding.evidence}
            self.assertEqual(evidence["ensemble_votes"], 5)
            self.assertEqual(evidence["pose_detected_frames"], 157)
            self.assertEqual(evidence["window_start"], 0.6)
            self.assertEqual(evidence["window_end"], 4.2)


if __name__ == "__main__":
    unittest.main()
