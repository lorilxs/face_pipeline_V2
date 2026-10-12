#!/usr/bin/env python3
"""Stage 5 face-resolution gate end-to-end smoke over ``test_data/stage5_tset``.

Runs the *real* ``Stage5FaceDetection`` stage (real SCRFD model, real FFmpeg
frame reader) on the 6 A-F validation fixtures, with
``configs/config.yaml``'s ``stage_05_face_resolution_filter.enabled`` set to
``true``.  The video-level gate must

    REJECT  A  (BV11bvNzhEBn)       -- measured SCRFD tight bbox short edge 55..71
    PASS    F  (BV125411Y7J8)       -- 119..152
    PASS    B  (BV12mVp61EEH)       -- 113..124
    PASS    C  (douyin)             -- 76..81
    PASS    D  (CNFACE_00001)       -- 193..224
    PASS    E  (vivihead)           -- 207..230

The only thing that differs from a full Stage0-5 run is that we point
``_resolve_video_path`` straight at the fixture MP4 (these are already the
face-present clips Stage 5 would see); every other code path is real: config
load, SCRFD loading, frame decode, ``FaceDetector``, ``classify_detected_faces``
and ``evaluate_video``.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path("/mnt/home/lxs/face_pipeline/Face_Pipeline_V2")
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from face_pipeline_v2.config.config_loader import ConfigLoader  # noqa: E402
from face_pipeline_v2.data.models import ClipInfo  # noqa: E402
from face_pipeline_v2.models.model_manager import ModelManager  # noqa: E402
from face_pipeline_v2.models.wrappers.scrfd import SCRFDWrapper  # noqa: E402
from face_pipeline_v2.pipeline.context import PipelineContext  # noqa: E402
from face_pipeline_v2.runtime.video_reader import FFmpegVideoReader  # noqa: E402
from face_pipeline_v2.stages.stage_05_face_detection.stage import (  # noqa: E402
    Stage5FaceDetection,
)
from face_pipeline_v2.utils.exceptions import FaceResolutionFilterError  # noqa: E402
from face_pipeline_v2.utils.logger import PipelineLogger  # noqa: E402

# filename -> (label, expected_gate_passed)
FIXTURES = {
    "BV11bvNzhEBn_scene_0009_clip_0000_faces_only.mp4": ("A", False),
    "BV125411Y7J8_scene_0008_clip_0000_faces_only.mp4": ("F", True),
    "BV12mVp61EEH_scene_0003_clip_0000_faces_only.mp4": ("B", True),
    "MS4wLjABAAAA1NdKwPxv_v0200fg10000d9jjg1nog65inotge4pg_scene_0006_clip_0000_faces_only.mp4": ("C", True),
    "CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000_faces_only.mp4": ("D", True),
    "15_scene_0000_clip_0000.mp4": ("E", True),
}


def count_frames(path: Path) -> int:
    """Exact decodable frame count using the same FFmpeg reader as Stage 5."""
    reader = FFmpegVideoReader(path)
    reader.start()  # no -vframes -> decode until EOF
    n = 0
    while reader.read_next() is not None:
        n += 1
    reader.close()
    return n


def _largest_bbox(dets):
    best = None
    best_area = -1.0
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = d
    return best


def measure_short_sides(path: Path, wrapper) -> dict:
    """Reference measurement of the SCRFD tight-bbox short edge distribution."""
    cap = cv2.VideoCapture(str(path))
    shorts = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        best = _largest_bbox(wrapper.detect(frame))
        if best is None:
            continue
        x1, y1, x2, y2 = best["bbox"]
        shorts.append(min(float(x2) - float(x1), float(y2) - float(y1)))
    cap.release()
    if not shorts:
        return {"frames": 0, "min": 0.0, "median": 0.0, "max": 0.0}
    shorts = sorted(shorts)
    return {
        "frames": len(shorts),
        "min": shorts[0],
        "median": statistics.median(shorts),
        "max": shorts[-1],
    }


def run_gate(stage, config: dict, video_path: Path) -> dict:
    video_id = video_path.stem
    fps = float(config.get("stage_04", {}).get("target_fps", 25))
    n = count_frames(video_path)
    clip = ClipInfo(
        clip_id=f"{video_id}_scene_0000_clip_0000",
        scene_id=f"{video_id}_scene_0000",
        video_id=video_id,
        start_frame=0,
        end_frame=n,
        start_time=0.0,
        end_time=n / fps,
        frame_count=n,
        duration=n / fps,
    )
    logger = PipelineLogger(
        name=f"stage5_smoke_{video_id}",
        log_dir=Path(tempfile.gettempdir()) / "stage5_smoke_logs",
    )
    context = PipelineContext(
        config=config,
        logger=logger,
        model_manager=ModelManager(device="cpu"),
        runtime_info={"video_id": video_id},
    )

    with mock.patch.object(stage, "_resolve_video_path", return_value=video_path):
        try:
            stage.run([clip], context)
            passed, reason = True, None
        except FaceResolutionFilterError as exc:
            passed, reason = False, exc.result.reject_reason

    audit = context.stage_audit.get("stage_05_face_resolution_filter", {})
    return {
        "video": video_path.name,
        "frames": n,
        "passed": passed,
        "reject_reason": reason,
        "qualified_ratio": audit.get("qualified_ratio"),
        "status_counts": audit.get("status_counts"),
    }


def main() -> int:
    loader = ConfigLoader(config_path=ROOT / "configs" / "config.yaml")
    config = loader.load()

    filter_cfg = config.get("stage_05_face_resolution_filter", {})
    print(
        "stage_05_face_resolution_filter: enabled =",
        filter_cfg.get("enabled"),
    )
    if not filter_cfg.get("enabled"):
        print("ABORT: stage_05_face_resolution_filter.enabled is not true.")
        return 2

    # Host inference: host onnxruntime has no CUDA/PPU execution provider, so
    # force CPU (bbox geometry is identical; only speed differs).
    config["models"]["scrfd"]["path"] = str(ROOT / "models" / "SCRFD")
    config["models"]["scrfd"]["device"] = "cpu"
    config["models"]["scrfd"]["execution_provider"] = "cpu"
    config["storage"]["output_dir"] = str(
        Path(tempfile.gettempdir()) / "stage5_smoke_output"
    )

    measure_wrapper = SCRFDWrapper(
        model_path=ROOT / "models" / "SCRFD",
        device="cpu",
        execution_provider="cpu",
        confidence_threshold=0.3,
    )
    measure_wrapper.load()

    stage = Stage5FaceDetection()
    rows = []
    failures = 0

    for filename, (label, expected) in FIXTURES.items():
        video_path = ROOT / "test_data" / "stage5_tset" / filename
        if not video_path.exists():
            print(f"[{label}] MISSING {video_path}")
            failures += 1
            continue

        short = measure_short_sides(video_path, measure_wrapper)
        result = run_gate(stage, config, video_path)
        result["label"] = label
        result["expected"] = expected
        result["short_side"] = short

        ok = result["passed"] == expected
        if not ok:
            failures += 1

        sc = result["status_counts"] or {}
        print(
            f"[{label}] gate={'PASS' if result['passed'] else 'REJECT'} "
            f"expect={'PASS' if expected else 'REJECT'} "
            f"{'OK ' if ok else 'MISMATCH '} "
            f"frames={result['frames']} "
            f"(face_frames={short['frames']}, short_side "
            f"min={short['min']:.0f} median={short['median']:.0f} "
            f"max={short['max']:.0f}) "
            f"ratio={result['qualified_ratio']} "
            f"counts={sc} reason={result['reject_reason']}"
        )
        rows.append(result)

    print()
    matched = sum(1 for r in rows if r["passed"] == r["expected"])
    print(
        f"SMOKE RESULT: {matched}/{len(rows)} matched expectations "
        f"(failures={failures})"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())