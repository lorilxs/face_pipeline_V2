#!/usr/bin/env python3
"""CLI entry point for the Stage5/Stage10 temporal stability audit.

End-to-end pipeline (all read-only w.r.t. the pipeline's outputs):

    1. extract_stage5  -- re-run SCRFD on the clip video -> stage5_frames.json
    2. extract_stage10 -- run InsightFace 2d106det on existing crops
                          -> stage10_landmarks.json
    3. analyze         -- temporal stability metrics -> temporal_stability.json
    4. render overlay videos for both stages
    5. generate the Markdown report in docs/

Usage (from the Face_Pipeline_V2 project root):

    PYTHONPATH=. python -m scripts.stage5_stage10_audit.run_audit \\
        --clip-id CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001

For a quick data-only smoke test (no videos, fewer frames):

    PYTHONPATH=. python -m scripts.stage5_stage10_audit.run_audit \\
        --max-frames 40 --skip-videos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``import face_pipeline_v2`` regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from . import analysis, report
from .data_io import load_stage5, load_stage10
from .extractor import (
    crop_frame_mapping,
    extract_stage10,
    extract_stage5,
    locate_clip_video,
    locate_crops_dir,
)
from .visualize_stage5 import render_stage5_video
from .visualize_stage10 import render_stage10_video


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clip-id",
        default="CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001",
        help="Target clip id.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=_PROJECT_ROOT / "outputs/stage0_17_e2e/stage0_11",
        help="Pipeline run directory (with manifest.json + clips/ + faces/).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJECT_ROOT / "docs/stage5_stage10_audit",
        help="Directory for data JSONs + overlay videos.",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=_PROJECT_ROOT / "docs/Stage5_Stage10_Temporal_Stability_Report.md",
        help="Output path for the Markdown report.",
    )
    p.add_argument(
        "--scrfd-model-dir",
        type=Path,
        default=_PROJECT_ROOT / "models/SCRFD",
        help="Directory containing scrfd_10g_bnkps.onnx.",
    )
    p.add_argument(
        "--landmark-model",
        type=Path,
        default=_PROJECT_ROOT / "models/insightface/2d106det.onnx",
        help="Path to the InsightFace 2d106det.onnx model.",
    )
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda'.")
    p.add_argument("--expand-ratio", type=float, default=2.0,
                   help="Stage5 bbox expansion factor (must match config).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit processing to the first N frames (smoke test).")
    p.add_argument("--skip-extract", action="store_true",
                   help="Reuse existing stage5/stage10 JSONs instead of re-running models.")
    p.add_argument("--skip-videos", action="store_true",
                   help="Skip the (slower) overlay-video rendering.")
    p.add_argument("--sigma", type=float, default=3.0,
                   help="Outlier threshold in std deviations.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run_dir = args.run_dir
    clip_id = args.clip_id
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stage5_json = out_dir / "stage5_frames.json"
    stage10_json = out_dir / "stage10_landmarks.json"
    stab_json = out_dir / "temporal_stability.json"
    video5 = out_dir / f"overlay_stage5_{clip_id}.mp4"
    video10 = out_dir / f"overlay_stage10_{clip_id}.mp4"

    video_path = locate_clip_video(run_dir, clip_id)
    crops_dir = locate_crops_dir(run_dir, clip_id)
    video_id = crops_dir.parent.parent.name  # faces/{video_id}/{clip_id}/crops
    _, mapping = crop_frame_mapping(crops_dir)
    print(f"[audit] clip video: {video_path}")
    print(f"[audit] crops dir : {crops_dir} ({len(mapping)} crops)")

    # 1. Stage 5 -----------------------------------------------------------
    if args.skip_extract and stage5_json.exists():
        print(f"[audit] reusing {stage5_json}")
    else:
        extract_stage5(
            video_path=video_path,
            out_path=stage5_json,
            scrfd_model_dir=args.scrfd_model_dir,
            expand_ratio=args.expand_ratio,
            device=args.device,
            max_frames=args.max_frames,
            clip_id=clip_id,
            video_id=video_id,
        )
    stage5 = load_stage5(stage5_json)

    # 2. Stage 10 ----------------------------------------------------------
    if args.skip_extract and stage10_json.exists():
        print(f"[audit] reusing {stage10_json}")
    else:
        extract_stage10(
            crops_dir=crops_dir,
            stage5_json_path=stage5_json,
            out_path=stage10_json,
            landmark_model_path=args.landmark_model,
            device=args.device,
            video_path=video_path,
        )
    stage10 = load_stage10(stage10_json)

    # 3. Analysis ----------------------------------------------------------
    summary = analysis.analyze(
        stage5, stage10, outlier_sigma=args.sigma
    )
    analysis.save_json(summary, stab_json)

    # 4. Overlay videos ----------------------------------------------------
    outlier_frames = set(summary["outlier_frames"])
    if not args.skip_videos:
        render_stage5_video(
            stage5, video_path, video5, fps=stage5.source.get("fps"),
            outlier_frames=outlier_frames, max_frames=args.max_frames,
        )
        render_stage10_video(
            stage10, video_path, video10, fps=stage10.source.get("fps"),
            outlier_frames=outlier_frames, max_frames=args.max_frames,
        )
    else:
        video5 = video10 = Path("(skipped)")

    # 5. Report ------------------------------------------------------------
    report.generate_report(
        summary=summary,
        report_path=args.report,
        video_path=video_path,
        stage5_video=video5,
        stage10_video=video10,
        run_dir=run_dir,
        crops_dir=crops_dir,
        crop_count=len(mapping),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

