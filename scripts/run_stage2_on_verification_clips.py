#!/usr/bin/env python3
"""Run the current Stage 2 scene detection on ``verification_face_only`` clips
and cut each detected scene into its own MP4 for manual review.

This drives the *actual* ``Stage2SceneSegmentation`` stage (reading its
parameters from ``configs/config.yaml``) so the boundaries match production
behaviour exactly, then splits the source clip with FFmpeg at every detected
boundary.

Outputs are written to ``test_data/stage2_test/<stem>_scene_0000.mp4``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from face_pipeline_v2.config.config_loader import ConfigLoader  # noqa: E402
from face_pipeline_v2.data.models import (  # noqa: E402
    SceneInfo,
    StandardizedVideoInfo,
)
from face_pipeline_v2.pipeline.context import PipelineContext  # noqa: E402
from face_pipeline_v2.stages.stage_02_scene_detection.stage import (  # noqa: E402
    Stage2SceneSegmentation,
)
from face_pipeline_v2.utils.logger import PipelineLogger  # noqa: E402

_DEFAULT_INPUTS = [
    PROJECT_ROOT
    / "outputs/Talkvid_Chinese_videos_clips001/video_4AVKdHSOLdQ_3_175.98_194.56_open_domain"
    / "verification_face_only/4AVKdHSOLdQ_3_175.98_194.56_scene_0000_clip_0000_faces_only.mp4",
    PROJECT_ROOT
    / "outputs/Talkvid_Chinese_videos_clips001/video_59QlHqVOCKw_11_471.02_482.27_open_domain"
    / "verification_face_only/59QlHqVOCKw_11_471.02_482.27_scene_0000_clip_0000_faces_only.mp4",
]

_OUT_DIR = PROJECT_ROOT / "test_data" / "stage2_test"


def _probe_stream(path: Path) -> dict:
    """Return basic stream metadata (fps, resolution, codec, start_time)."""
    from face_pipeline_v2.utils.exceptions import VideoReadError

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=r_frame_rate,width,height,codec_name:format=start_time,duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoReadError(
            f"ffprobe failed for '{path}': {result.stderr.strip()[:300]}"
        )
    payload = json.loads(result.stdout)
    stream = payload.get("streams", [{}])[0]
    fmt = payload.get("format", {})
    num, den = (stream.get("r_frame_rate") or "25/1").split("/")
    fps = float(num) / float(den) if float(den) else 25.0
    return {
        "fps": fps,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "codec": stream.get("codec_name", "h264"),
        "start_time": float(fmt.get("start_time") or 0.0),
        "duration": float(fmt.get("duration") or 0.0),
    }


def _cut_scene(src: Path, out_dir: Path, scene: SceneInfo, stem: str, index: int) -> Path:
    """Cut one SceneInfo range from ``src`` into its own MP4 (accurate re-encode)."""
    out = out_dir / f"{stem}_scene_{index:04d}.mp4"
    duration = scene.end_time - scene.start_time
    start = scene.start_time
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.6f}",
        "-i", str(src),
        "-t", f"{duration:.6f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-an",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for '{src}' scene {scene.scene_id}: "
            f"{result.stderr.strip()[-400:]}"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "videos",
        nargs="*",
        default=None,
        help="Verification face-only videos to process (defaults to the two configured clips).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_OUT_DIR,
        help="Directory to write scene clips into.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
        help="Path to config.yaml.",
    )
    args = parser.parse_args(argv)

    inputs = [Path(v) for v in args.videos] if args.videos else list(_DEFAULT_INPUTS)

    config = ConfigLoader(args.config).load()
    logger = PipelineLogger(
        name="stage2_verify", log_dir=None, level=logging.DEBUG
    )
    context = PipelineContext(config=config, logger=logger, model_manager=None)
    stage = Stage2SceneSegmentation()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for video_path in inputs:
        if not video_path.exists():
            logger.error(f"Input video not found: {video_path}")
            return 1

        meta = _probe_stream(video_path)
        logger.info(
            "\n" + "=" * 70 + "\n"
            f"Video : {video_path.name}\n"
            f"        {meta['width']}x{meta['height']} @ {meta['fps']:.3f} fps, "
            f"{meta['duration']:.3f}s, codec={meta['codec']}, "
            f"start_time={meta['start_time']:.6f}\n"
            + "=" * 70
        )

        std_info = StandardizedVideoInfo(
            video_path=video_path,
            audio_path=Path(""),
            target_fps=int(round(meta["fps"])),
            resolution=(meta["width"], meta["height"]),
            codec=meta["codec"],
        )

        scenes: list[SceneInfo] = stage.run(input_data=std_info, context=context)

        logger.info(f"Detected {len(scenes)} scene(s):")
        for i, sc in enumerate(scenes):
            logger.info(
                f"  [{i:02d}] {sc.scene_id:45s} "
                f"frames={sc.start_frame:5d}..{sc.end_frame:<5d} "
                f"t={sc.start_time:7.3f}..{sc.end_time:7.3f} "
                f"dur={sc.duration:7.3f}s"
            )

        for i, sc in enumerate(scenes):
            try:
                out_path = _cut_scene(video_path, args.out_dir, sc, video_path.stem, i)
                logger.info(f"  -> wrote {out_path.name}")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"  -> failed to cut scene {i}: {exc}")
                return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())