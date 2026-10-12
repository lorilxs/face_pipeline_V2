#!/usr/bin/env python3
import sys
from pathlib import Path
import cv2
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from face_pipeline_v2.stages.stage_02_scene_detection.scene_detector import SceneDetector
VD = PROJECT_ROOT / "outputs/Talkvid_Chinese_videos_clips001"
V1 = VD/"video_4AVKdHSOLdQ_3_175.98_194.56_open_domain"/"verification_face_only/4AVKdHSOLdQ_3_175.98_194.56_scene_0000_clip_0000_faces_only.mp4"
V2 = VD/"video_59QlHqVOCKw_11_471.02_482.27_open_domain"/"verification_face_only/59QlHqVOCKw_11_471.02_482.27_scene_0000_clip_0000_faces_only.mp4"
cfg = dict(enabled=True, band_top=0.33, band_bottom=0.67, edge_z=4.0, bright_abs_min=0.002, bright_z=3.0, bright_luma=235.0, min_subtitle_frames=5, pad_frames=4)
d = SceneDetector(threshold=6.0, min_scene_len=15, adaptive_factor=2.0)
for name, path in (("V1", V1), ("V2", V2)):
    intervals, noise = d._analyze_video(path, cfg)
    print(f"\n{name}: noise_ceiling={noise!r}")
    print(f"  intervals(frames): {intervals}")
    for s, e in intervals:
        print(f"    -> {s/25:.3f}s .. {e/25:.3f}s")
