#!/usr/bin/env python3
"""Check douyin *_faces_only.mp4 frame counts vs frame_index (same logic as
_scan_mux_truncation.py) to confirm none were built by the old -shortest code."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("outputs/douyin_videos_hp_0916")


def resolve_npz(sample_root: Path, clip_id: str) -> Path:
    cands = [
        sample_root / "stage0_11" / "stage17_fusion" / clip_id / "training_sample.npz",
        sample_root / "stage17_multimodal_export" / "timestamp" / f"{clip_id}.npz",
    ]
    return next((c for c in cands if c.exists()), None)


def expected_frames(sample_root: Path, clip_id: str):
    av = json.loads(
        (sample_root / "stage0_11" / "stage_inputs" / f"{clip_id}_av_alignment.json").read_text()
    )
    fps = float(av.get("fps") or 25.0)
    clip_start_frame = int(round(float(av.get("clip_start") or 0.0) * fps))
    npz = resolve_npz(sample_root, clip_id)
    if npz is None:
        return None
    fi = np.sort(np.load(npz, allow_pickle=False)["frame_index"].astype(np.int64))
    if fi.size == 0:
        return 0
    absolute = fi if int(fi.min()) >= clip_start_frame else fi + clip_start_frame
    return len({int(k) for k in absolute if int(k) >= 0})


def mp4_frames(path: Path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        nf = json.loads(r.stdout)["streams"][0].get("nb_frames")
    except Exception:
        return None
    return int(nf) if nf and nf != "N/A" else None


def main() -> int:
    checked = affected = noref = 0
    for sample in sorted(ROOT.glob("video_*")):
        vfo = sample / "verification_face_only"
        if not vfo.is_dir():
            continue
        for mp in sorted(vfo.glob("*_faces_only.mp4")):
            clip = mp.stem[: -len("_faces_only")]
            exp = expected_frames(sample, clip)
            if exp is None:
                noref += 1
                print(f"  NOREF {mp.name}")
                continue
            got = mp4_frames(mp)
            checked += 1
            if got is not None and got != exp:
                affected += 1
                print(f"  AFFECT {mp.name}: expected={exp} got={got}")
    print(f"checked={checked} affected={affected} noref={noref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
