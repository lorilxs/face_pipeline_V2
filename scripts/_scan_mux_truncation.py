#!/usr/bin/env python3
"""Scan all *_faces_only.mp4 under the four output roots and report clips whose
muxed mp4 video frame count != frame_index count (the old -shortest truncation).

Exact expected frames replicate build_verification_face_only_videos logic:
sort frame_index, absolute-map with clip_start_frame, dedup, filter negatives.
Outputs affected sample roots (one per line) for a targeted --force re-build.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOTS = [
    "outputs/Talkvid_Chinese_videos_clips001",
    "outputs/Talkvid_Chinese_videos_clips002",
    "outputs/Talkvid_Chinese_videos_clips003",
    "outputs/chinese_talking_videos_bilibili",
]


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
    # metadata nb_frames first (fast, no decode); fall back to duration*fps
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,duration,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        st = json.loads(r.stdout)["streams"][0]
    except Exception:
        return None
    nf = st.get("nb_frames")
    if nf and nf != "N/A":
        return int(nf)
    dur = st.get("duration")
    rfr = st.get("r_frame_rate")
    if dur and rfr and "/" in rfr:
        num, den = rfr.split("/")
        try:
            return int(round(float(dur) * float(num) / float(den)))
        except Exception:
            return None
    return None


def main() -> int:
    affected_roots = set()
    affected_clips = []
    checked = 0
    for root_s in ROOTS:
        root = Path(root_s)
        if not root.is_dir():
            continue
        for sample in sorted(root.glob("video_*")):
            vfo = sample / "verification_face_only"
            if not vfo.is_dir():
                continue
            for mp in sorted(vfo.glob("*_faces_only.mp4")):
                clip = mp.stem[: -len("_faces_only")]
                exp = expected_frames(sample, clip)
                if exp is None:
                    continue
                got = mp4_frames(mp)
                checked += 1
                if got is not None and got != exp:
                    affected_clips.append((str(mp), exp, got))
                    affected_roots.add(str(sample))
    print(f"checked={checked}  affected_clips={len(affected_clips)}  "
          f"affected_roots={len(affected_roots)}")
    for mp, exp, got in affected_clips[:40]:
        print(f"  AFFECT {Path(mp).name}: expected={exp} got={got}")
    with open("logs/affected_roots_for_fix.txt", "w") as fh:
        for r in sorted(affected_roots):
            fh.write(r + "\n")
    with open("logs/affected_clips_for_fix.txt", "w") as fh:
        for mp, exp, got in sorted(affected_clips):
            fh.write(f"{mp}\t{exp}\t{got}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
