"""Regenerate ONLY the visual artifacts from already-generated aligned images.

Avoids re-running landmark inference / alignment. Also produces a few
old-run-vs-regenerated-default side-by-side frames so the near-identity can
be eyeballed.
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

from scripts.regen_aligned_visual_check import (  # noqa: E402
    _make_contact_sheet,
    _make_comparison,
)

CLIP = "CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000"
RUN = Path("outputs/stage0_11_batch/CNFACE_00001_BV18Qg96YE1W_000000")
OUT = Path("outputs/alignment_visual_check")
EX = RUN / "aligned_faces" / CLIP

_A = re.compile(r"^aligned_(\d+)_.*_(.+)\.jpg$")

variants_out = {
    "default": OUT / "default" / CLIP,
    "sf_0.85": OUT / "sf_0.85" / CLIP,
    "sf_1.15": OUT / "sf_1.15" / CLIP,
}
eyes = {
    "default": np.array([[87.53, 118.16], [168.07, 117.72]]),
    "sf_0.85": np.array([[93.787, 118.0], [162.212, 118.0]]),
    "sf_1.15": np.array([[81.713, 118.0], [174.288, 118.0]]),
}

for label, d in variants_out.items():
    _make_contact_sheet(d, OUT / f"montage_{label}.png")

sample_frames = tuple(
    int(p.name[8:14]) for p in sorted(variants_out["default"].glob("aligned_*.jpg"))[::25][:4]
)
_make_comparison(variants_out, OUT, sample_frames, eyes)

# --- old-vs-new default comparison (with pixel-diff heatmap) ---
for fidx in sample_frames:
    reg = sorted(variants_out["default"].glob(f"aligned_{fidx:06d}_*.jpg"))
    old = sorted(EX.glob(f"aligned_{fidx:06d}_*.jpg"))
    if not reg or not old:
        continue
    a = cv2.imread(str(reg[0]))
    b = cv2.imread(str(old[0]))
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    heat = np.clip(diff * 8, 0, 255).astype(np.uint8)
    W = a.shape[1] * 3 + 20
    H = a.shape[0] + 30
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[30:30 + a.shape[0], 0:a.shape[1]] = a
    canvas[30:30 + b.shape[0], a.shape[1] + 10:2 * a.shape[1] + 10] = b
    canvas[30:30 + heat.shape[0], 2 * (a.shape[1] + 10):3 * a.shape[1] + 20] = heat
    cv2.putText(canvas, "NEW default", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, "OLD run", (a.shape[1] + 14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, "diff x8", (2 * (a.shape[1] + 10) + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(OUT / f"old_vs_new_frame_{fidx:06d}.png"), canvas)
    print(f"old_vs_new_frame_{fidx:06d}.png  maxdiff={diff.max()} meandiff={diff.mean():.3f}")
