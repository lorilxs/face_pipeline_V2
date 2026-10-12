"""Quantify pixel differences between regenerated default aligned faces and
the previous run's aligned_faces (same clip, same template)."""
import re
from pathlib import Path

import cv2
import numpy as np

RUN = Path("outputs/stage0_11_batch/CNFACE_00001_BV18Qg96YE1W_000000")
CLIP = "CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000"
REG = Path("outputs/alignment_visual_check/default") / CLIP
EX = RUN / "aligned_faces" / CLIP

_A = re.compile(r"^aligned_(\d+)_.*_(.+)\.jpg$")

maxes, means, fracs, frac_gt1 = [], [], [], []
n = 0
for r in sorted(REG.glob("aligned_*.jpg")):
    m = _A.match(r.name)
    if not m:
        continue
    fidx = int(m.group(1))
    cand = list(EX.glob(f"aligned_{fidx:06d}_*.jpg"))
    if not cand:
        continue
    a = cv2.imread(str(r)).astype(np.int16)
    b = cv2.imread(str(cand[0])).astype(np.int16)
    d = np.abs(a - b)
    n += 1
    maxes.append(float(d.max()))
    means.append(float(d.mean()))
    fracs.append(float((d > 0).mean()))
    frac_gt1.append(float((d > 1).mean()))

print(f"frames compared: {n}")
print(f"max_abs_diff   : min={min(maxes):.1f} max={max(maxes):.1f}")
print(f"mean_abs_diff  : min={min(means):.3f} max={max(means):.3f}")
print(f"frac_px_diff>0 : min={min(fracs):.4f} max={max(fracs):.4f}")
print(f"frac_px_diff>1 : min={min(frac_gt1):.4f} max={max(frac_gt1):.4f}")
