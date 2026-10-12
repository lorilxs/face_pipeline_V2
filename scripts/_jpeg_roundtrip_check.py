"""Verify that the default-vs-existing diff is only JPEG re-encode noise.

Test: decode one existing aligned jpg → re-encode with cv2.imwrite at default
quality → decode again → compare. If the round-trip diff magnitude matches the
regenerate-vs-existing diff, the alignment content is identical and the
difference is purely JPEG compression.
"""
import re
from pathlib import Path

import cv2
import numpy as np

RUN = Path("outputs/stage0_11_batch/CNFACE_00001_BV18Qg96YE1W_000000")
CLIP = "CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000"
EX = RUN / "aligned_faces" / CLIP

files = sorted(EX.glob("aligned_*.jpg"))[:8]

for f in files:
    b = cv2.imread(str(f))                       # existing decode
    tmp = Path("/tmp/_reencode_test.jpg")
    cv2.imwrite(str(tmp), b)                     # re-encode default quality
    c = cv2.imread(str(tmp))                     # decode re-encode
    d = np.abs(b.astype(np.int16) - c.astype(np.int16))
    print(
        f"{f.name}: roundtrip mean={d.mean():.3f} max={d.max():.0f} "
        f"frac>0={(d > 0).mean():.4f} frac>1={(d > 1).mean():.4f}"
    )
