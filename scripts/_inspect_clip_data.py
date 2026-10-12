"""Inspect existing Stage 5 crops and Stage 11 aligned outputs for one clip."""
import re
from pathlib import Path
import cv2

RUN = Path("outputs/stage0_11_batch/CNFACE_00001_BV18Qg96YE1W_000000")

aligned_root = RUN / "aligned_faces"
crops_root = RUN / "faces" / "CNFACE_00001_BV18Qg96YE1W_000000"

print("=== aligned_faces per clip ===")
for d in sorted(aligned_root.iterdir()):
    if d.is_dir():
        print(f"{d.name}: {len(list(d.glob('*.jpg')))}")

print("\n=== crops per clip ===")
for d in sorted(crops_root.iterdir()):
    crops = d / "crops"
    if crops.is_dir():
        print(f"{d.name}: {len(list(crops.glob('*.jpg')))}")

print("\n=== sample aligned filename ===")
afiles = sorted((aligned_root / "CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000").glob("*.jpg"))
print(afiles[0].name if afiles else "none")

print("\n=== sample crop size (first crop of scene_0002_clip_0000) ===")
cdir = crops_root / "CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000" / "crops"
cfiles = sorted(cdir.glob("*.jpg"))
if cfiles:
    im = cv2.imread(str(cfiles[0]))
    print("crop", cfiles[0].name, "shape", im.shape)

print("\n=== aligned image size ===")
if afiles:
    aim = cv2.imread(str(afiles[0]))
    print("aligned", afiles[0].name, "shape", aim.shape)
