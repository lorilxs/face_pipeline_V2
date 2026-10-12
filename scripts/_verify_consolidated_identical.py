#!/usr/bin/env python3
"""Verify consolidated *_new_pipeline/videos copies are byte-identical to the
rebuilt verification_face_only/*_faces_only.mp4 sources for all rebuilt roots."""
import os
import hashlib
import sys

PROJ = "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2"
ROOT_LIST = os.path.join(PROJ, "logs", "rebuild_roots_for_fix.txt")
OUT_LOG = os.path.join(PROJ, "logs", "verify_consolidated_identical.log")


def md5(path: str, block: int = 1 << 20) -> str:
    hh = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            hh.update(chunk)
    return hh.hexdigest()


def main() -> int:
    with open(ROOT_LIST) as fh:
        roots = [l.strip() for l in fh if l.strip()]
    missing = mism = ok = 0
    lines = []
    for root in roots:
        out_root, batch = root.split("/")[:2]
        vdir = os.path.join(PROJ, out_root, batch, batch + "_new_pipeline", "videos")
        src_dir = os.path.join(PROJ, root, "verification_face_only")
        if not os.path.isdir(src_dir):
            continue
        for f in sorted(os.listdir(src_dir)):
            if not f.endswith("_faces_only.mp4") or f.endswith("_original.mp4"):
                continue
            s = os.path.join(src_dir, f)
            d = os.path.join(vdir, f)
            if not os.path.exists(d):
                missing += 1
                lines.append(f"MISSING {d}")
                continue
            if os.path.getsize(s) != os.path.getsize(d) or md5(s) != md5(d):
                mism += 1
                lines.append(f"MISMATCH {s}  <->  {d}")
                continue
            ok += 1
    summary = f"verified identical: {ok}  missing: {missing}  mismatched: {mism}"
    with open(OUT_LOG, "w") as fh:
        fh.write("\n".join(lines + [summary]) + "\n")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
