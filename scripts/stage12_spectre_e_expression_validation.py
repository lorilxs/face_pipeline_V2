#!/usr/bin/env python3
"""Validate SPECTRE E_expression jaw vs old E_flame raw jaw vs EMICA jaw.

Compares three jaw/head sources on the same aligned faces:
  1. NEW  SPECTRE wrapper with spectre_model.tar -> E_expression jaw (official)
  2. OLD  SPECTRE wrapper with deca_model.tar    -> E_flame raw jaw (previous)
  3. EMICA jaw_pose read from an existing flame_features.npz audit field

Also reports head pose per source and the per-axis jaw magnitude (the FLAME
jaw axis-angle x component drives mouth opening; official E_expression
ReLU-clamps it non-negative).

Usage:
  python scripts/stage12_spectre_e_expression_validation.py \
      --face_dir <aligned_faces clip dir> \
      --npz <path to existing flame_features.npz> \
      --n_frames 200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from face_pipeline_v2.models.wrappers.spectre import SPECTREWrapper  # noqa: E402

DEFAULT_NEW_CKPT = (
    _PROJECT_ROOT
    / "third_party/inferno/spectre/spectre/data/spectre_model.tar"
)
DEFAULT_OLD_CKPT = (
    _PROJECT_ROOT
    / "third_party/inferno/spectre/spectre/data/deca_model.tar"
)


def run_wrapper(ckpt: Path, face_paths):
    """Load wrapper on a checkpoint and encode all faces. Returns numpy dict."""
    import cv2
    import torch

    w = SPECTREWrapper(Path(ckpt), device="cpu")
    w.load()
    tensors = []
    for p in face_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"cannot read {p}")
        tensors.append(w.preprocess_image(img, target_size=(224, 224)))
    batch = torch.cat(tensors, dim=0)
    out = w.encode(batch)
    return {k: v.cpu().numpy() for k, v in out.items()}


def _axis_max(a: np.ndarray):
    return [float(a[:, i].max()) for i in range(a.shape[1])]


def _axis_mean_abs(a: np.ndarray):
    return [float(np.abs(a[:, i]).mean()) for i in range(a.shape[1])]


def stats(name: str, a: np.ndarray):
    d = np.abs(np.diff(a, axis=0))
    return {
        "name": name,
        "shape": list(a.shape),
        "mean_abs": float(np.abs(a).mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "delta_mean": float(d.mean()),
        "delta_max": float(d.max()),
        "per_axis_max": _axis_max(a),
        "per_axis_mean_abs": _axis_mean_abs(a),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--face_dir", required=True, help="aligned faces clip dir")
    ap.add_argument("--npz", default=None, help="existing flame_features.npz")
    ap.add_argument(
        "--npz_field",
        default="emica_jawpose",
        help="npz key holding the EMICA jaw reference "
        "(ablation npz: 'emica_jawpose'; e2e production npz: 'jaw_pose')",
    )
    ap.add_argument("--new_ckpt", default=str(DEFAULT_NEW_CKPT))
    ap.add_argument("--old_ckpt", default=str(DEFAULT_OLD_CKPT))
    ap.add_argument("--n_frames", type=int, default=200)
    args = ap.parse_args()

    face_dir = Path(args.face_dir)
    faces = sorted(face_dir.glob("aligned_*.jpg"))
    if not faces:
        faces = sorted(face_dir.glob("*.jpg"))
    faces = faces[: args.n_frames]
    print(f"faces: {len(faces)} from {face_dir}")
    if not faces:
        raise SystemExit("no aligned faces found")

    new = run_wrapper(Path(args.new_ckpt), faces)
    old = run_wrapper(Path(args.old_ckpt), faces)

    emica_jaw = None
    if args.npz and Path(args.npz).exists():
        f = np.load(args.npz)
        if args.npz_field in f:
            n = min(len(new["jaw_pose"]), f[args.npz_field].shape[0])
            emica_jaw = f[args.npz_field][:n]
        else:
            print(
                f"note: npz field {args.npz_field!r} not found; "
                f"available: {list(f.keys())}"
            )

    rows = [
        stats("jaw  NEW(E_expression)", new["jaw_pose"]),
        stats("jaw  OLD(E_flame raw) ", old["jaw_pose"]),
    ]
    if emica_jaw is not None:
        rows.append(stats("jaw  EMICA            ", emica_jaw))
    rows.append(stats("head NEW(E_flame s.tar)", new["head_pose"]))
    rows.append(stats("head OLD(E_flame d.tar)", old["head_pose"]))

    out = {"n_frames": len(faces), "metrics": rows}
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # Interpretation helper
    jn = new["jaw_pose"]
    print("\n-- mouth-opening check (jaw x axis, rad / deg) --")
    for label, arr in (("NEW_Eexpr", jn), ("OLD_Eflame", old["jaw_pose"])):
        mx = arr[:, 0]
        print(
            f"{label:12s} jaw_x mean={mx.mean():+.4f} "
            f"std={mx.std():.4f} max={mx.max():+.4f} "
            f"(~{np.degrees(mx.max()):.1f} deg)"
        )
    if emica_jaw is not None:
        ex = emica_jaw[:, 0]
        print(
            f"{'EMICA':12s} jaw_x mean={ex.mean():+.4f} "
            f"std={ex.std():.4f} max={ex.max():+.4f} "
            f"(~{np.degrees(ex.max()):.1f} deg)"
        )


if __name__ == "__main__":
    main()
