#!/usr/bin/env python3
"""Stage12 SPECTRE-pose-replacement mesh-diff — FLAME forward comparison.

Re-extracts Stage12 params with the FIXED ([0,1]) wrappers, then runs the
inferno ``DecaFLAME.FLAME`` forward model twice per frame:

  Variant A (current Stage12 output): pose = [EMICA global, fused jaw]
  Variant B (proposed replacement):   pose = [SPECTRE global, SPECTRE jaw]

(expression = EMICA expcode, shape = 0, eye/neck = 0 in both — isolates the
POSE swap).  Reports per-vertex displacement stats and renders front-view
mesh projections (A vs B vs overlay) to PNG for the analysis report.

Run inside face_pipeline_lxs + emica_env:
  docker exec face_pipeline_lxs bash -c \\
    'cd /workspace/Face_Pipeline_V2 && PYTHONPATH=/workspace/Face_Pipeline_V2 \\
     /workspace/emica_env/bin/python scripts/stage12_pose_mesh_diff.py \\
       --face_dir <stage11 aligned_faces clip dir> \\
       --emica_ckpt models/EMICA/.../loss_total=3.88806534.ckpt \\
       --spectre_tar third_party/inferno/spectre/spectre/data/deca_model.tar \\
       --n_frames 32 --render_frames 0,8,16,24'
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY = _PROJECT_ROOT / "third_party"
_INFERNO_ROOT = _THIRD_PARTY / "inferno" / "inferno-master"
_SPECTRE_ROOT = _THIRD_PARTY / "inferno" / "spectre"
for _p in (_THIRD_PARTY, _INFERNO_ROOT, _SPECTRE_ROOT, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class FlameCfg:
    """Minimal config namespace for inferno DecaFLAME.FLAME."""

    def __init__(self, flame_model_path, flame_lmk_embedding_path,
                 n_shape=100, n_exp=100):
        self.flame_model_path = str(flame_model_path)
        self.flame_lmk_embedding_path = str(flame_lmk_embedding_path)
        self.n_shape = n_shape
        self.n_exp = n_exp


def build_flame(device):
    from inferno.models.DecaFLAME import FLAME
    cfg = FlameCfg(
        flame_model_path=_PROJECT_ROOT / "models/FLAME/geometry/generic_model.pkl",
        flame_lmk_embedding_path=_SPECTRE_ROOT / "spectre/data/landmark_embedding.npy",
        n_shape=100, n_exp=100,
    )
    flame = FLAME(cfg).to(device).eval()
    faces = flame.faces_tensor.cpu().numpy()  # [F, 3]
    return flame, faces


def render_frame(verts_a, verts_b, faces, out_path, title=""):
    """Front-view (XY) mesh projection: A | B | overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if title:
        fig.suptitle(title, fontsize=13)

    def _xy(v):
        return v[:, 0], v[:, 1]  # front view: X horizontal, Y vertical

    for ax, v, lbl, col in [
        (axes[0], verts_a, "A: EMICA global + fused jaw", "#d62728"),
        (axes[1], verts_b, "B: SPECTRE global + SPECTRE jaw", "#1f77b4"),
    ]:
        x, y = _xy(v)
        ax.triplot(Triangulation(x, y, faces), color=col, linewidth=0.3, alpha=0.9)
        ax.scatter(x, y, s=0.5, color=col, alpha=0.4)
        ax.set_title(lbl, fontsize=10)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    xa, ya = _xy(verts_a); xb, yb = _xy(verts_b)
    ax.triplot(Triangulation(xa, ya, faces), color="#d62728", linewidth=0.3, alpha=0.8)
    ax.triplot(Triangulation(xb, yb, faces), color="#1f77b4", linewidth=0.3, alpha=0.8)
    ax.set_title("overlay (red=A, blue=B)", fontsize=10)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage12 SPECTRE-pose-replacement FLAME mesh-diff")
    ap.add_argument("--face_dir", required=True)
    ap.add_argument("--emica_ckpt", required=True)
    ap.add_argument("--spectre_tar", required=True)
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--jaw_fusion_lambda", type=float, default=0.6)
    ap.add_argument("--render_frames", default="0,8,16,24",
                    help="Comma-separated frame indices to render to PNG.")
    ap.add_argument("--out_dir", default="outputs/stage12_pose_analysis")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[mesh_diff] device = {device}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(args.face_dir, "aligned_*.jpg")))
    if not imgs:
        print(f"[mesh_diff] no aligned_*.jpg in {args.face_dir}"); sys.exit(1)
    imgs = imgs[: args.n_frames]
    print(f"[mesh_diff] {len(imgs)} frames")

    from face_pipeline_v2.models.wrappers.emica import EMICAWrapper
    from face_pipeline_v2.models.wrappers.spectre import SPECTREWrapper

    emica = EMICAWrapper(Path(args.emica_ckpt), device=device)
    spectre = SPECTREWrapper(Path(args.spectre_tar), device=device)
    emica.load(); spectre.load()

    tensors = [emica.preprocess_image(cv2.imread(p), target_size=(224, 224))
               for p in imgs if cv2.imread(p) is not None]
    batch = torch.cat(tensors, dim=0).to(device)
    print(f"[mesh_diff] batch {tuple(batch.shape)} range=[{batch.min():.3f},{batch.max():.3f}]")

    with torch.no_grad():
        em_out = emica.encode(batch)
        sp_out = spectre.encode(batch)
    exp = em_out["expcode"]                      # [B,100]
    em_global = em_out["globalpose"]              # [B,3]
    em_jaw = em_out["jawpose"]                   # [B,3]
    sp_global = sp_out["head_pose"]               # [B,3]
    sp_jaw = sp_out["jaw_pose"]                  # [B,3]
    lam = args.jaw_fusion_lambda
    fused_jaw = (1.0 - lam) * em_jaw + lam * sp_jaw
    B = exp.shape[0]

    flame, faces = build_flame(device)
    zeros_shape = torch.zeros(B, 100, device=device)
    poseA = torch.cat([em_global, fused_jaw], dim=1)   # [B,6]
    poseB = torch.cat([sp_global, sp_jaw], dim=1)      # [B,6]
    with torch.no_grad():
        vertsA, _, _ = flame(zeros_shape, exp, poseA)  # [B,5023,3]
        vertsB, _, _ = flame(zeros_shape, exp, poseB)
    vertsA = vertsA.cpu().numpy(); vertsB = vertsB.cpu().numpy()
    # --- vertex displacement stats (Variant A vs Variant B) --------------------
    disp = np.linalg.norm(vertsA - vertsB, axis=2)        # [B, 5023]
    per_frame_mean = disp.mean(axis=1)                    # [B]
    per_frame_max = disp.max(axis=1)                      # [B]
    overall_mean = float(disp.mean())
    overall_max = float(disp.max())

    print("\n" + "=" * 78)
    print("FLAME mesh-diff (Variant A vs Variant B) - per-vertex L2 displacement")
    print("=" * 78)
    print(f"frames={B}  jaw_fusion_lambda={lam}  (shape=0, exp=EMICA, isolates pose swap)")
    print(f"per-frame mean disp: mean={per_frame_mean.mean()*1000:6.3f} mm  "
          f"min={per_frame_mean.min()*1000:6.3f} mm  max={per_frame_mean.max()*1000:6.3f} mm")
    print(f"per-frame max  disp: mean={per_frame_max.mean()*1000:6.3f} mm  "
          f"max={per_frame_max.max()*1000:6.3f} mm")
    print(f"overall mean disp = {overall_mean*1000:.3f} mm   "
          f"overall max disp = {overall_max*1000:.3f} mm")

    # --- render selected frames -------------------------------------------------
    render_idx = [int(i) for i in args.render_frames.split(",") if i.strip()]
    render_idx = [i for i in render_idx if 0 <= i < B]
    print(f"[mesh_diff] rendering frames {render_idx} -> {out_dir}")
    for fi in render_idx:
        out_png = out_dir / f"mesh_diff_frame{fi:04d}.png"
        render_frame(vertsA[fi], vertsB[fi], faces, out_png,
                     title=f"frame {fi}: FLAME front-view (shape=0, exp=EMICA)")
        print(f"  saved {out_png}")

    # --- pose magnitude comparison (for the report) -----------------------------
    gnA = np.linalg.norm(em_global.cpu().numpy(), axis=1)
    gnB = np.linalg.norm(sp_global.cpu().numpy(), axis=1)
    jnA = np.linalg.norm(fused_jaw.cpu().numpy(), axis=1)
    jnB = np.linalg.norm(sp_jaw.cpu().numpy(), axis=1)
    print("\nPose magnitude (axis-angle L2 norm, rad):")
    print(f"  global: A(EMICA)  mean={gnA.mean():.4f}  |  B(SPECTRE) mean={gnB.mean():.4f}"
          f"  |  ratio B/A={gnB.mean()/max(gnA.mean(),1e-6):.2f}")
    print(f"  jaw:    A(fused)  mean={jnA.mean():.4f}  |  B(SPECTRE) mean={jnB.mean():.4f}"
          f"  |  ratio B/A={jnB.mean()/max(jnA.mean(),1e-6):.2f}")
    print("\nDONE")
    emica.unload()


if __name__ == "__main__":
    main()
