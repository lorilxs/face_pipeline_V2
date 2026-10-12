#!/usr/bin/env python3
"""Stage12 unit verification — post-normalization-fix pose / expression stats.

EXERCISES THE FIXED WRAPPERS (``EMICAWrapper`` + ``SPECTREWrapper``) using each
wrapper's own ``preprocess_image`` — now **float [0,1]** (the DECA / EMICA /
SPECTRE training distribution produced by skimage ``warp()``, NOT [-1,1]).
It reproduces the exact Stage12 dual-model inference path from
``face_pipeline_v2/stages/stage_12_flame/processor.py`` (one preprocess fed to
both encoders, jaw fusion λ) and prints the Phase-2 deliverable:

  | Field Name        | 中文名称     | Type  | Description                    |
  |-------------------|------------|-------|--------------------------------|
  | global_pose_norm  | 头部旋转幅度 | float | axis-angle L2 norm (rad)       |
  | jaw_pose_norm     | 下巴旋转幅度 | float | axis-angle L2 norm (rad, fused)|
  | expression_std    | 表情变化    | float | expression 动态范围             |

Targets:  global_pose 0 ~ 1 rad   |   jaw_pose 0 ~ 0.5 rad.

Run inside the ``face_pipeline_lxs`` container with ``emica_env``:

    docker exec face_pipeline_lxs bash -c \\
      'cd /workspace/Face_Pipeline_V2 && \\
       PYTHONPATH=/workspace/Face_Pipeline_V2 \\
       /workspace/emica_env/bin/python scripts/stage12_unit_verify.py \\
         --face_dir <stage11 aligned_faces clip dir> \\
         --emica_ckpt models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/model-epoch=38-val/loss_total=3.88806534.ckpt \\
         --spectre_tar third_party/inferno/spectre/spectre/data/deca_model.tar \\
         --n_frames 32'
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

# Make third_party/ inferno + spectre + project root importable (mirrors runner.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY = _PROJECT_ROOT / "third_party"
_INFERNO_ROOT = _THIRD_PARTY / "inferno" / "inferno-master"
_SPECTRE_ROOT = _THIRD_PARTY / "inferno" / "spectre"
for _p in (_THIRD_PARTY, _INFERNO_ROOT, _SPECTRE_ROOT, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _norm(a: np.ndarray) -> np.ndarray:
    """Row-wise L2 norm of an [N, D] array -> [N]."""
    return np.linalg.norm(a, axis=1)


def _stats(a: np.ndarray) -> tuple:
    return (float(a.mean()), float(a.std()), float(a.min()), float(a.max()))


def _row(name_en: str, name_cn: str, stats: tuple, target: str,
         result: str = "") -> str:
    mean, std, mn, mx = stats
    return (f"| {name_en:22s} | {name_cn:14s} | {mean:7.4f} | {std:7.4f} | "
            f"{mn:7.4f} | {mx:7.4f} | {target:13s} | {result:4s} |")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage12 post-fix unit verification (pose/expression stats)")
    ap.add_argument("--face_dir", required=True,
                    help="Directory containing aligned_*.jpg from Stage11.")
    ap.add_argument("--emica_ckpt", required=True, help="EMICA checkpoint .ckpt")
    ap.add_argument("--spectre_tar", required=True, help="SPECTRE/DECA deca_model.tar")
    ap.add_argument("--n_frames", type=int, default=32,
                    help="Number of aligned frames to encode (default 32).")
    ap.add_argument("--jaw_fusion_lambda", type=float, default=0.6,
                    help="SPECTRE weight for fused jaw (default 0.6, = Stage12).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[unit_verify] device = {device}")

    imgs = sorted(glob.glob(os.path.join(args.face_dir, "aligned_*.jpg")))
    if not imgs:
        print(f"[unit_verify] no aligned_*.jpg in {args.face_dir}")
        sys.exit(1)
    imgs = imgs[: args.n_frames]
    print(f"[unit_verify] using {len(imgs)} aligned faces from\n  {args.face_dir}")

    # --- load the FIXED wrappers -------------------------------------------------
    from face_pipeline_v2.models.wrappers.emica import EMICAWrapper
    from face_pipeline_v2.models.wrappers.spectre import SPECTREWrapper

    emica = EMICAWrapper(Path(args.emica_ckpt), device=device)
    spectre = SPECTREWrapper(Path(args.spectre_tar), device=device)
    emica.load()
    spectre.load()

    # --- preprocess via the FIXED wrapper.preprocess_image (float [0,1]) --------
    # NOTE: processor.py:66 uses emica.preprocess_image() and feeds the SAME
    # tensor to both encoders (processor.py:78,80). We mirror that here so the
    # verification exercises the exact pipeline code path, including the fix.
    tensors = []
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        t = emica.preprocess_image(img, target_size=(224, 224))  # [1,3,224,224]
        tensors.append(t)
    if not tensors:
        print("[unit_verify] no readable images"); sys.exit(1)
    batch = torch.cat(tensors, dim=0).to(device)
    lo, hi = float(batch.min()), float(batch.max())
    print(f"[unit_verify] batch {tuple(batch.shape)}  value range = [{lo:.3f}, {hi:.3f}]")
    if lo < -0.01:
        print("[unit_verify] !! range includes negatives -> normalization fix NOT "
              "applied (expected [0,1]); aborting.")
        sys.exit(2)
    # --- dual-model encode (mirrors processor.py:78-80) -------------------------
    with torch.no_grad():
        em_out = emica.encode(batch)   # expcode[B,100] jawpose[B,3] globalpose[B,3]
        sp_out = spectre.encode(batch)  # head_pose[B,3] jaw_pose[B,3]

    exp = em_out["expcode"].cpu().numpy()           # [B,100]
    em_global = em_out["globalpose"].cpu().numpy()  # [B,3]
    em_jaw = em_out["jawpose"].cpu().numpy()        # [B,3]
    sp_global = sp_out["head_pose"].cpu().numpy()   # [B,3]
    sp_jaw = sp_out["jaw_pose"].cpu().numpy()       # [B,3]
    lam = args.jaw_fusion_lambda
    fused_jaw = (1.0 - lam) * em_jaw + lam * sp_jaw  # [B,3]  (processor.py:95-96)

    global_norm = _norm(em_global)
    em_jaw_norm = _norm(em_jaw)
    sp_jaw_norm = _norm(sp_jaw)
    fused_jaw_norm = _norm(fused_jaw)
    exp_norm = _norm(exp)                            # per-frame expression magnitude
    exp_temporal_per_coeff = exp.std(axis=0)         # [100] temporal std / coeff

    g_ok = global_norm.mean() <= 1.0
    j_ok = fused_jaw_norm.mean() <= 0.5

    print("\n" + "=" * 96)
    print("STAGE12 UNIT VERIFICATION - pose / expression stats (post [0,1] fix)")
    print("=" * 96)
    print(f"frames={len(imgs)}  jaw_fusion_lambda={lam}  norm=tensor/255.0->[0,1]")
    print()
    hdr = (f"| {'Field Name':22s} | {'中文名称':14s} | {'mean':>7s} | {'std':>7s} | "
           f"{'min':>7s} | {'max':>7s} | {'target':13s} | {'res':4s} |")
    sep = ("|" + "-" * 24 + "|" + "-" * 16 + "|" + "-" * 9 + "|" + "-" * 9
           + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 15 + "|" + "-" * 6 + "|")
    print(hdr); print(sep)
    # --- required deliverable rows ---
    print(_row("global_pose_norm", "头部旋转幅度",
               _stats(global_norm), "0~1 rad", "PASS" if g_ok else "FAIL"))
    print(_row("jaw_pose_norm", "下巴旋转幅度",
               _stats(fused_jaw_norm), "0~0.5 rad", "PASS" if j_ok else "FAIL"))
    print(_row("expression_std", "表情变化",
               _stats(exp_temporal_per_coeff), "(info)", ""))
    # --- cross-reference rows ---
    print(sep)
    print(_row("jaw_emica_norm", "下巴(EMICA)",
               _stats(em_jaw_norm), "0~0.5 rad", ""))
    print(_row("jaw_spectre_norm", "下巴(SPECTRE)",
               _stats(sp_jaw_norm), "0~0.5 rad", ""))
    print(_row("head_spectre_norm", "头旋转(SPECTRE)",
               _stats(_norm(sp_global)), "(cross-ref)", ""))
    print(_row("expression_norm", "表情幅度(L2)",
               _stats(exp_norm), "(info)", ""))
    print()

    print(f"global_pose_norm.mean  = {global_norm.mean():.4f} rad   "
          f"-> {'PASS (<=1.0)' if g_ok else 'FAIL (>1.0)'}")
    print(f"jaw_pose_norm.mean     = {fused_jaw_norm.mean():.4f} rad   "
          f"-> {'PASS (<=0.5)' if j_ok else 'FAIL (>0.5)'}  (fused: "
          f"{1 - lam:.1f}*EMICA + {lam:.1f}*SPECTRE)")
    print(f"expression_std         = {exp_temporal_per_coeff.mean():.4f}  "
          f"(mean per-coeff temporal std across {len(imgs)} frames)")
    print()
    print("VERDICT:",
          "PASS - normalization fix verified (global<1, jaw<0.5 rad)."
          if (g_ok and j_ok) else
          "FAIL - pose still out of range; re-check normalization / inputs.")
    emica.unload()



if __name__ == "__main__":
    main()
