#!/usr/bin/env python3
"""Stage12 model probe — run EMICA + SPECTRE on real aligned faces inside Docker.

READ-ONLY diagnostic. Loads the same encoders used by Stage12 and dumps:
  - EMICA jawpose / globalpose / expcode statistics
  - SPECTRE E_flame raw 236-dim parameter blocks (shape/tex/exp/pose/cam/light)
  - SPECTRE pose slice [200:206] — is it [global, jaw]? radians? axis-angle?
  - Whether full SPECTRE.encode returns a tuple (upstream behaviour)

Usage (inside face_pipeline_lxs, emica_env):
    python scripts/stage12_model_probe.py \
        --face_dir <dir-with-aligned-*.jpg> \
        --emica_ckpt models/EMICA/.../model-epoch=38-val/loss_total=3.88806534.ckpt \
        --spectre_tar third_party/inferno/spectre/spectre/data/deca_model.tar \
        [--n_frames 8]
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.WARNING)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY = _PROJECT_ROOT / "third_party"
_INFERNO_ROOT = _THIRD_PARTY / "inferno" / "inferno-master"
_SPECTRE_ROOT = _THIRD_PARTY / "inferno" / "spectre"
for _p in (_THIRD_PARTY, _INFERNO_ROOT, _SPECTRE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def preprocess(img_bgr, target=(224, 224), norm="neg1_1"):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, target[::-1]).astype(np.float32)
    if norm == "0_1":
        img = img / 255.0
    elif norm == "neg1_1":
        img = img / 127.5 - 1.0
    else:
        raise ValueError(norm)
    t = torch.from_numpy(img).permute(2, 0, 1)
    return t.unsqueeze(0)


def block_stats(a, name, start, end):
    """a: [B, 236] raw SPECTRE params; print stats for a block slice."""
    b = a[:, start:end]
    mag = np.linalg.norm(b, axis=1)
    print(f"  [{name:10s}] idx {start:3d}:{end:<3d} "
          f"mean={b.mean():+.4f} std={b.std():.4f} "
          f"min={b.min():+.4f} max={b.max():+.4f} "
          f"norm_mean={mag.mean():.4f} norm_max={mag.max():.4f}")
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face_dir", required=True)
    ap.add_argument("--emica_ckpt", required=True)
    ap.add_argument("--spectre_tar", required=True)
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--norm", default="neg1_1", choices=["neg1_1", "0_1"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    imgs = sorted(glob.glob(os.path.join(args.face_dir, "aligned_*.jpg")))
    if not imgs:
        print("no aligned images found in", args.face_dir)
        sys.exit(1)
    imgs = imgs[: args.n_frames]
    print(f"loading {len(imgs)} images from {args.face_dir}")

    tensors = []
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        tensors.append(preprocess(img, norm=args.norm))
    if not tensors:
        print("no valid images"); sys.exit(1)
    batch = torch.cat(tensors, dim=0).to(device)
    print(f"batch shape = {tuple(batch.shape)}")

    # ------------------------------------------------------------------ EMICA
    print("\n" + "=" * 70)
    print("EMICA (minimal DecaEncoder + ExpressionEncoder)")
    from face_pipeline_v2.models.wrappers.emica import EMICAWrapper

    emica = EMICAWrapper(Path(args.emica_ckpt), device=device)
    emica.load()
    with torch.no_grad():
        out = emica.encode(batch)
    for k in ("jawpose", "globalpose", "expcode"):
        a = out[k].cpu().numpy()
        mag = np.linalg.norm(a, axis=1)
        print(f"  emica.{k:11s} shape={a.shape} "
              f"mean={a.mean():+.4f} std={a.std():.4f} "
              f"min={a.min():+.4f} max={a.max():+.4f} "
              f"norm_mean={mag.mean():.4f} ({np.degrees(mag.mean()):.1f} deg)")
        if k in ("jawpose", "globalpose"):
            print(f"     per-axis mean={a.mean(0).round(4).tolist()} "
                  f"std={a.std(0).round(4).tolist()}")
    emica.unload()

    # ---------------------------------------------------------------- SPECTRE
    print("\n" + "=" * 70)
    print("SPECTRE (ResnetEncoder E_flame direct + full SPECTRE.encode)")
    from face_pipeline_v2.models.wrappers.spectre import SPECTREWrapper

    sp = SPECTREWrapper(Path(args.spectre_tar), device=device)
    sp.load()
    with torch.no_grad():
        raw = sp._model(batch).cpu().numpy()   # [B, 236]
    print(f"  E_flame raw shape = {raw.shape}")
    print("  per-block stats of raw 236-dim output:")
    block_stats(raw, "shape", 0, 100)
    block_stats(raw, "tex", 100, 150)
    block_stats(raw, "exp", 150, 200)
    pose = block_stats(raw, "pose", 200, 206)
    block_stats(raw, "cam", 206, 209)
    block_stats(raw, "light", 209, 236)
    print("  pose[0:3] (global) mean =", pose[:, :3].mean(0).round(4).tolist())
    print("  pose[3:6] (jaw)    mean =", pose[:, 3:].mean(0).round(4).tolist())
    print("  pose[3:6] (jaw) norm_mean =", np.linalg.norm(pose[:, 3:], axis=1).mean().round(4))



    # full SPECTRE.encode — what does it return?
    print("\n" + "=" * 70)
    print("full SPECTRE.encode return-type check")
    try:
        from spectre.src.spectre import SPECTRE
        from spectre.config import get_cfg_defaults
    except Exception as exc:
        print(f"  [skip] cannot import full SPECTRE: {type(exc).__name__}: {exc}")
        print("\nPROBE DONE")
        return

    cfg = get_cfg_defaults()
    cfg.pretrained_modelpath = args.spectre_tar
    cfg.model.topology_path = str(_SPECTRE_ROOT / "spectre/data/head_template.obj")
    cfg.model.flame_model_path = str(_SPECTRE_ROOT / "spectre/data/FLAME2020/generic_model.pkl")
    cfg.model.flame_lmk_embedding_path = str(_SPECTRE_ROOT / "spectre/data/landmark_embedding.npy")
    cfg.model.face_mask_path = str(_SPECTRE_ROOT / "spectre/data/uv_face_mask.png")
    cfg.model.face_eye_mask_path = str(_SPECTRE_ROOT / "spectre/data/uv_face_eye_mask.png")
    cfg.model.mean_tex_path = str(_SPECTRE_ROOT / "spectre/data/mean_texture.jpg")
    cfg.model.fixed_displacement_path = str(_SPECTRE_ROOT / "spectre/data/fixed_displacement_256.npy")
    cfg.model.dense_template_path = str(_SPECTRE_ROOT / "spectre/data/texture_data_256.npy")
    model = SPECTRE(config=cfg, device=device)
    model.eval()
    ret = model.encode(batch)   # tensor, NOT dict — mirror upstream call
    print("  encode(tensor) returns type:", type(ret).__name__)
    if isinstance(ret, tuple):
        codedict, deca_exp, deca_jaw = ret
        print("  -> tuple (codedict, deca_exp, deca_jaw)")
        print("  codedict keys:", list(codedict.keys()))
        p = codedict["pose"].cpu().numpy()
        print("  codedict['pose'] shape", p.shape)
        print("  codedict['pose'][:,:3] (global) mean =", p[:, :3].mean(0).round(4).tolist())
        print("  codedict['pose'][:,3:] (jaw)    mean =", p[:, 3:].mean(0).round(4).tolist())
        print("  codedict['pose'][:,3:] (jaw) norm_mean =",
              np.linalg.norm(p[:, 3:], axis=1).mean().round(4))
        print("  E_expression-refined jaw norm_mean =",
              np.linalg.norm(deca_jaw.cpu().numpy(), axis=1).mean().round(4))
    else:
        print("  keys:", list(ret.keys()))
        p = ret["pose"].cpu().numpy()
        print("  pose[:,3:] (jaw) norm_mean =", np.linalg.norm(p[:, 3:], axis=1).mean().round(4))

    # Now try the exact runner call: encode({"image": tensor})
    print("\n" + "=" * 70)
    print("mirror runner.py spectre_encode: model.encode({'image': tensor})")
    try:
        ret2 = model.encode({"image": batch})
        print("  -> returned type:", type(ret2).__name__)
    except Exception as exc:
        print(f"  -> RAISED {type(exc).__name__}: {exc}")

    print("\nPROBE DONE")


if __name__ == "__main__":
    main()
