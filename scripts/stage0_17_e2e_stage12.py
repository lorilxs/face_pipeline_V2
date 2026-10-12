#!/usr/bin/env python3
"""Stage12 in emica_ppu_env driver — official pipeline code path (Environment B).

Runs EMICA + SPECTRE on a clip's aligned faces using the pipeline's official
Stage12 building blocks, inside the emica_ppu_env virtualenv:

  - ModelManager.load_model("emica"/"spectre") -> EMICAWrapper/SPECTREWrapper
  - extract_flame_parameters()   (official Stage12 processor)
  - runner.save_outputs()        (official Stage12 output writer)
    -> <output_dir>/<clip_id>/flame_features.npz + flame_metadata.json

This matches Stage12FLAME direct-mode exactly (AISDD Rule 3: models loaded
via ModelManager) while staying inside the emica_ppu_env, per environment policy.

The standalone runner.py CLI is not used because its full-inferno path pulls
in the MICA encoder which depends on modules absent from emica_ppu_env (loguru
etc.); the wrapper path above is the verified-working official route.

Usage (inside the face_pipeline_lxs container, emica_ppu_env activated):
    python scripts/stage0_17_e2e_stage12.py \\
        --input_metadata <clip>_aligned_faces.json \\
        --output_dir outputs/stage0_17_e2e/stage12_flame \\
        --clip_id CLIP_ID \\
        [--num_frames 200] [--batch_size 8] [--jaw_fusion_lambda 0.6]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# inferno third_party (needed by the wrappers' internals)
for _p in (PROJECT_ROOT / "third_party",
           PROJECT_ROOT / "third_party/inferno/inferno-master",
           PROJECT_ROOT / "third_party/inferno/spectre"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from face_pipeline_v2.data.models import AlignedFaceInfo, FLAMEFeature  # noqa: E402
from face_pipeline_v2.models.model_manager import ModelManager  # noqa: E402
from face_pipeline_v2.stages.stage_12_flame import runner as runner_mod  # noqa: E402
from face_pipeline_v2.stages.stage_12_flame.processor import (  # noqa: E402
    extract_flame_parameters,
)

DEFAULT_EMICA_CKPT = ("models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/"
                      "model-epoch=38-val/loss_total=3.88806534.ckpt")
DEFAULT_SPECTRE_CKPT = "third_party/inferno/spectre/spectre/data/deca_model.tar"


def _load_aligned_faces(meta_path: Path) -> List[AlignedFaceInfo]:
    records = json.loads(meta_path.read_text(encoding="utf-8"))
    faces: List[AlignedFaceInfo] = []
    for rec in records:
        faces.append(AlignedFaceInfo(
            clip_id=rec["clip_id"],
            frame_index=rec["frame_index"],
            face_id=rec.get("face_id", "face_0"),
            timestamp=rec["timestamp"],
            aligned_face_path=Path(rec["aligned_face_path"]),
            affine_matrix=tuple(rec.get("affine_matrix", (1., 0., 0., 0., 1., 0.))),
            original_image_size=tuple(rec.get("original_image_size", (256, 256))),
            aligned_resolution=tuple(rec.get("aligned_resolution", (256, 256))),
        ))
    return faces


def _feature_to_result(f: FLAMEFeature) -> Dict[str, Any]:
    return {
        "frame_index": f.frame_index,
        "timestamp": f.timestamp,
        "clip_id": f.clip_id,
        "expression_code": list(f.expcode),
        "jaw_pose": list(f.jawpose),
        "head_pose": list(f.globalpose),
        "eye_pose": list(f.eye_pose),
        "spectre_jawpose": list(f.spectre_jawpose) if f.spectre_jawpose else [0.0] * 3,
        "emica_jawpose": list(f.emica_jawpose) if f.emica_jawpose else [0.0] * 3,
        "shapecode": list(f.shapecode),
    }

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_metadata", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--clip_id", required=True)
    ap.add_argument("--num_frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--jaw_fusion_lambda", type=float, default=1.0)
    ap.add_argument("--expression_source", default="emica",
                    help="expression source: 'emica' (default) | 'spectre'. "
                         "Single-source experiments only (mixed forbidden).")
    ap.add_argument("--head_source", default="emica",
                    help="head_pose source: 'emica' (default) | 'spectre'. "
                         "Single-source experiments only (mixed forbidden).")
    ap.add_argument("--jaw_source", default="emica",
                    help="jaw_pose source: 'emica' (default, production "
                         "scheme) | 'spectre'. Single-source experiments only "
                         "(mixed forbidden).")
    ap.add_argument("--emica_checkpoint",
                    default=str(PROJECT_ROOT / DEFAULT_EMICA_CKPT))
    ap.add_argument("--spectre_checkpoint",
                    default=str(PROJECT_ROOT / DEFAULT_SPECTRE_CKPT))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage12 emica_ppu_env] device={device} torch={torch.__version__} "
          f"expression_source={args.expression_source} "
          f"head_source={args.head_source} jaw_source={args.jaw_source}",
          flush=True)

    faces = _load_aligned_faces(Path(args.input_metadata))
    if args.num_frames and args.num_frames > 0:
        faces = faces[:args.num_frames]
    print(f"[Stage12 emica_ppu_env] clip={args.clip_id} frames={len(faces)}", flush=True)
    if not faces:
        print("No faces to process.", flush=True)
        return 1

    t0 = time.monotonic()
    mm = ModelManager(device=device)
    print("[Stage12 emica_ppu_env] loading EMICA ...", flush=True)
    emica = mm.load_model("emica", args.emica_checkpoint)
    print("[Stage12 emica_ppu_env] loading SPECTRE ...", flush=True)
    spectre = mm.load_model("spectre", args.spectre_checkpoint)
    t_models = time.monotonic() - t0
    print(f"[Stage12 emica_ppu_env] models loaded in {t_models:.1f}s", flush=True)

    t1 = time.monotonic()
    features = extract_flame_parameters(
        aligned_faces=faces,
        emica_model=emica,
        spectre_model=spectre,
        jaw_fusion_lambda=args.jaw_fusion_lambda,
        expression_source=args.expression_source,
        head_source=args.head_source,
        jaw_source=args.jaw_source,
        batch_size=args.batch_size,
        device=device,
    )
    t_infer = time.monotonic() - t1
    print(f"[Stage12 emica_ppu_env] inference {len(features)} frames in {t_infer:.1f}s "
          f"({t_infer / max(len(features), 1):.2f}s/frame)", flush=True)

    results = [_feature_to_result(f) for f in features]
    out_dir = Path(args.output_dir)
    runner_mod.save_outputs(
        results=results,
        output_dir=out_dir,
        clip_id=args.clip_id,
        jaw_fusion_lambda=args.jaw_fusion_lambda,
        expression_source=args.expression_source,
        head_source=args.head_source,
        jaw_source=args.jaw_source,
    )
    print(f"[Stage12 emica_ppu_env] saved -> {out_dir / args.clip_id}", flush=True)

    if results:
        exp = np.asarray([r["expression_code"] for r in results])
        pose = np.hstack([np.asarray([r["head_pose"] for r in results]),
                          np.asarray([r["jaw_pose"] for r in results])])
        print(f"[Stage12 emica_ppu_env] expression shape={exp.shape} "
              f"std={exp.std():.4f} mean={exp.mean():.4f}", flush=True)
        print(f"[Stage12 emica_ppu_env] pose shape={pose.shape} std={pose.std():.4f}",
              flush=True)
    print("[Stage12 emica_ppu_env] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

