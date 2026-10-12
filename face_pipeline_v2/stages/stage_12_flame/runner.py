#!/usr/bin/env python3
"""Stage 12 FLAME parameter extraction — standalone runner for Docker subprocess.

This script runs inside the Docker container ``face_pipeline_lxs`` with the
``emica_ppu_env`` activated. It reads aligned face images from Stage 11, runs
EMICA + SPECTRE dual-model inference with jaw_pose = EMICA (final adopted
scheme, no fusion), and writes output to the specified directory.

Usage (inside Docker):
    source /workspace/emica_ppu_env/bin/activate
    python stage12_runner.py \\
        --input_metadata /path/to/input_metadata.json \\
        --output_dir /path/to/outputs/stage12_flame/ \\
        --emica_checkpoint /path/to/emica.ckpt \\
        --spectre_checkpoint /path/to/spectre.tar \\
        --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from face_pipeline_v2.stages.stage_12_flame.processor import (
    _pad_expression,
    validate_sources,
)

# ---------------------------------------------------------------------------
# Path setup: register third_party/ for inferno imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_THIRD_PARTY = _PROJECT_ROOT / "third_party"
_INFERNO_ROOT = _THIRD_PARTY / "inferno" / "inferno-master"
_SPECTRE_ROOT = _THIRD_PARTY / "inferno" / "spectre"

# Repo root on sys.path so face_pipeline_v2.* (wrappers) are importable.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

for _p in (_THIRD_PARTY, _INFERNO_ROOT, _SPECTRE_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

os.environ["LIGHTNING_SUPPRESS_LOGGER"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("stage12_runner")
# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_emica(checkpoint_path: str, device: str) -> Any:
    """Load EMICA encoder via the production EMICAWrapper.

    The full-inferno FaceReconstructionBase route pulls in the MICA encoder
    (loguru, etc.) that is absent from emica_ppu_env and has never been functional;
    the wrapper loads only the DECA + Expression ResNet50 encoders and is the
    verified official route (see scripts/stage0_17_e2e_stage12.py).
    """
    logger.info("Loading EMICA (wrapper) from %s", checkpoint_path)
    from face_pipeline_v2.models.wrappers.emica import EMICAWrapper
    wrapper = EMICAWrapper(Path(checkpoint_path), device=device)
    wrapper.load()
    return wrapper


def _load_spectre(checkpoint_path: str, device: str) -> Any:
    """Load SPECTRE E_flame via the production SPECTREWrapper."""
    logger.info("Loading SPECTRE (wrapper) from %s", checkpoint_path)
    from face_pipeline_v2.models.wrappers.spectre import SPECTREWrapper
    wrapper = SPECTREWrapper(Path(checkpoint_path), device=device)
    wrapper.load()
    return wrapper


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(img_bgr: np.ndarray,
                     target_size: tuple = (224, 224)) -> torch.Tensor:
    """Preprocess BGR image to normalized tensor [1, 3, H, W] in [0, 1] (float; matches DECA/EMICA/SPECTRE training distribution)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size[::-1])
    tensor = torch.from_numpy(
        img_resized.astype(np.float32)).permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0)


# ---------------------------------------------------------------------------
# EMICA / SPECTRE inference helpers
# ---------------------------------------------------------------------------

def emica_encode(model: Any, images: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run EMICA encoder on a batch of images."""
    with torch.no_grad():
        values = model.encode(images)
    return values


def spectre_encode(model: Any, images: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run SPECTRE encoder on a batch of images.

    SPECTRE.encode takes a tensor (not a dict) and returns a 3-tuple
    ``(codedict, deca_exp, deca_jaw)``; we keep only ``codedict`` whose
    ``"pose"`` is ``[global(3), jaw(3)]`` (jaw = pose[3:6]).
    """
    with torch.no_grad():
        ret = model.encode(images)
    codedict = ret[0] if isinstance(ret, tuple) else ret
    return codedict
# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def run_stage12(
    faces: List[Dict[str, Any]],
    emica_model: Any,
    spectre_model: Any,
    jaw_fusion_lambda: float,
    batch_size: int,
    device: str,
    expression_source: str = "emica",
    head_source: str = "emica",
    jaw_source: str = "emica",
) -> List[Dict[str, Any]]:
    """Run EMICA + SPECTRE batch inference, return FLAME feature dicts.

    Per-field source selection (Stage12 EMICA vs SPECTRE ablation):
      - Experiment A (production default): all sources = EMICA
      - Experiment B: all sources = SPECTRE
    Mixed sources are forbidden (validate_sources raises ValueError).
    ``jaw_fusion_lambda`` is deprecated and never influences the result.
    """
    validate_sources(expression_source, head_source, jaw_source)
    results: List[Dict[str, Any]] = []
    n_faces = len(faces)

    for batch_start in range(0, n_faces, batch_size):
        batch_end = min(batch_start + batch_size, n_faces)
        batch_faces = faces[batch_start:batch_end]

        images: List[torch.Tensor] = []
        valid_indices: List[int] = []

        for i, face in enumerate(batch_faces):
            img_path = face["aligned_face_path"]
            if not Path(img_path).exists():
                logger.warning("Image not found: %s", img_path)
                continue
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("Failed to read: %s", img_path)
                continue
            tensor = preprocess_image(img, target_size=(224, 224))
            images.append(tensor)
            valid_indices.append(batch_start + i)

        if not images:
            continue

        batch_tensor = torch.cat(images, dim=0).to(device)

        emica_out = emica_encode(emica_model, batch_tensor)
        spectre_out = spectre_encode(spectre_model, batch_tensor)

        B = batch_tensor.shape[0]
        for i in range(B):
            face_idx = valid_indices[i]
            face = faces[face_idx]

            emica_exp = emica_out["expcode"][i].cpu().numpy()
            emica_jaw = emica_out["jawpose"][i].cpu().numpy()
            emica_global = emica_out["globalpose"][i].cpu().numpy()

            # SPECTRE raw outputs (audit trail + source selection)
            if "expression" in spectre_out:
                spectre_exp = _pad_expression(
                    spectre_out["expression"][i].cpu().numpy(), 100
                )
            else:
                spectre_exp = np.zeros(100, dtype=np.float32)
            if "head_pose" in spectre_out:
                spectre_head = spectre_out["head_pose"][i].cpu().numpy()
            elif "pose" in spectre_out:
                pose = spectre_out["pose"][i].cpu().numpy()
                spectre_head = pose[:3] if pose.shape[-1] >= 6 else pose[0:3]
            else:
                spectre_head = np.zeros(3, dtype=np.float32)
            if "jaw_pose" in spectre_out:
                spectre_jaw = spectre_out["jaw_pose"][i].cpu().numpy()
            elif "pose" in spectre_out:
                pose = spectre_out["pose"][i].cpu().numpy()
                spectre_jaw = pose[3:6] if pose.shape[-1] >= 6 else pose[0:3]
            else:
                spectre_jaw = np.zeros(3, dtype=np.float32)

            # --- Per-field source selection (single-source experiments only) ---
            # Production default (Experiment A): all EMICA.
            expression_code = (
                emica_exp if expression_source == "emica" else spectre_exp
            )
            head_pose = (
                emica_global if head_source == "emica" else spectre_head
            )
            jaw_pose = emica_jaw if jaw_source == "emica" else spectre_jaw

            result = {
                "frame_index": face["frame_index"],
                "timestamp": face["timestamp"],
                "clip_id": face["clip_id"],
                "expression_code": expression_code.tolist(),
                "jaw_pose": jaw_pose.tolist(),
                "head_pose": head_pose.tolist(),
                "eye_pose": [0.0] * 6,
                "spectre_jawpose": spectre_jaw.tolist(),
                "emica_jawpose": emica_jaw.tolist(),  # debug/analysis only
                "shapecode": [0.0] * 300,
            }
            results.append(result)

    return results
# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_outputs(
    results: List[Dict[str, Any]],
    output_dir: Path,
    clip_id: str,
    jaw_fusion_lambda: float,
    expression_source: str = "emica",
    head_source: str = "emica",
    jaw_source: str = "emica",
) -> None:
    """Save FLAME features and metadata.

    Writes:
        output_dir/clip_id/flame_features.npz
        output_dir/clip_id/flame_metadata.json
        output_dir/clip_id/expression.npy, jaw_pose.npy, etc.

    Metadata ``expression_source`` / ``head_source`` / ``jaw_source`` record the
    REAL source used for each field (Experiment A: all emica; Experiment B: all
    spectre).
    """
    if not results:
        logger.warning("No results to save for clip %s", clip_id)
        return

    clip_dir = output_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    T = len(results)
    expression = np.zeros((T, 100), dtype=np.float32)
    jaw_pose = np.zeros((T, 3), dtype=np.float32)
    head_pose = np.zeros((T, 3), dtype=np.float32)
    eye_pose = np.zeros((T, 6), dtype=np.float32)
    spectre_jawpose = np.zeros((T, 3), dtype=np.float32)
    emica_jawpose = np.zeros((T, 3), dtype=np.float32)
    shapecode = np.zeros((T, 300), dtype=np.float32)
    frame_indices = np.zeros(T, dtype=np.int32)
    timestamps = np.zeros(T, dtype=np.float64)

    for i, r in enumerate(results):
        expression[i] = r["expression_code"]
        jaw_pose[i] = r["jaw_pose"]
        head_pose[i] = r["head_pose"]
        eye_pose[i] = r["eye_pose"]
        spectre_jawpose[i] = r["spectre_jawpose"]
        emica_jawpose[i] = r["emica_jawpose"]
        shapecode[i] = r["shapecode"]
        frame_indices[i] = r["frame_index"]
        timestamps[i] = r["timestamp"]

    flame_data = {
        "expression_code": expression,
        "jaw_pose": jaw_pose,
        "head_pose": head_pose,
        "eye_pose": eye_pose,
        "spectre_jawpose": spectre_jawpose,
        "emica_jawpose": emica_jawpose,
        "shapecode": shapecode,
        "frame_index": frame_indices,
        "timestamp": timestamps,
    }
    np.savez(str(clip_dir / "flame_features.npz"), **flame_data)
    logger.info("Saved flame_features.npz to %s", clip_dir)

    metadata = {
        "clip_id": clip_id,
        "num_frames": T,
        "jaw_fusion_lambda": jaw_fusion_lambda,   # DEPRECATED — no longer used
        "jaw_source": jaw_source,                 # real jaw_pose source
        "head_source": head_source,               # real head_pose source
        "expression_source": expression_source,   # real expression source
        "backend": "EMICA+SPECTRE",
        "expression_dim": 100,
        "jaw_pose_dim": 3,
        "head_pose_dim": 3,
        "eye_pose_dim": 6,
        "spectre_jawpose_dim": 3,
        "emica_jawpose_dim": 3,
        "shapecode_dim": 300,
        "frame_indices": frame_indices.tolist(),
        "timestamps": timestamps.tolist(),
        "notes": [
            f"expression_source: {expression_source}",
            f"head_source: {head_source}",
            f"jaw_source: {jaw_source}",
            "single-source experiment (mixed source forbidden)",
            "spectre_jawpose: SPECTRE raw jaw, preserved for audit/comparison",
            "emica_jawpose: EMICA raw jaw, preserved for audit/comparison",
            "shapecode: zeros(300) — MICA not loaded",
            "eye_pose: zeros(6) — Stage13 placeholder",
        ],
    }
    meta_path = clip_dir / "flame_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved metadata to %s", meta_path)

    np.save(str(clip_dir / "expression.npy"), expression)
    np.save(str(clip_dir / "jaw_pose.npy"), jaw_pose)
    np.save(str(clip_dir / "head_pose.npy"), head_pose)
    np.save(str(clip_dir / "eye_pose.npy"), eye_pose)
    logger.info("Saved individual .npy arrays to %s", clip_dir)
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage12 FLAME Parameter Extraction Runner")
    parser.add_argument("--input_metadata", required=True,
                        help="JSON file with aligned face metadata.")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for flame_features.")
    parser.add_argument("--emica_checkpoint", required=True,
                        help="Path to EMICA checkpoint (.ckpt).")
    parser.add_argument("--spectre_checkpoint", required=True,
                        help="Path to SPECTRE checkpoint (.tar).")
    parser.add_argument("--jaw_fusion_lambda", type=float, default=1.0,
                        help="DEPRECATED — no longer used: jaw_pose = EMICA "
                             "(final adopted scheme), no fusion. Value "
                             "retained for backward-compat/metadata only "
                             "(default: 1.0).")
    parser.add_argument("--expression_source", default="emica",
                        help="expression source: 'emica' (default) | "
                             "'spectre'. Single-source experiments only "
                             "(mixed source forbidden).")
    parser.add_argument("--head_source", default="emica",
                        help="head_pose source: 'emica' (default) | "
                             "'spectre'. Single-source experiments only "
                             "(mixed source forbidden).")
    parser.add_argument("--jaw_source", default="emica",
                        help="jaw_pose source: 'emica' (default, production "
                             "scheme) | 'spectre'. Single-source experiments "
                             "only (mixed source forbidden).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Inference batch size (default: 8).")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (default: cuda).")
    parser.add_argument("--clip_id", default=None,
                        help="Clip ID (auto-detected if not provided).")
    args = parser.parse_args()

    input_meta_path = Path(args.input_metadata)
    if not input_meta_path.exists():
        logger.error("Input metadata not found: %s", input_meta_path)
        sys.exit(1)

    with open(input_meta_path, "r") as f:
        faces = json.load(f)

    if not faces:
        logger.error("No faces in input metadata.")
        sys.exit(1)

    clip_id = args.clip_id or faces[0].get("clip_id", "unknown_clip")
    logger.info("Processing clip: %s, %d frames", clip_id, len(faces))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"

    emica_model = _load_emica(args.emica_checkpoint, device)
    spectre_model = _load_spectre(args.spectre_checkpoint, device)

    results = run_stage12(
        faces=faces,
        emica_model=emica_model,
        spectre_model=spectre_model,
        jaw_fusion_lambda=args.jaw_fusion_lambda,
        batch_size=args.batch_size,
        device=device,
        expression_source=args.expression_source,
        head_source=args.head_source,
        jaw_source=args.jaw_source,
    )

    logger.info("Inference complete: %d/%d frames processed",
                len(results), len(faces))

    output_dir = Path(args.output_dir)
    save_outputs(
        results=results,
        output_dir=output_dir,
        clip_id=clip_id,
        jaw_fusion_lambda=args.jaw_fusion_lambda,
        expression_source=args.expression_source,
        head_source=args.head_source,
        jaw_source=args.jaw_source,
    )

    if results:
        exp_arr = np.array([r["expression_code"] for r in results])
        logger.info("Expression stats: mean=%.4f, std=%.4f, all_zero=%s",
                     exp_arr.mean(), exp_arr.std(),
                     bool(np.allclose(exp_arr, 0.0)))
        logger.info("Output saved to: %s", output_dir / clip_id)

    logger.info("Stage12 runner completed successfully.")


if __name__ == "__main__":
    main()