"""Stage 12 FLAME parameter extraction — EMICA + SPECTRE dual-model inference."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from face_pipeline_v2.data.models import AlignedFaceInfo, FLAMEFeature

logger = logging.getLogger(__name__)

# 允许的 FLAME 参数来源 (AISDD Stage12 EMICA vs SPECTRE ablation).
# 实验禁止 mixed source (曾导致嘴部 mesh 扭曲、参数空间不一致)。
_ALLOWED_SOURCES = ("emica", "spectre")


def validate_sources(
    expression_source: str,
    head_source: str,
    jaw_source: str,
) -> None:
    """校验 Stage12 FLAME 参数来源组合 (禁止 mixed source)。

    Args:
        expression_source: 表情来源 ("emica" | "spectre")。
        head_source: 头部姿态来源 ("emica" | "spectre")。
        jaw_source: 下颌姿态来源 ("emica" | "spectre")。

    Raises:
        ValueError: 来源非法, 或三个来源不一致 (mixed source — 已知会
            导致嘴部 mesh 扭曲与参数空间不一致, 实验禁止)。
    """
    for name, src in (
        ("expression_source", expression_source),
        ("head_source", head_source),
        ("jaw_source", jaw_source),
    ):
        if src not in _ALLOWED_SOURCES:
            raise ValueError(
                f"Invalid {name}={src!r}. Allowed: {_ALLOWED_SOURCES}."
            )
    if not (expression_source == head_source == jaw_source):
        raise ValueError(
            "Mixed FLAME parameter sources are FORBIDDEN: "
            f"expression_source={expression_source}, head_source={head_source}, "
            f"jaw_source={jaw_source}. Use all 'emica' (Experiment A) or all "
            "'spectre' (Experiment B). Mixing is known to distort the mouth "
            "mesh / break parameter-space consistency."
        )


def _pad_expression(exp: np.ndarray, target_dim: int = 100) -> np.ndarray:
    """Zero-pad an expression vector to ``target_dim`` (50-dim SPECTRE -> 100)."""
    exp = np.asarray(exp, dtype=np.float32).reshape(-1)
    if exp.shape[0] == target_dim:
        return exp
    if exp.shape[0] > target_dim:
        return exp[:target_dim].copy()
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: exp.shape[0]] = exp
    return padded


def extract_flame_parameters(
    aligned_faces: List[AlignedFaceInfo],
    emica_model: Any,
    spectre_model: Any,
    jaw_fusion_lambda: float = 1.0,
    batch_size: int = 8,
    device: str = "cpu",
    expression_source: str = "emica",
    head_source: str = "emica",
    jaw_source: str = "emica",
) -> List[FLAMEFeature]:
    """Run EMICA + SPECTRE dual-model inference with per-field source selection.

    Stage12 EMICA vs SPECTRE full-parameter ablation (AISDD):
      - Experiment A (production default): all sources = EMICA
            expression  = EMICA expcode      [100]
            head_pose   = EMICA globalpose   [3]
            jaw_pose    = EMICA jawpose      [3]
      - Experiment B: all sources = SPECTRE
            expression  = SPECTRE expression [100] (50-dim latent slice,
                          zero-padded — see SPECTREWrapper)
            head_pose   = SPECTRE head_pose  [3]
            jaw_pose    = SPECTRE jaw_pose   [3]

    Mixed sources are FORBIDDEN (validate_sources raises ValueError).

    Args:
        aligned_faces: List of AlignedFaceInfo from Stage 11.
        emica_model: Loaded EMICAWrapper instance.
        spectre_model: Loaded SPECTREWrapper instance.
        jaw_fusion_lambda: DEPRECATED — no longer participates in computation.
                           Retained only for backward-compat interface /
                           metadata audit. jaw_pose comes from ``jaw_source``.
        expression_source: Expression source ("emica" | "spectre").
        head_source: Head pose source ("emica" | "spectre").
        jaw_source: Jaw pose source ("emica" | "spectre").
        batch_size: Inference batch size.
        device: Device to run inference on.

    Returns:
        List of FLAMEFeature, one per input face.

    Raises:
        ValueError: On invalid or mixed sources.
    """
    validate_sources(expression_source, head_source, jaw_source)

    if not aligned_faces:
        return []

    results: List[FLAMEFeature] = []
    n_faces = len(aligned_faces)

    for batch_start in range(0, n_faces, batch_size):
        batch_end = min(batch_start + batch_size, n_faces)
        batch_faces = aligned_faces[batch_start:batch_end]

        # Load and preprocess images
        images = []
        frame_indices = []
        timestamps = []
        clip_ids = []
        for face_info in batch_faces:
            img_path = Path(face_info.aligned_face_path)
            if not img_path.exists():
                logger.warning(
                    "Aligned face image not found, skipping: %s", img_path
                )
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(
                    "Failed to read aligned face image: %s", img_path
                )
                continue
            tensor = emica_model.preprocess_image(img, target_size=(224, 224))
            images.append(tensor)
            frame_indices.append(face_info.frame_index)
            timestamps.append(face_info.timestamp)
            clip_ids.append(face_info.clip_id)

        if not images:
            continue

        batch_tensor = torch.cat(images, dim=0).to(device)

        # Run EMICA encoder
        emica_output = emica_model.encode(batch_tensor)
        # Run SPECTRE encoder
        spectre_output = spectre_model.encode(batch_tensor)

        # Extract per-sample parameters
        B = batch_tensor.shape[0]
        for i in range(B):
            # EMICA outputs
            emica_exp = emica_output["expcode"][i].cpu().numpy()       # [100]
            emica_jaw = emica_output["jawpose"][i].cpu().numpy()        # [3]
            emica_global = emica_output["globalpose"][i].cpu().numpy()  # [3]

            # SPECTRE raw outputs (always kept for audit / source selection)
            spectre_exp = _pad_expression(
                spectre_output["expression"][i].cpu().numpy(), 100
            )                                                        # [100]
            spectre_head = spectre_output["head_pose"][i].cpu().numpy()  # [3]
            spectre_jaw = spectre_output["jaw_pose"][i].cpu().numpy()    # [3]

            # --- Per-field source selection (single-source experiments only) ---
            # Production default (Experiment A): all EMICA.
            expression = (
                emica_exp if expression_source == "emica" else spectre_exp
            )
            head_pose = (
                emica_global if head_source == "emica" else spectre_head
            )
            jaw_pose = emica_jaw if jaw_source == "emica" else spectre_jaw

            # --- Build FLAMEFeature ---
            feature = FLAMEFeature(
                frame_index=frame_indices[i],
                shapecode=tuple([0.0] * 300),       # MICA not loaded
                expcode=tuple(expression.tolist()),  # [100] (EMICA or SPECTRE)
                jawpose=tuple(jaw_pose.tolist()),    # [3] selected source
                globalpose=tuple(head_pose.tolist()),  # [3] selected source
                cam=tuple([0.0] * 3),               # not used (DECA off)
                eye_pose=tuple([0.0] * 6),          # fixed zeros, Stage13 placeholder
                spectre_jawpose=tuple(spectre_jaw.tolist()),  # audit trail
                emica_jawpose=tuple(emica_jaw.tolist()),      # debug/analysis
                timestamp=timestamps[i],
                clip_id=clip_ids[i],
            )
            results.append(feature)

    return results