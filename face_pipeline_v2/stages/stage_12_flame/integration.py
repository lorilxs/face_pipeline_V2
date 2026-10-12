"""Stage12 pipeline integration — subprocess bridge for Docker EMICA/SPECTRE.

Stage12 must run inside the ``face_pipeline_lxs`` Docker container with
``emica_ppu_env`` activated because EMICA/SPECTRE depend on torch+CUDA and
inferno packages not available in the host Python environment.

This module provides:
- ``stage12_aligned_faces_to_metadata()``: converts AlignedFaceInfo list to
  JSON metadata file for the subprocess.
- ``invoke_stage12_runner()``: calls the Docker subprocess.
- ``read_stage12_outputs()``: reads the .npz and .json output back into
  FLAMEFeature objects.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from face_pipeline_v2.data.models import AlignedFaceInfo, FLAMEFeature

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def stage12_aligned_faces_to_metadata(
    aligned_faces: List[AlignedFaceInfo],
    metadata_path: Path,
) -> None:
    """Serialize AlignedFaceInfo list to JSON metadata for the subprocess.

    Args:
        aligned_faces: List of AlignedFaceInfo from Stage 11.
        metadata_path: Output JSON path.
    """
    records: List[Dict[str, Any]] = []
    for face in aligned_faces:
        records.append({
            "frame_index": face.frame_index,
            "timestamp": face.timestamp,
            "clip_id": face.clip_id,
            "aligned_face_path": str(face.aligned_face_path),
        })
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def invoke_stage12_runner(
    input_metadata: Path,
    output_dir: Path,
    emica_checkpoint: str,
    spectre_checkpoint: str,
    jaw_fusion_lambda: float = 1.0,
    expression_source: str = "emica",
    head_source: str = "emica",
    jaw_source: str = "emica",
    batch_size: int = 8,
    device: str = "cuda",
    clip_id: Optional[str] = None,
    timeout: int = 3600,
) -> None:
    """Invoke Stage12 runner inside Docker container.

    Args:
        input_metadata: Path to JSON metadata file.
        output_dir: Output directory for FLAME features.
        emica_checkpoint: Path to EMICA checkpoint.
        spectre_checkpoint: Path to SPECTRE checkpoint.
        jaw_fusion_lambda: DEPRECATED — no longer used (metadata only).
        expression_source: Expression source ("emica" | "spectre").
        head_source: Head pose source ("emica" | "spectre").
        jaw_source: Jaw pose source ("emica" | "spectre").
        batch_size: Inference batch size.
        device: Device for inference.
        clip_id: Clip ID (auto-detected if None).
        timeout: Timeout in seconds.

    Raises:
        subprocess.TimeoutExpired: If the subprocess times out.
        RuntimeError: If the subprocess returns non-zero exit code.
    """
    runner_path = _PROJECT_ROOT / "face_pipeline_v2" / "stages" / "stage_12_flame" / "runner.py"

    cmd_parts = [
        "source /workspace/emica_ppu_env/bin/activate &&",
        "python", str(runner_path),
        "--input_metadata", str(input_metadata),
        "--output_dir", str(output_dir),
        "--emica_checkpoint", str(emica_checkpoint),
        "--spectre_checkpoint", str(spectre_checkpoint),
        "--jaw_fusion_lambda", str(jaw_fusion_lambda),
        "--expression_source", expression_source,
        "--head_source", head_source,
        "--jaw_source", jaw_source,
        "--batch_size", str(batch_size),
        "--device", device,
    ]
    if clip_id:
        cmd_parts.extend(["--clip_id", clip_id])

    cmd = " ".join(cmd_parts)
    docker_cmd = [
        "docker", "exec", "-i", "face_pipeline_lxs",
        "bash", "-c", cmd,
    ]

    print(f"[Stage12 Integration] Running: {' '.join(docker_cmd)}")
    result = subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.stdout:
        print(f"[Stage12 stdout]\n{result.stdout}")
    if result.stderr:
        print(f"[Stage12 stderr]\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage12 runner failed with exit code {result.returncode}\n"
            f"stderr: {result.stderr[-2000:]}"
        )


def read_stage12_outputs(
    output_dir: Path,
    clip_id: str,
) -> List[FLAMEFeature]:
    """Read Stage12 .npz output and convert to FLAMEFeature objects.

    Args:
        output_dir: Base output directory.
        clip_id: Clip ID (fallback if npz has no per-frame clip_id).

    Returns:
        List of FLAMEFeature objects.

    Raises:
        FileNotFoundError: If the .npz file is not found.
    """
    npz_path = output_dir / clip_id / "flame_features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"FLAME features not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    T = data["expression_code"].shape[0]

    # --- camera params: read from npz if available, else fallback to zeros ---
    if "cam" in data:
        cam_array = data["cam"]
    elif "camera_params" in data:
        cam_array = data["camera_params"]
    else:
        cam_array = None  # will fallback to zeros per-frame

    # --- timestamp: read from npz if available ---
    has_timestamps = "timestamp" in data
    # --- clip_id: read from npz if available (per-frame) ---
    has_clip_ids = "clip_id" in data
    # --- emica_jawpose: optional debug field (absent in pre-adoption npz) ---
    has_emica_jaw = "emica_jawpose" in data

    features: List[FLAMEFeature] = []
    for i in range(T):
        # Camera
        if cam_array is not None:
            cam_val = tuple(cam_array[i].tolist())
        else:
            cam_val = tuple([0.0] * 3)

        # Timestamp
        ts_val = float(data["timestamp"][i]) if has_timestamps else 0.0

        # Clip ID (per-frame or fallback to function arg)
        if has_clip_ids:
            cid_val = str(data["clip_id"][i])
        else:
            cid_val = clip_id

        feature = FLAMEFeature(
            frame_index=int(data["frame_index"][i]),
            shapecode=tuple(data["shapecode"][i].tolist()),
            expcode=tuple(data["expression_code"][i].tolist()),
            jawpose=tuple(data["jaw_pose"][i].tolist()),
            globalpose=tuple(data["head_pose"][i].tolist()),
            cam=cam_val,
            eye_pose=tuple(data["eye_pose"][i].tolist()),
            spectre_jawpose=tuple(data["spectre_jawpose"][i].tolist()),
            emica_jawpose=(tuple(data["emica_jawpose"][i].tolist())
                           if has_emica_jaw else None),
            timestamp=ts_val,
            clip_id=cid_val,
        )
        features.append(feature)

    return features