#!/usr/bin/env python3
"""Stage12 Full Pipeline End-to-End Test (Subprocess Mode).

Uses existing Stage 0-11 output from ``outputs/stage0_11_test_verify/``
to run Stage12 FLAME extraction in subprocess mode (Docker container).

This test:
1. Reads aligned face images from Stage11 output
2. Constructs AlignedFaceInfo objects
3. Runs Stage12 in subprocess mode (docker exec face_pipeline_lxs)
4. Validates output FLAMEFeature shapes and content

Usage (from project root, on host machine):
    python tests/manual/test_full_pipeline_stage12.py

Prerequisites:
    - Docker container ``face_pipeline_lxs`` must be running
    - ``emica_env`` must exist in the container
    - ``outputs/stage0_11_test_verify/`` must exist (Stage 0-11 already run)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------- path setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------- pipeline deps
from face_pipeline_v2.config.config_loader import ConfigLoader
from face_pipeline_v2.data.models import AlignedFaceInfo, FLAMEFeature
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.utils.logger import PipelineLogger
from face_pipeline_v2.stages.stage_12_flame.stage import Stage12FLAME

# ============================================================================
# Configuration
# ============================================================================

# Existing Stage 0-11 output directory
STAGE0_11_OUTPUT = PROJECT_ROOT / "outputs" / "stage0_11_test_verify"

# Config file
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Stage12 output directory (within the test base)
STAGE12_OUTPUT = STAGE0_11_OUTPUT / "stage12_flame"

# Use only the first clip to keep test fast
MAX_CLIPS = 1
MAX_FACES_PER_CLIP = 10  # limit faces per clip for speed
def _parse_frame_index(filename: str) -> int:
    """Extract frame index from aligned face filename.

    Expected format: aligned_000100_CNFACE_..._face_000.jpg
    """
    try:
        parts = filename.split("_")
        if len(parts) >= 2 and parts[0] == "aligned":
            return int(parts[1])
    except (ValueError, IndexError):
        pass
    return 0


def _build_aligned_faces(
    aligned_dir: Path,
    clip_id: str,
    max_faces: int = 0,
) -> List[AlignedFaceInfo]:
    """Build AlignedFaceInfo list from aligned face image files."""
    faces: List[AlignedFaceInfo] = []
    image_files = sorted(aligned_dir.glob("*.jpg"))

    if max_faces > 0:
        image_files = image_files[:max_faces]

    for img_path in image_files:
        frame_idx = _parse_frame_index(img_path.name)
        face = AlignedFaceInfo(
            clip_id=clip_id,
            frame_index=frame_idx,
            face_id=f"{clip_id}_face_000",
            timestamp=frame_idx / 25.0,
            aligned_face_path=img_path,
            affine_matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            original_image_size=(256, 256),
            aligned_resolution=(256, 256),
        )
        faces.append(face)

def _validate_flame_features(
    features: List[FLAMEFeature],
    expected_count: int,
) -> Dict[str, Any]:
    """Validate FLAMEFeature output dimensions and content."""
    report: Dict[str, Any] = {
        "count": len(features),
        "expected_count": expected_count,
        "count_ok": len(features) == expected_count,
        "checks": [],
    }

    if not features:
        report["checks"].append({
            "check": "non_empty", "ok": False, "msg": "No features returned"
        })
        return report

    report["checks"].append({"check": "non_empty", "ok": True})
    f = features[0]

    # expcode [100]
    exp_ok = len(f.expcode) == 100
    report["checks"].append({
        "check": "expcode[100]", "ok": exp_ok,
        "actual": len(f.expcode),
        "sample_first_5": list(f.expcode[:5]) if exp_ok else None,
    })

    # jawpose [3]
    jaw_ok = len(f.jawpose) == 3
    report["checks"].append({
        "check": "jawpose[3]", "ok": jaw_ok,
        "actual": len(f.jawpose),
        "values": list(f.jawpose) if jaw_ok else None,
    })

    # globalpose [3]
    glob_ok = len(f.globalpose) == 3
    report["checks"].append({
        "check": "globalpose[3]", "ok": glob_ok,
        "actual": len(f.globalpose),
        "values": list(f.globalpose) if glob_ok else None,
    })

    # eye_pose [6]
    eye_ok = len(f.eye_pose) == 6
    report["checks"].append({
        "check": "eye_pose[6]", "ok": eye_ok,
        "actual": len(f.eye_pose),
        "values": list(f.eye_pose) if eye_ok else None,
    })

    # shapecode [300]
    shape_ok = len(f.shapecode) == 300
    report["checks"].append({
        "check": "shapecode[300]", "ok": shape_ok,
        "actual": len(f.shapecode),
    })

    # cam [3]
    cam_ok = len(f.cam) == 3
    report["checks"].append({
        "check": "cam[3]", "ok": cam_ok,
        "actual": len(f.cam),
        "values": list(f.cam) if cam_ok else None,
    })

    # posecode = globalpose + jawpose = [6]
    pose_ok = len(f.posecode) == 6
    report["checks"].append({
        "check": "posecode[6] (globalpose+jawpose)",
        "ok": pose_ok, "actual": len(f.posecode),
    })

    # spectre_jawpose [3]
    if f.spectre_jawpose is not None:
        sp_ok = len(f.spectre_jawpose) == 3
        report["checks"].append({
            "check": "spectre_jawpose[3]", "ok": sp_ok,
            "actual": len(f.spectre_jawpose),
            "values": list(f.spectre_jawpose) if sp_ok else None,
        })
    else:
        report["checks"].append({
            "check": "spectre_jawpose", "ok": False,
            "msg": "spectre_jawpose is None (expected non-None)",
        })

    # Check that expcode is not all zeros
    all_exps = np.array([list(f.expcode) for f in features])
    exp_mean = float(all_exps.mean())
    exp_std = float(all_exps.std())
    exp_not_zero = not np.allclose(all_exps, 0.0)
    report["checks"].append({
        "check": "expcode not all zeros", "ok": exp_not_zero,
        "mean": round(exp_mean, 6), "std": round(exp_std, 6),
    })

    # Check that jawpose is not all zeros
    all_jaws = np.array([list(f.jawpose) for f in features])
    jaw_not_zero = not np.allclose(all_jaws, 0.0)
    report["checks"].append({
        "check": "jawpose not all zeros", "ok": jaw_not_zero,
        "mean": [round(float(all_jaws[:, i].mean()), 6) for i in range(3)],
    })

    # Check that eye_pose is all zeros (placeholder)
    all_eyes = np.array([list(f.eye_pose) for f in features])
    eye_all_zero = bool(np.allclose(all_eyes, 0.0))
    report["checks"].append({
        "check": "eye_pose all zeros (placeholder)", "ok": eye_all_zero,
    })

    # Check that shapecode is all zeros (MICA not loaded)
    all_shapes = np.array([list(f.shapecode) for f in features])
    shape_all_zero = bool(np.allclose(all_shapes, 0.0))
    report["checks"].append({
        "check": "shapecode all zeros (MICA not loaded)", "ok": shape_all_zero,
    })

    # Overall: all dimensional checks passed
    all_dims_ok = all(
        c.get("ok", False) for c in report["checks"]
        if c["check"] not in (
            "shapecode all zeros (MICA not loaded)",
            "eye_pose all zeros (placeholder)",
        )
    )
    report["all_dims_ok"] = all_dims_ok
    return report
def _check_docker_environment() -> Dict[str, Any]:
    """Check if Docker container and emica_env are available."""
    import subprocess as sp

    env_report: Dict[str, Any] = {}

    # Check docker container is running
    try:
        result = sp.run(
            ["docker", "exec", "-i", "face_pipeline_lxs", "echo", "OK"],
            capture_output=True, text=True, timeout=5,
        )
        env_report["docker_container_running"] = (
            result.returncode == 0 and "OK" in result.stdout
        )
        if result.stderr:
            env_report["docker_stderr"] = result.stderr.strip()
    except Exception as e:
        env_report["docker_container_running"] = False
        env_report["docker_error"] = str(e)

    # Check emica_env exists
    if env_report.get("docker_container_running"):
        try:
            result = sp.run(
                ["docker", "exec", "-i", "face_pipeline_lxs",
                 "bash", "-c",
                 "source /workspace/emica_env/bin/activate && "
                 "python -c 'import torch; print(torch.__version__)'"],
                capture_output=True, text=True, timeout=10,
            )
            env_report["emica_env_available"] = result.returncode == 0
            if result.stdout.strip():
                env_report["torch_version"] = result.stdout.strip()
            if result.stderr:
                env_report["emica_stderr"] = result.stderr.strip()[:500]
        except Exception as e:
            env_report["emica_env_available"] = False
            env_report["emica_error"] = str(e)

    # Check checkpoint files exist
    emica_ckpt = Path(
        "models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/"
        "model-epoch=38-val/loss_total=3.88806534.ckpt"
    )
    spectre_ckpt = Path(
        "third_party/inferno/spectre/spectre/data/deca_model.tar"
    )
    env_report["emica_ckpt_exists"] = emica_ckpt.exists()
    env_report["emica_ckpt_path"] = str(emica_ckpt.resolve())
    env_report["spectre_ckpt_exists"] = spectre_ckpt.exists()
    env_report["spectre_ckpt_path"] = str(spectre_ckpt.resolve())

    return env_report
def main() -> int:
    """Run Stage12 end-to-end test."""
    print("=" * 72)
    print("Stage12 Full Pipeline End-to-End Test")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------ Step 1
    print("[1/5] Checking Docker environment...")
    env_report = _check_docker_environment()
    print(f"  Docker container running: {env_report['docker_container_running']}")
    print(f"  emica_env available: {env_report.get('emica_env_available', 'N/A')}")
    print(f"  Torch version: {env_report.get('torch_version', 'N/A')}")
    print(f"  EMICA ckpt exists: {env_report['emica_ckpt_exists']}")
    print(f"  SPECTRE ckpt exists: {env_report['spectre_ckpt_exists']}")
    print()

    if not env_report["docker_container_running"]:
        print("ERROR: Docker container 'face_pipeline_lxs' is not running.")
        print("  Start it with: docker start face_pipeline_lxs")
        return 1

    # ------------------------------------------------------------------ Step 2
    print("[2/5] Loading pipeline config...")
    loader = ConfigLoader(config_path=CONFIG_PATH)
    config = loader.load()
    loader.validate(config)

    config["storage"]["output_dir"] = str(STAGE0_11_OUTPUT)
    config["stage12"]["output_dir"] = str(STAGE12_OUTPUT)
    config["stage12"]["runtime_mode"] = "subprocess"
    config["stage12"]["batch_size"] = 4
    print(f"  Config: {CONFIG_PATH}")
    print(f"  runtime_mode: {config['stage12']['runtime_mode']}")
    print(f"  output_dir: {STAGE12_OUTPUT}")
    print()

    # ------------------------------------------------------------------ Step 3
    print("[3/5] Building AlignedFaceInfo from Stage 0-11 output...")
    aligned_root = STAGE0_11_OUTPUT / "aligned_faces"
    if not aligned_root.exists():
        print(f"ERROR: Aligned faces dir not found: {aligned_root}")
        return 1

    all_faces: List[AlignedFaceInfo] = []
    clip_dirs = sorted(aligned_root.iterdir())
    if not clip_dirs:
        print("ERROR: No clip directories found.")
        return 1

    print(f"  Found {len(clip_dirs)} clip directories")
    for clip_dir in clip_dirs[:MAX_CLIPS]:
        clip_id = clip_dir.name
        faces = _build_aligned_faces(clip_dir, clip_id, MAX_FACES_PER_CLIP)
        all_faces.extend(faces)
        print(f"    {clip_id}: {len(faces)} faces")

    print(f"  Total faces: {len(all_faces)}")
    print()

    if not all_faces:
        print("ERROR: No aligned faces found.")
        return 1

    # ------------------------------------------------------------------ Step 4
    print("[4/5] Running Stage12 (subprocess mode)...")
    reports_dir = PROJECT_ROOT / "tests" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger = PipelineLogger(name="stage12_e2e_test", log_dir=reports_dir)
    context = PipelineContext(
        config=config,
        logger=logger,
        model_manager=None,
        runtime_info={
            "video_id": "stage0_11_test_verify",
            "clip_id": all_faces[0].clip_id if all_faces else "unknown",
        },
    )

    stage = Stage12FLAME()
    t0 = time.monotonic()
    try:
        flame_features = stage.run(all_faces, context)
        elapsed = round(time.monotonic() - t0, 2)
        print(f"  Stage12 completed in {elapsed}s")
        print(f"  Generated {len(flame_features)} FLAMEFeature(s)")
    except Exception as e:
        elapsed = round(time.monotonic() - t0, 2)
        print(f"  Stage12 FAILED after {elapsed}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()

    # ------------------------------------------------------------------ Step 5
    print("[5/5] Validating FLAMEFeature output...")
    report = _validate_flame_features(flame_features, len(all_faces))
    print(f"  Feature count: {report['count']} (expected {report['expected_count']})")
    print(f"  Count OK: {report['count_ok']}")
    for check in report["checks"]:
        status = "PASS" if check.get("ok") else "FAIL"
        extra = ""
        if "actual" in check:
            extra = f" (actual={check['actual']})"
        if "values" in check:
            extra += f" values={check['values']}"
        if "mean" in check and "std" in check:
            extra += f" mean={check['mean']} std={check['std']}"
        if "msg" in check:
            extra += f" msg={check['msg']}"
        print(f"    [{status}] {check['check']}{extra}")
    print()

    # ------------------------------------------------------------------ Summary
    print("=" * 72)
    overall = "PASS" if report["all_dims_ok"] else "FAIL"
    print(f"STAGE12 E2E TEST: {overall}")
    print(f"  Environment: docker={env_report['docker_container_running']} "
          f"emica_env={env_report.get('emica_env_available', 'N/A')}")
    print(f"  Input: {len(all_faces)} faces from {len(clip_dirs[:MAX_CLIPS])} clip(s)")
    print(f"  Output: {len(flame_features)} FLAMEFeature(s)")
    print(f"  Elapsed: {elapsed}s")
    print(f"  All dim checks: {report['all_dims_ok']}")
    print("=" * 72)

    # Write JSON report
    report_path = reports_dir / "stage12_e2e_test_report.json"
    full_report = {
        "overall": overall,
        "environment": env_report,
        "input": {
            "source": str(STAGE0_11_OUTPUT),
            "clips": [d.name for d in clip_dirs[:MAX_CLIPS]],
            "total_faces": len(all_faces),
        },
        "output": {
            "feature_count": len(flame_features),
            "elapsed_s": elapsed,
        },
        "validation": report,
    }
    report_path.write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nDetailed report: {report_path}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
    return faces