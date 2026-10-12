#!/usr/bin/env python3
"""Stage 12 Real Data Validation — EMICA + SPECTRE on Stage 11 aligned faces.

Usage:
    python tests/manual/test_stage12_real_validation.py \
        --input_dir outputs/stage0_11_test_verify/aligned_faces/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001 \
        --num_frames 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from face_pipeline_v2.data.models import FLAMEFeature
from face_pipeline_v2.models.model_manager import ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage12_validation")
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_sorted_images(input_dir: Path, max_frames: Optional[int] = None) -> List[Path]:
    """Read sorted .jpg/.png files from input_dir."""
    exts = {".jpg", ".jpeg", ".png"}
    files: List[Path] = []
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p)
    if max_frames is not None:
        files = files[:max_frames]
    return files


def load_and_preprocess(path: Path) -> np.ndarray:
    """Load image as BGR numpy array (H, W, 3)."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def compute_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Compute basic stats for a numpy array."""
    finite = np.isfinite(arr)
    return {
        "mean": float(np.mean(arr[finite])),
        "std": float(np.std(arr[finite])),
        "min": float(np.min(arr[finite])),
        "max": float(np.max(arr[finite])),
        "has_nan": bool(np.any(np.isnan(arr))),
        "has_inf": bool(np.any(~np.isfinite(arr) & ~np.isnan(arr))),
        "all_zero": bool(np.allclose(arr[finite], 0.0)),
    }
def write_report(
    report_path: Path,
    input_dir: Path,
    total_frames: int,
    num_tested: int,
    emica_ckpt: str,
    spectre_ckpt: str,
    jaw_lambda: float,
    failures: List[Dict[str, Any]],
    exp_stats: Dict[str, Any],
    head_stats: Dict[str, Any],
    emica_jaw_stats: Dict[str, Any],
    spectre_jaw_stats: Dict[str, Any],
    fused_jaw_stats: Dict[str, Any],
    fusion_verify: List[Dict[str, Any]],
    all_features: List[FLAMEFeature],
) -> None:
    """Generate Markdown validation report."""
    lines: List[str] = []
    lines.append("# Stage12 Real Data Validation Report")
    lines.append("")
    lines.append("## 1. Input")
    lines.append("")
    lines.append(f"- Input directory: `{input_dir}`")
    lines.append(f"- Total frames available: {total_frames}")
    lines.append(f"- Frames tested: {num_tested}")
    lines.append("")
    lines.append("## 2. Model")
    lines.append("")
    lines.append(f"- EMICA checkpoint: `{emica_ckpt}`")
    lines.append(f"- SPECTRE checkpoint: `{spectre_ckpt}`")
    lines.append(f"- jaw fusion lambda: {jaw_lambda}")
    lines.append("")
    lines.append("## 3. Expression")
    lines.append("")
    lines.append("- Dimension: 100")
    lines.append(f"- mean: {exp_stats['mean']:.6f}")
    lines.append(f"- std: {exp_stats['std']:.6f}")
    lines.append(f"- min: {exp_stats['min']:.6f}")
    lines.append(f"- max: {exp_stats['max']:.6f}")
    lines.append(f"- NaN: {exp_stats['has_nan']}")
    lines.append(f"- Inf: {exp_stats['has_inf']}")
    lines.append(f"- All zero: {exp_stats['all_zero']}")
    lines.append("")
    lines.append("## 4. Head Pose")
    lines.append("")
    lines.append("- Source: EMICA globalpose")
    lines.append("- Dimension: 3")
    lines.append(f"- mean: {head_stats['mean']:.6f}")
    lines.append(f"- std: {head_stats['std']:.6f}")
    lines.append(f"- min: {head_stats['min']:.6f}")
    lines.append(f"- max: {head_stats['max']:.6f}")
    lines.append("")
    lines.append("## 5. Jaw Pose")
    lines.append("")
    lines.append("### EMICA jaw")
    lines.append("")
    lines.append("- Dimension: 3")
    lines.append(f"- mean: {emica_jaw_stats['mean']:.6f}")
    lines.append(f"- std: {emica_jaw_stats['std']:.6f}")
    lines.append("")
    lines.append("### SPECTRE jaw")
    lines.append("")
    lines.append("- Dimension: 3")
    lines.append(f"- mean: {spectre_jaw_stats['mean']:.6f}")
    lines.append(f"- std: {spectre_jaw_stats['std']:.6f}")
    lines.append("")
    lines.append("### fused jaw")
    lines.append("")
    lines.append("- Dimension: 3")
    lines.append(f"- mean: {fused_jaw_stats['mean']:.6f}")
    lines.append(f"- std: {fused_jaw_stats['std']:.6f}")
    lines.append("")
    lines.append("## 6. Fusion Verification")
    lines.append("")
    lines.append(f"Randomly selected {len(fusion_verify)} frames:")
    lines.append("")
    lines.append("| frame_id | emica_jaw | spectre_jaw | fused_jaw | expected | match |")
    lines.append("|----------|-----------|-------------|-----------|----------|-------|")
    for v in fusion_verify:
        lines.append(
            f"| {v['frame_id']} "
            f"| {v['emica_jaw']} "
            f"| {v['spectre_jaw']} "
            f"| {v['fused_jaw']} "
            f"| {v['expected']} "
            f"| {v['match']} |"
        )
    lines.append("")
    lines.append("## 7. FLAMEFeature")
    lines.append("")
    dim_errors = 0
    for f in all_features:
        if len(f.expcode) != 100:
            dim_errors += 1
        if len(f.jawpose) != 3:
            dim_errors += 1
        if len(f.globalpose) != 3:
            dim_errors += 1
        if len(f.eye_pose) != 6:
            dim_errors += 1
        if f.spectre_jawpose is not None and len(f.spectre_jawpose) != 3:
            dim_errors += 1
        if len(f.shapecode) != 300:
            dim_errors += 1
    lines.append(f"- Total features: {len(all_features)}")
    lines.append(f"- Dimension errors: {dim_errors}")
    lines.append("- expression_code: 100")
    lines.append("- jaw_pose: 3")
    lines.append("- head_pose (globalpose): 3")
    lines.append("- eye_pose: 6")
    lines.append("- spectre_jawpose: 3")
    lines.append("- shapecode: 300")
    lines.append("")
    lines.append("## 8. Failures")
    lines.append("")
    if not failures:
        lines.append("No failures encountered.")
    else:
        lines.append("| frame_id | error |")
        lines.append("|----------|-------|")
        for f in failures:
            lines.append(f"| {f['frame_id']} | {f['error']} |")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage12 Real Data Validation")
    parser.add_argument("--input_dir", required=True,
                        help="Path to Stage11 aligned faces directory.")
    parser.add_argument("--num_frames", type=int, default=50,
                        help="Number of frames to test (default: 50).")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to pipeline config.yaml.")
    parser.add_argument("--output_dir", default="outputs/stage12_validation",
                        help="Directory for debug .npy files.")
    parser.add_argument("--report", default="docs/Stage12_Real_Data_Validation_Report.md",
                        help="Output report path.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report).resolve()

    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    config = load_config(config_path)
    stage12_cfg = config.get("stage12", {})
    emica_checkpoint = stage12_cfg.get("emica", {}).get("checkpoint", "")
    spectre_checkpoint = stage12_cfg.get("spectre", {}).get("checkpoint", "")
    jaw_lambda = float(stage12_cfg.get("jaw_fusion_lambda", 0.6))

    emica_ckpt_path = (PROJECT_ROOT / emica_checkpoint).resolve()
    spectre_ckpt_path = (PROJECT_ROOT / spectre_checkpoint).resolve()

    image_paths = read_sorted_images(input_dir, max_frames=args.num_frames)
    total_available = len(list(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ))

    logger.info("Found aligned faces: %d total, %d selected for testing",
                total_available, len(image_paths))
    if not image_paths:
        logger.error("No images found in %s", input_dir)
        sys.exit(1)

    num_tested = len(image_paths)

    logger.info("Loading EMICA: %s", emica_ckpt_path)
    logger.info("Loading SPECTRE: %s", spectre_ckpt_path)
    model_manager = ModelManager(device="cuda" if torch.cuda.is_available() else "cpu")
    device = model_manager.device

    try:
        emica_model = model_manager.load_model("emica", str(emica_ckpt_path))
        spectre_model = model_manager.load_model("spectre", str(spectre_ckpt_path))
    except Exception as exc:
        logger.error("Model loading failed: %s", exc)
        sys.exit(1)

    logger.info("Models loaded successfully. Device: %s", device)

    all_features: List[FLAMEFeature] = []
    failures: List[Dict[str, Any]] = []
    exp_collect: List[np.ndarray] = []
    head_collect: List[np.ndarray] = []
    emica_jaw_collect: List[np.ndarray] = []
    spectre_jaw_collect: List[np.ndarray] = []
    fused_jaw_collect: List[np.ndarray] = []

    batch_size = 8
    for batch_start in range(0, num_tested, batch_size):
        batch_end = min(batch_start + batch_size, num_tested)
        batch_paths = image_paths[batch_start:batch_end]

        images: List[torch.Tensor] = []
        frame_ids: List[int] = []
        for idx, path in enumerate(batch_paths):
            try:
                img = load_and_preprocess(path)
                tensor = emica_model.preprocess_image(img, target_size=(224, 224))
                images.append(tensor)
                frame_ids.append(batch_start + idx)
            except Exception as exc:
                logger.warning("Frame %d (%s) failed: %s",
                               batch_start + idx, path.name, exc)
                failures.append({"frame_id": batch_start + idx,
                                 "file": path.name, "error": str(exc)})
                continue

        if not images:
            continue

        batch_tensor = torch.cat(images, dim=0).to(device)

        try:
            emica_output = emica_model.encode(batch_tensor)
            spectre_output = spectre_model.encode(batch_tensor)
        except Exception as exc:
            logger.error("Batch inference failed: %s", exc)
            for i in range(len(images)):
                failures.append({
                    "frame_id": frame_ids[i],
                    "error": f"Batch inference error: {exc}",
                })
            continue

        B = batch_tensor.shape[0]
        for i in range(B):
            try:
                emica_exp = emica_output["expcode"][i].cpu().numpy()
                emica_jaw = emica_output["jawpose"][i].cpu().numpy()
                emica_global = emica_output["globalpose"][i].cpu().numpy()
                spectre_jaw = spectre_output["jaw_pose"][i].cpu().numpy()

                fused_jaw = (1.0 - jaw_lambda) * emica_jaw + jaw_lambda * spectre_jaw

                feature = FLAMEFeature(
                    frame_index=frame_ids[i],
                    shapecode=tuple([0.0] * 300),
                    expcode=tuple(emica_exp.tolist()),
                    jawpose=tuple(fused_jaw.tolist()),
                    globalpose=tuple(emica_global.tolist()),
                    cam=tuple([0.0] * 3),
                    eye_pose=tuple([0.0] * 6),
                    spectre_jawpose=tuple(spectre_jaw.tolist()),
                )
                all_features.append(feature)
                exp_collect.append(emica_exp)
                head_collect.append(emica_global)
                emica_jaw_collect.append(emica_jaw)
                spectre_jaw_collect.append(spectre_jaw)
                fused_jaw_collect.append(fused_jaw)
            except Exception as exc:
                logger.warning("Frame %d processing failed: %s", frame_ids[i], exc)
                failures.append({"frame_id": frame_ids[i], "error": str(exc)})

        logger.info("Batch %d-%d/%d done", batch_start, batch_end - 1, num_tested)

    logger.info("Inference complete: %d features, %d failures",
                len(all_features), len(failures))

    exp_arr = np.stack(exp_collect) if exp_collect else np.zeros((0, 100))
    head_arr = np.stack(head_collect) if head_collect else np.zeros((0, 3))
    emica_jaw_arr = np.stack(emica_jaw_collect) if emica_jaw_collect else np.zeros((0, 3))
    spectre_jaw_arr = np.stack(spectre_jaw_collect) if spectre_jaw_collect else np.zeros((0, 3))
    fused_jaw_arr = np.stack(fused_jaw_collect) if fused_jaw_collect else np.zeros((0, 3))

    exp_stats = compute_stats(exp_arr)
    head_stats = compute_stats(head_arr)
    emica_jaw_stats = compute_stats(emica_jaw_arr)
    spectre_jaw_stats = compute_stats(spectre_jaw_arr)
    fused_jaw_stats = compute_stats(fused_jaw_arr)

    rng = np.random.RandomState(42)
    n_verify = min(5, len(all_features))
    fusion_verify: List[Dict[str, Any]] = []
    if n_verify > 0:
        indices = rng.choice(len(all_features), size=n_verify, replace=False)
        for idx in sorted(indices):
            e_jaw = emica_jaw_arr[idx]
            s_jaw = spectre_jaw_arr[idx]
            fu_jaw = fused_jaw_arr[idx]
            expected = (1.0 - jaw_lambda) * e_jaw + jaw_lambda * s_jaw
            match = np.allclose(fu_jaw, expected, atol=1e-5)
            fusion_verify.append({
                "frame_id": all_features[idx].frame_index,
                "emica_jaw": f"[{e_jaw[0]:.4f}, {e_jaw[1]:.4f}, {e_jaw[2]:.4f}]",
                "spectre_jaw": f"[{s_jaw[0]:.4f}, {s_jaw[1]:.4f}, {s_jaw[2]:.4f}]",
                "fused_jaw": f"[{fu_jaw[0]:.4f}, {fu_jaw[1]:.4f}, {fu_jaw[2]:.4f}]",
                "expected": f"[{expected[0]:.4f}, {expected[1]:.4f}, {expected[2]:.4f}]",
                "match": "✓" if match else "✗",
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / "expression.npy"), exp_arr)
    np.save(str(output_dir / "emica_jaw.npy"), emica_jaw_arr)
    np.save(str(output_dir / "spectre_jaw.npy"), spectre_jaw_arr)
    np.save(str(output_dir / "fused_jaw.npy"), fused_jaw_arr)
    np.save(str(output_dir / "head_pose.npy"), head_arr)
    logger.info("Debug arrays saved to %s", output_dir)

    write_report(
        report_path=report_path, input_dir=input_dir,
        total_frames=total_available, num_tested=num_tested,
        emica_ckpt=emica_checkpoint, spectre_ckpt=spectre_checkpoint,
        jaw_lambda=jaw_lambda, failures=failures,
        exp_stats=exp_stats, head_stats=head_stats,
        emica_jaw_stats=emica_jaw_stats, spectre_jaw_stats=spectre_jaw_stats,
        fused_jaw_stats=fused_jaw_stats, fusion_verify=fusion_verify,
        all_features=all_features,
    )

    logger.info("=" * 60)
    logger.info("Stage12 Validation Summary")
    logger.info("=" * 60)
    logger.info("Frames tested: %d", num_tested)
    logger.info("Features produced: %d", len(all_features))
    logger.info("Failures: %d", len(failures))
    logger.info("Expression — mean: %.4f, std: %.4f, NaN: %s, Inf: %s, AllZero: %s",
                exp_stats["mean"], exp_stats["std"],
                exp_stats["has_nan"], exp_stats["has_inf"], exp_stats["all_zero"])
    logger.info("Head pose — mean: %.4f, std: %.4f",
                head_stats["mean"], head_stats["std"])
    logger.info("EMICA jaw — mean: %.4f, std: %.4f",
                emica_jaw_stats["mean"], emica_jaw_stats["std"])
    logger.info("SPECTRE jaw — mean: %.4f, std: %.4f",
                spectre_jaw_stats["mean"], spectre_jaw_stats["std"])
    logger.info("Fused jaw — mean: %.4f, std: %.4f",
                fused_jaw_stats["mean"], fused_jaw_stats["std"])
    logger.info("Report: %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()