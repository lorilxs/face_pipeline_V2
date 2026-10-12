#!/usr/bin/env python3
"""Verify which execution backend SCRFD ONNX actually uses.

Prints:
    1. onnxruntime version
    2. onnxruntime available providers
    3. detected runtime backend (PPU / CUDA / CPU)
    4. the provider SCRFD actually runs on (session.get_providers())
    5. single-image inference latency (ms)

Usage (from the Face_Pipeline_V2 project root):
    python3 scripts/test_scrfd_backend.py
    python3 scripts/test_scrfd_backend.py --execution-provider cpu
    python3 scripts/test_scrfd_backend.py --execution-provider ppu --image /path/to/img.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from the project root without installing the package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SCRFD ONNX backend.")
    parser.add_argument(
        "--model", default=str(_PROJECT_ROOT / "models" / "SCRFD"),
        help="Path to the SCRFD model directory (default: models/SCRFD).",
    )
    parser.add_argument(
        "--execution-provider", default="auto",
        choices=("auto", "ppu", "cuda", "cpu"),
        help="Execution provider preference (default: auto).",
    )
    parser.add_argument(
        "--image", default=None,
        help="Optional image path for a real detection. Defaults to a synthetic image.",
    )
    parser.add_argument(
        "--num-runs", type=int, default=20,
        help="Number of timed inference runs (default: 20).",
    )
    args = parser.parse_args()

    # 1. onnxruntime version + providers
    _print_section("1. ONNX Runtime")
    try:
        import onnxruntime as ort
        print(f"onnxruntime version : {ort.__version__}")
        print(f"available providers : {ort.get_available_providers()}")
    except ImportError as exc:
        print(f"onnxruntime NOT installed: {exc}")
        return 1

    # 2. Detected runtime backend
    _print_section("2. Detected runtime backend")
    from face_pipeline_v2.runtime.backend import get_backend
    runtime = get_backend()
    print(f"backend             : {runtime.backend.value}")
    print(f"onnx_active         : {runtime.onnx_active}")
    print(f"torch_available     : {runtime.torch_available} ({runtime.torch_backend})")

    # 3. Load SCRFD and report actual provider
    _print_section("3. SCRFD actual provider")
    from face_pipeline_v2.models.wrappers.scrfd import SCRFDWrapper

    wrapper = SCRFDWrapper(
        model_path=Path(args.model),
        execution_provider=args.execution_provider,
    )
    wrapper.load()
    print(f"requested provider  : {args.execution_provider}")
    print(f"active providers    : {wrapper.active_providers}")
    if not wrapper.is_loaded:
        print("SCRFD failed to load.")
        return 1

    on_ppu_or_gpu = any(
        p != "CPUExecutionProvider" for p in wrapper.active_providers
    )
    print(
        "VERDICT             : "
        + ("ACCELERATED (PPU/CUDA)" if on_ppu_or_gpu else "CPU (not accelerated)")
    )

    # 4. Single-image inference timing
    _print_section("4. Single-image inference latency")
    import cv2
    import numpy as np

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read image: {args.image}; using synthetic 640x640.")
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    else:
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # warm-up
    for _ in range(3):
        wrapper.detect(img)

    t0 = time.perf_counter()
    for _ in range(args.num_runs):
        dets = wrapper.detect(img)
    dt_ms = (time.perf_counter() - t0) / args.num_runs * 1000.0

    print(f"image shape         : {img.shape}")
    print(f"runs                : {args.num_runs}")
    print(f"avg latency         : {dt_ms:.2f} ms/inference")
    print(f"detections          : {len(dets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
