"""Runtime backend detection for Face_Pipeline_V2.

Auto-detects the available compute backend:
    - PPU: PPUExecutionProvider registered in onnxruntime.
    - CUDA: CUDAExecutionProvider registered in onnxruntime.
    - CPU: Always available as fallback.

Also checks PyTorch backend availability for SyncNet / HyperIQA.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List


class Backend(enum.Enum):
    """Compute backend identifier."""
    PPU = "ppu"
    CUDA = "cuda"
    CPU = "cpu"


@dataclass(frozen=True)
class RuntimeInfo:
    """Immutable snapshot of the current runtime environment."""

    onnx_providers: List[str] = field(default_factory=list)
    """All onnxruntime execution providers available."""

    onnx_active: str = "CPUExecutionProvider"
    """Currently selected ONNX provider."""

    torch_backend: str = "cpu"
    """PyTorch device backend: 'ppu', 'cuda', or 'cpu'."""

    torch_available: bool = False
    """Whether PyTorch can be imported."""

    backend: Backend = Backend.CPU
    """Active compute backend."""

    @property
    def has_gpu(self) -> bool:
        """Return True if any GPU backend is active."""
        return self.backend in (Backend.PPU, Backend.CUDA)


# ── Detection ───────────────────────────────────────────────────────────

# Known PPU provider names (vendor-dependent).
# NOTE: "AzureExecutionProvider" is a *CPU* execution provider (Azure's
# CPU-optimized EP), NOT a PPU/GPU backend — it must not be classified as PPU.
_PPU_PROVIDER_NAMES = frozenset({
    "PPUExecutionProvider",
    "HgExecutionProvider",
    "DcuExecutionProvider",
})

# Provider selection priority (best → worst)
_PROVIDER_PRIORITY = (
    *_PPU_PROVIDER_NAMES,  # any PPU variant
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


def _detect_onnx_providers() -> List[str]:
    """Return the list of onnxruntime providers available."""
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except ImportError:
        return []


def _select_onnx_provider(available: List[str]) -> str:
    """Select the best ONNX provider from available list.

    Priority: PPU → CUDA → CPU.
    """
    available_set = frozenset(available)
    for candidate in _PROVIDER_PRIORITY:
        if candidate in available_set:
            return candidate
    return "CPUExecutionProvider"


def _detect_torch() -> tuple:
    """Check PyTorch availability and backend.

    Returns:
        (torch_backend_name, is_available).
    """
    try:
        import torch
    except ImportError:
        return ("cpu", False)

    try:
        if torch.cuda.is_available():
            return ("cuda", True)
    except Exception:
        pass

    # PPU PyTorch builds may identify via a different mechanism
    try:
        if hasattr(torch, "_C") and hasattr(torch._C, "_ppu_is_available"):
            if torch._C._ppu_is_available():
                return ("ppu", True)
    except Exception:
        pass

    return ("cpu", True)


def _classify_backend(
    onnx_active: str,
    torch_backend: str,
) -> Backend:
    """Classify the overall backend from individual provider names."""
    active_lower = onnx_active.lower()
    if any(name in active_lower for name in ("ppu", "hg", "dcu")):
        return Backend.PPU
    if "cuda" in active_lower:
        return Backend.CUDA
    if torch_backend == "ppu":
        return Backend.PPU
    if torch_backend == "cuda":
        return Backend.CUDA
    return Backend.CPU


# ── Public API ──────────────────────────────────────────────────────────

_runtime_info: RuntimeInfo | None = None


def get_backend() -> RuntimeInfo:
    """Detect and return the runtime environment (cached).

    Returns:
        RuntimeInfo with onnx providers, torch status, and active backend.
    """
    global _runtime_info
    if _runtime_info is not None:
        return _runtime_info

    onnx_providers = _detect_onnx_providers()
    onnx_active = _select_onnx_provider(onnx_providers)
    torch_backend, torch_available = _detect_torch()
    backend = _classify_backend(onnx_active, torch_backend)

    _runtime_info = RuntimeInfo(
        onnx_providers=onnx_providers,
        onnx_active=onnx_active,
        torch_backend=torch_backend,
        torch_available=torch_available,
        backend=backend,
    )
    return _runtime_info
