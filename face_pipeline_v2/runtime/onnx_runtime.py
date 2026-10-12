"""Unified ONNX Runtime session creation.

Provides a single entry point for creating ONNX InferenceSession
objects with automatic provider selection and CPU fallback.

All model wrappers that use ONNX should call create_onnx_session()
instead of ort.InferenceSession() directly.

Execution provider selection is driven by an ``execution_provider``
preference (``auto`` / ``ppu`` / ``cuda`` / ``cpu``):

    auto  -> PPU EP (if registered) -> CUDA EP -> CPU
    ppu   -> PPU EP -> CUDA EP -> CPU   (on CUDA-compatible PPU, "ppu" == CUDA EP)
    cuda  -> CUDA EP -> CPU
    cpu   -> CPU only

The actual provider that onnxruntime ends up using is logged at INFO.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from face_pipeline_v2.runtime.backend import _PPU_PROVIDER_NAMES

logger = logging.getLogger(__name__)

# Valid values for the ``execution_provider`` preference.
VALID_EXECUTION_PROVIDERS = ("auto", "ppu", "cuda", "cpu")


def get_available_onnx_providers() -> List[str]:
    """Return the list of onnxruntime execution providers (fresh query)."""
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except ImportError:
        return []


def resolve_execution_providers(
    preference: str,
    available: Optional[List[str]] = None,
) -> List[str]:
    """Resolve an ordered provider chain for the given preference.

    Args:
        preference: One of ``auto`` / ``ppu`` / ``cuda`` / ``cpu``.
        available: Optional pre-fetched provider list. If None, queried live.

    Returns:
        Ordered provider names (deduplicated, CPU always last unless
        ``preference == "cpu"`` which returns CPU only).
    """
    pref = (preference or "auto").lower()
    if pref not in VALID_EXECUTION_PROVIDERS:
        logger.warning(
            "Unknown execution_provider '%s' (valid: %s); falling back to 'auto'.",
            preference, ", ".join(VALID_EXECUTION_PROVIDERS),
        )
        pref = "auto"

    avail = set(available if available is not None else get_available_onnx_providers())

    ppu_names = [p for p in _PPU_PROVIDER_NAMES if p in avail]
    cuda_available = "CUDAExecutionProvider" in avail

    if pref == "cpu":
        chain: List[str] = []
    elif pref == "cuda":
        chain = (["CUDAExecutionProvider"] if cuda_available else [])
    else:
        # "auto" and "ppu": on CUDA-compatible PPU hardware (e.g. T-Head
        # PPU-ZW810E) the PPU is reached via CUDAExecutionProvider, so both
        # native PPU EPs and CUDA EP are valid PPU backends.
        chain = ppu_names + (["CUDAExecutionProvider"] if cuda_available else [])

    chain.append("CPUExecutionProvider")

    # Deduplicate while preserving order.
    seen = set()
    ordered: List[str] = []
    for p in chain:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def create_onnx_session(
    model_path: str | Path,
    *,
    providers: List[str] | None = None,
    execution_provider: str = "auto",
    session_options: Any = None,
) -> Any:
    """Create an onnxruntime InferenceSession with automatic provider selection.

    Provider priority (when ``providers`` is not given explicitly):
        1. PPUExecutionProvider (or vendor equivalent) — if available
        2. CUDAExecutionProvider — if available (this is how CUDA-compatible
           PPUs such as T-Head PPU-ZW810E are reached)
        3. CPUExecutionProvider — always available as fallback

    Args:
        model_path: Path to the .onnx model file.
        providers: Explicit provider list (overrides auto-detection and
                   ``execution_provider``). If None, uses ``execution_provider``.
        execution_provider: Backend preference: ``auto`` / ``ppu`` / ``cuda`` / ``cpu``.
        session_options: Optional onnxruntime.SessionOptions.

    Returns:
        ort.InferenceSession configured with the best available provider.

    Raises:
        FileNotFoundError: If model_path does not exist.
        RuntimeError: If session creation fails with all providers.
    """
    import onnxruntime as ort

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    if providers is None:
        providers = resolve_execution_providers(execution_provider)
    else:
        # Deduplicate an explicit list while preserving order.
        seen = set()
        deduped: List[str] = []
        for p in providers:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        providers = deduped

    opts = session_options or ort.SessionOptions()

    last_error: Exception | None = None
    for provider in providers:
        try:
            sess = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=[provider],
            )
            logger.info(
                "ONNX session created for '%s': requested='%s' providers=%s active=%s",
                model_path.name,
                execution_provider,
                providers,
                sess.get_providers(),
            )
            return sess
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Provider %s failed for '%s': %s", provider, model_path.name, exc
            )
            continue

    raise RuntimeError(
        f"Failed to create ONNX session for {model_path.name}. "
        f"Tried providers {providers}. Last error: {last_error}"
    )
