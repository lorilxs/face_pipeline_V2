"""Unified runtime backend management for Face_Pipeline_V2.

Provides:
    - Backend auto-detection (CPU, PPU, CUDA).
    - Unified ONNX Runtime session creation with automatic
      provider selection and graceful fallback.

Usage:
    from face_pipeline_v2.runtime import get_backend, create_onnx_session

    backend = get_backend()
    session = create_onnx_session("models/SCRFD/scrfd_10g_bnkps.onnx")
"""

from face_pipeline_v2.runtime.backend import Backend, get_backend
from face_pipeline_v2.runtime.onnx_runtime import create_onnx_session

__all__ = ["Backend", "get_backend", "create_onnx_session"]
