"""InsightFace 106-point landmark model wrapper (ONNX).

Per AISDD, this wrapper encapsulates the 2d106det.onnx model for
high-precision facial landmark extraction. Stage 10 uses this
wrapper for per-face-crop inference.

Model: insightface 2d106det
  - Input:  (1, 3, 192, 192) float32
  - Output: (1, 212) float32  (106 points × 2, normalized coords)

Interface:
    wrapper = InsightFaceLandmarkWrapper(model_path=Path("models/insightface/2d106det.onnx"))
    wrapper.load()
    landmarks = wrapper.predict(face_crop)  # np.ndarray (H, W, 3) BGR → np.ndarray (106, 2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np


class InsightFaceLandmarkWrapper:
    """InsightFace 106-point facial landmark detector wrapper (ONNX).

    Encapsulates 2d106det.onnx model loading, preprocessing, inference,
    and coordinate de-normalization.

    Usage:
        wrapper = InsightFaceLandmarkWrapper(
            model_path=Path("models/insightface/2d106det.onnx")
        )
        wrapper.load()
        landmarks = wrapper.predict(face_crop_bgr)
        # landmarks.shape == (106, 2), coords in crop pixel space
    """

    # Model input size (fixed by architecture)
    INPUT_SIZE: Tuple[int, int] = (192, 192)

    # Number of landmarks
    NUM_LANDMARKS: int = 106

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        """Initialize the InsightFace landmark wrapper.

        Args:
            model_path: Path to the 2d106det.onnx file.
            device: Target device hint ("cpu", "cuda", "auto").
                    ONNX runtime handles provider selection automatically
                    via create_onnx_session.
        """
        self._model_path: Path = Path(model_path)
        self._device: str = device
        self._session: Any = None
        self._input_name: str = ""
        self._output_names: list = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        """Return the configured model path."""
        return self._model_path

    @property
    def device(self) -> str:
        """Return the target device hint."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded successfully."""
        return self._session is not None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the 2d106det ONNX model and inspect input/output metadata.

        Raises:
            FileNotFoundError: If the ONNX model file does not exist.
            RuntimeError: If ONNX session creation fails.
        """
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"2d106det ONNX model not found: {self._model_path.resolve()}"
            )

        from face_pipeline_v2.runtime.onnx_runtime import create_onnx_session

        self._session = create_onnx_session(self._model_path)

        # --- Inspect inputs ---
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError(
                "2d106det ONNX model has no inputs — corrupted model?"
            )
        self._input_name = inputs[0].name
        input_shape = inputs[0].shape
        # Expected: ['batch', 3, 192, 192]
        if len(input_shape) != 4:
            raise RuntimeError(
                f"Unexpected 2d106det input shape: {input_shape}. "
                f"Expected 4D tensor (N, 3, 192, 192)."
            )

        # --- Inspect outputs ---
        outputs = self._session.get_outputs()
        if not outputs:
            raise RuntimeError(
                "2d106det ONNX model has no outputs — corrupted model?"
            )
        self._output_names = [o.name for o in outputs]
        output_shape = outputs[0].shape
        # Expected: [1, 212] — 106 landmarks × (x, y)
        if output_shape[-1] != 212:
            raise RuntimeError(
                f"Unexpected 2d106det output shape: {output_shape}. "
                f"Expected last dim = 212 (106 points × 2)."
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, face_crop: np.ndarray) -> np.ndarray:
        """Extract 106 landmarks from a face crop image (BGR).

        Args:
            face_crop: BGR face crop image as numpy array of shape (H, W, 3).

        Returns:
            Numpy array of shape (106, 2) with (x, y) coordinates
            in the input crop's pixel space.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._session is None:
            raise RuntimeError(
                "2d106det model is not loaded. Call load() first."
            )

        crop_h, crop_w = face_crop.shape[:2]

        # Preprocess: resize → RGB → normalize → NCHW
        input_tensor = self._preprocess(face_crop)

        # Inference
        outputs = self._session.run(
            self._output_names, {self._input_name: input_tensor}
        )

        # outputs[0] has shape (1, 212) — normalized to [-1, 1] range
        raw = outputs[0]  # (1, 212)

        # De-normalize from [-1, 1] → pixel coords in 192×192 input space
        #   pixel = (normalized + 1) / 2 * 192
        landmarks_norm = raw.reshape(self.NUM_LANDMARKS, 2)  # (106, 2)

        # Scale from 192×192 model input space → crop pixel space
        #   crop_x = pixel_x / 192 * crop_w = (norm + 1) / 2 * crop_w
        landmarks_crop = np.zeros_like(landmarks_norm)
        landmarks_crop[:, 0] = (
            (landmarks_norm[:, 0] + 1.0) / 2.0 * crop_w
        )
        landmarks_crop[:, 1] = (
            (landmarks_norm[:, 1] + 1.0) / 2.0 * crop_h
        )

        return landmarks_crop

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a BGR face crop for 2d106det ONNX inference.

        Steps:
            1. Resize to 192×192.
            2. BGR → RGB.
            3. Keep raw [0, 255] values — the ONNX graph normalizes internally
               via Sub(127.5) + Mul(0.0078).
            4. Transpose to NCHW.

        Args:
            image: BGR image (H, W, 3).

        Returns:
            Input tensor of shape (1, 3, 192, 192) float32.
        """
        target_w, target_h = self.INPUT_SIZE

        # Resize to fixed input size
        resized = cv2.resize(image, (target_w, target_h))

        # BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Keep raw [0, 255] pixel values — the ONNX graph applies its own
        # normalization: (x - 127.5) * 0.0078  (Sub + Mul nodes at the input).
        # Dividing by 255 here would feed near-constant input and corrupt
        # the landmark output.
        blob = rgb.astype(np.float32)

        # HWC → NCHW
        input_tensor = np.expand_dims(
            np.transpose(blob, (2, 0, 1)), axis=0
        ).astype(np.float32)

        return input_tensor
