"""6DRepNet head pose estimation model wrapper (PyTorch).

Per AISDD 3.10, 6DRepNet estimates head pose (yaw, pitch, roll) from
face crop images. Stage 6 uses this wrapper for per-frame inference.

Uses the official sixdrepnet pip package for model architecture and
checkpoint loading.

Device selection:
    - "auto": CUDA if torch.cuda.is_available() else CPU.
    - "cpu" / "cuda": explicit.

Interface:
    wrapper = SixDRepNetWrapper(model_path=Path("models/6DRepNet"), device="auto")
    wrapper.load()
    results = wrapper.predict(images)  # List[dict] with yaw, pitch, roll
"""

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch


class SixDRepNetWrapper:
    """6DRepNet head pose estimator wrapper (PyTorch / official package)."""

    def __init__(self, model_path: Path, device: str = "auto") -> None:
        """Initialize 6DRepNet wrapper.

        Args:
            model_path: Path to model weights directory
                        (containing 6DRepNet_300W_LP_AFLW2000.pth).
            device: "auto" (detect CUDA), "cpu", or "cuda".
        """
        self._model_path: Path = Path(model_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device: str = device
        self._model: Any = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load 6DRepNet from checkpoint via official package.

        Raises:
            FileNotFoundError: If model_path or checkpoint missing.
        """
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"6DRepNet model path does not exist: {self._model_path.resolve()}"
            )

        ckpt_path = self._model_path / "6DRepNet_300W_LP_AFLW2000.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"6DRepNet checkpoint not found: {ckpt_path.resolve()}"
            )

        # Use local third_party copy for offline deployment
        # Import SixDRepNet_Detector directly (aliased as SixDRepNet in __init__)
        from third_party.sixdrepnet.regressor import SixDRepNet_Detector

        gpu_id = 0 if self._device == "cuda" else -1
        self._model = SixDRepNet_Detector(gpu_id=gpu_id, dict_path=str(ckpt_path))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        """Estimate head pose for face crop images (BGR).

        Args:
            images: List of BGR face crop images (H×W×C).

        Returns:
            List of dicts: {yaw, pitch, roll, confidence}.
        """
        if self._model is None:
            raise RuntimeError("6DRepNet not loaded. Call load() first.")
        if not images:
            return []

        results: List[Dict[str, float]] = []
        for img in images:
            yaw, pitch, roll = self._model.predict(img)
            results.append({
                "yaw": float(yaw.item()) if hasattr(yaw, "item") else float(yaw),
                "pitch": float(pitch.item()) if hasattr(pitch, "item") else float(pitch),
                "roll": float(roll.item()) if hasattr(roll, "item") else float(roll),
                "confidence": 1.0,
            })
        return results

    def predict_batch(
        self, images: List[np.ndarray], batch_size: int = 32,
    ) -> List[Dict[str, float]]:
        """Estimate head pose for face crop images using batched inference.

        Processes images in batches for GPU efficiency, using the same
        preprocessing and decoding logic as the single-image predict().

        Args:
            images: List of BGR face crop images (H×W×C).
            batch_size: Number of images per forward pass (default 32).

        Returns:
            List of dicts: {yaw, pitch, roll, confidence}.
            Results are identical to predict() within floating-point tolerance.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("6DRepNet not loaded. Call load() first.")
        if not images:
            return []

        try:
            return self._predict_batch_impl(images, batch_size)
        except torch.cuda.OutOfMemoryError:
            if batch_size > 8:
                import logging
                _logger = logging.getLogger(__name__)
                _logger.warning(
                    f"6DRepNet batch OOM at batch_size={batch_size}, "
                    f"retrying with batch_size={batch_size // 2}"
                )
                torch.cuda.empty_cache()
                return self.predict_batch(images, batch_size // 2)
            raise

    def _predict_batch_impl(
        self, images: List[np.ndarray], batch_size: int,
    ) -> List[Dict[str, float]]:
        """Internal batched inference implementation."""
        from PIL import Image

        # Reuse the exact same transforms as SixDRepNet_Detector.predict()
        # (Resize(224), CenterCrop(224), ToTensor, Normalize)
        transform = self._model.transformations

        # Import the same decoding functions used by the single-image path
        from third_party.sixdrepnet import utils

        results: List[Dict[str, float]] = []

        for start in range(0, len(images), batch_size):
            batch_imgs = images[start:start + batch_size]

            # ---- Preprocess: BGR→RGB→PIL→Transform (same as single-image) ----
            tensors = []
            for img in batch_imgs:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                tensor = transform(pil)  # [3, 224, 224]
                tensors.append(tensor)

            batch = torch.stack(tensors)  # [B, 3, 224, 224]

            if self._model.gpu != -1:
                batch = batch.cuda(self._model.gpu)

            # ---- Forward: single call to model (no per-image loop) ----
            with torch.no_grad():
                # self._model.model is SixDRepNet (nn.Module)
                # forward returns [B, 3, 3] rotation matrices
                pred = self._model.model(batch)

                # Reuse the exact same decoding as single-image path:
                #   compute_euler_angles_from_rotation_matrices → [B, 3] radians
                #   * 180/π → degrees
                #   order: (pitch, yaw, roll)
                euler = (
                    utils.compute_euler_angles_from_rotation_matrices(pred)
                    * 180.0 / np.pi
                )

            euler_np = euler.cpu().numpy()  # [B, 3] — (pitch, yaw, roll) from model
            for i in range(euler_np.shape[0]):
                # Match the single-image predict() swap:
                #   self._model.predict(img) returns (pitch, yaw, roll)
                #   wrapper predicts swaps them: yaw=pitch, pitch=yaw
                pitch_model = float(euler_np[i, 0])
                yaw_model = float(euler_np[i, 1])
                roll_model = float(euler_np[i, 2])
                results.append({
                    "yaw": pitch_model,
                    "pitch": yaw_model,
                    "roll": roll_model,
                    "confidence": 1.0,
                })

        return results
