"""EMICA model wrapper — minimal Stage12 encoder loading.

Per AISDD 4.5.1, Stages MUST NOT interact with the underlying model library
directly. This wrapper loads only the DECA and Expression encoders from
the EMICA checkpoint and provides expcode / jawpose / globalpose extraction
for Stage 12 jaw fusion.

Does NOT load: MICA encoder, FLAME decoder, renderer, shape model.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EMICAWrapper:
    """Minimal EMICA encoder wrapper for Stage12.

    Loads only the two ResNet50 encoders (DECA + Expression) from the
    EMICA checkpoint, skipping ~60 % of the original checkpoint size.
    """

    EXP_DIM = 100
    JAW_POSE_DIM = 3
    GLOBAL_POSE_DIM = 3
    TEX_DIM = 50
    CAM_DIM = 3
    LIGHT_DIM = 27
    EXPECTED_INPUT_SIZE = (224, 224)

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
    ) -> None:
        self._checkpoint_path: Path = Path(checkpoint_path)
        self._device: str = device
        self._deca_encoder: Any = None
        self._expression_encoder: Any = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._deca_encoder is not None and self._expression_encoder is not None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load DECA and Expression encoders from EMICA checkpoint.

        Raises:
            FileNotFoundError: If checkpoint not found.
            RuntimeError: If model loading fails.
        """
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"EMICA checkpoint not found: {self._checkpoint_path}"
            )

        logger.info(
            "Loading EMICA encoders: %s (device=%s)",
            self._checkpoint_path, self._device,
        )
        try:
            from omegaconf import OmegaConf
            from inferno.models.FaceReconstruction.FaceEncoder import (
                DecaEncoder,
                ExpressionEncoder,
            )

            deca_cfg = OmegaConf.create({
                "backbone": "ResNet50",
                "input_size": 224,
                "predicts": {
                    "texcode": self.TEX_DIM,
                    "jawpose": self.JAW_POSE_DIM,
                    "globalpose": self.GLOBAL_POSE_DIM,
                    "cam": self.CAM_DIM,
                    "lightcode": self.LIGHT_DIM,
                },
                "trainable": False,
                "last_layer_init_zero": False,
                "swin_type": None,
                "transformer": None,
            })
            expr_cfg = OmegaConf.create({
                "backbone": "ResNet50",
                "input_size": 224,
                "predicts": {"expcode": self.EXP_DIM},
                "trainable": False,
                "last_layer_init_zero": False,
                "swin_type": None,
                "transformer": None,
            })

            deca_encoder = DecaEncoder(deca_cfg)
            expression_encoder = ExpressionEncoder(expr_cfg)

            checkpoint = torch.load(
                str(self._checkpoint_path),
                map_location=self._device,
                weights_only=False,
            )
            state_dict = checkpoint["state_dict"]

            DECA_PREFIX = "face_encoder.mica_deca_encoder.deca_encoder.encoder."
            EXPR_PREFIX = "face_encoder.expression_encoder.encoder."

            deca_state = {}
            expr_state = {}
            for key, value in state_dict.items():
                if key.startswith(DECA_PREFIX):
                    new_key = key[len(DECA_PREFIX):]
                    deca_state[new_key] = value
                elif key.startswith(EXPR_PREFIX):
                    new_key = key[len(EXPR_PREFIX):]
                    expr_state[new_key] = value

            logger.info(
                "Extracted %d DECA weights, %d Expression weights "
                "(skipped MICA encoder / renderer / shape_model).",
                len(deca_state), len(expr_state),
            )

            deca_encoder.encoder.load_state_dict(deca_state, strict=True)
            expression_encoder.encoder.load_state_dict(expr_state, strict=True)

            deca_encoder = deca_encoder.to(self._device)
            deca_encoder.eval()
            expression_encoder = expression_encoder.to(self._device)
            expression_encoder.eval()

            self._deca_encoder = deca_encoder
            self._expression_encoder = expression_encoder

            logger.info("EMICA encoders loaded successfully (minimal mode).")
        except Exception as exc:
            logger.error("Failed to load EMICA encoders: %s", exc)
            raise RuntimeError(
                f"Failed to load EMICA encoders from {self._checkpoint_path}"
            ) from exc

    def unload(self) -> None:
        """Free GPU memory used by the encoders."""
        if self._deca_encoder is not None:
            del self._deca_encoder
            self._deca_encoder = None
        if self._expression_encoder is not None:
            del self._expression_encoder
            self._expression_encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run EMICA encoders on a batch of face images.

        Args:
            images: Normalized face images [B, 3, 224, 224].

        Returns:
            Dict with keys: expcode [B, 100], jawpose [B, 3],
            globalpose [B, 3].
        """
        if not self.is_loaded:
            raise RuntimeError("EMICA encoders not loaded. Call load() first.")
        batch = {"image": images.to(self._device)}
        with torch.no_grad():
            batch = self._deca_encoder.encode(batch)
            batch = self._expression_encoder.encode(batch)
        return {
            "expcode": batch["expcode"],
            "jawpose": batch["jawpose"],
            "globalpose": batch["globalpose"],
        }

    def extract_parameters(
        self, batch: Dict[str, torch.Tensor], frame_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """Extract numpy float tuples from the batch dict.

        Args:
            batch: Dict from encode().
            frame_indices: Frame indices per sample.

        Returns:
            List of dicts with keys: frame_index, expcode, jawpose,
            globalpose.
        """
        B = batch["expcode"].shape[0]
        results: List[Dict[str, Any]] = []
        for i in range(B):
            entry = {
                "frame_index": frame_indices[i],
                "expcode": tuple(batch["expcode"][i].cpu().numpy().tolist()),
                "jawpose": tuple(batch["jawpose"][i].cpu().numpy().tolist()),
                "globalpose": tuple(batch["globalpose"][i].cpu().numpy().tolist()),
            }
            results.append(entry)
        return results

    @staticmethod
    def preprocess_image(
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """Convert numpy image to normalized tensor for EMICA.

        Args:
            image: Input image (H, W, 3) in BGR, uint8 [0, 255].
            target_size: Desired output size (H, W).

        Returns:
            Tensor [1, 3, H, W] normalized to [0, 1] (float; matches the
            DECA/EMICA/SPECTRE training distribution — skimage ``warp()``
            output, NOT [-1, 1]).
        """
        import cv2
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size[::-1])
        tensor = torch.from_numpy(image.astype(np.float32))
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        tensor = tensor / 255.0           # [0, 255] -> [0, 1] float (matches DECA/EMICA/SPECTRE training distribution)
        tensor = tensor.unsqueeze(0)        # [C, H, W] -> [1, C, H, W]
        return tensor
