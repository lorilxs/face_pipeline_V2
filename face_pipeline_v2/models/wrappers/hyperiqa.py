"""HyperIQA face quality assessment model wrapper.

Per AISDD 3.12, HyperIQA evaluates visual quality of face crop images.
Stage 8 uses this wrapper for per-frame quality inference.

Loading order:
    1. third_party/HyperIQA (primary — real PyTorch model).
    2. FileNotFoundError raised if checkpoint is missing (no silent fallback).

Interface:
    wrapper = HyperIQAWrapper(model_path=Path("models/HyperIQA"), device="cpu")
    wrapper.load()
    scores = wrapper.predict(images)  # List[float]  (0-100)
"""

from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

# ── Ensure third_party/HyperIQA is importable ─────────────────────────
import sys as _sys
_THIRD_PARTY_HYPERIQA = Path(__file__).resolve().parents[3] / "third_party" / "HyperIQA"
if str(_THIRD_PARTY_HYPERIQA) not in _sys.path:
    _sys.path.insert(0, str(_THIRD_PARTY_HYPERIQA))


class HyperIQAWrapper:
    """HyperIQA face quality assessor wrapper.

    Encapsulates HyperNet + TargetNet loading and per-image quality
    inference. Provides a unified predict() interface for Stage 8.
    """

    # Input preprocessing (matches official demo)
    _RESIZE = (384, 512)  # (height, width) — official: Resize((512, 384))
    _CROP_SIZE = 224
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Model architecture parameters
    _HYPERNET_ARGS = (16, 112, 224, 112, 56, 28, 14, 7)

    def __init__(self, model_path: Path, device: str = "auto") -> None:
        """Initialize HyperIQA wrapper.

        Args:
            model_path: Path to model weights directory
                        (containing the .pth checkpoint file).
            device: "auto" (detect CUDA), "cpu", or "cuda".
        """
        self._model_path: Path = Path(model_path)
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device: str = device
        self._hypernet: Any = None
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded."""
        return self._loaded

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the HyperIQA model from checkpoint.

        Searches for the checkpoint file in model_path directory.
        Accepts filenames: *.pth, *.pkl.

        Raises:
            FileNotFoundError: If no checkpoint file is found.
            RuntimeError: If model loading fails.
        """
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"HyperIQA model path does not exist: {self._model_path.resolve()}"
            )

        ckpt_path = self._find_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError(
                f"HyperIQA checkpoint not found in {self._model_path.resolve()}. "
                f"Expected .pth or .pkl file."
            )

        self._load_torch(ckpt_path)

    def _find_checkpoint(self) -> Path | None:
        """Locate the HyperIQA checkpoint file (prefer .pkl over .pth)."""
        for pattern in ["*.pkl", "*.pth"]:
            candidates = sorted(self._model_path.glob(pattern))
            if candidates:
                return candidates[0]
        return None

    def _load_torch(self, ckpt_path: Path) -> None:
        """Load HyperNet and apply checkpoint weights.

        Patches torch.model_zoo to use local resnet50 checkpoint
        for offline deployment (HyperNet init calls resnet50_backbone
        with pretrained=True which tries to download from pytorch.org).
        """
        import torch
        import torch.utils.model_zoo as model_zoo

        # Resolve local resnet50 path
        resnet50_local = self._model_path / "resnet50-19c8e357.pth"
        _original_load_url = model_zoo.load_url

        if resnet50_local.exists():
            def _local_load_url(url, **kw):
                # resnet50-19c8e357.pth is shipped as a legacy POSIX tar
                # (not a zipfile-serialized checkpoint). weights_only=True
                # would reject it, and the legacy tar loader cannot map
                # directly onto cuda; load to cpu (load_state_dict copies).
                # This is a trusted local model file.
                return torch.load(str(resnet50_local), map_location="cpu", weights_only=False)
            model_zoo.load_url = _local_load_url

        try:
            from models import HyperNet  # from third_party/HyperIQA
            self._hypernet = HyperNet(*self._HYPERNET_ARGS)
        finally:
            model_zoo.load_url = _original_load_url
        # HyperIQA checkpoint contains full model objects (not just state_dict),
        # so weights_only=True would fail. This is a trusted local model file.
        checkpoint = torch.load(str(ckpt_path), map_location=self._device, weights_only=False)

        # Resolve state dict (handle wrapper formats)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "params" in checkpoint:
                state_dict = checkpoint["params"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Strip "base_model." prefix if present (some training checkpoints
        # wrap HyperNet inside a parent module).
        cleaned = {}
        for key, value in state_dict.items():
            new_key = key.replace("base_model.", "", 1) if key.startswith("base_model.") else key
            cleaned[new_key] = value

        self._hypernet.load_state_dict(cleaned)
        self._hypernet.eval()
        if "cuda" in self._device:
            self._hypernet = self._hypernet.cuda()
        self._loaded = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, images: List[np.ndarray]) -> List[float]:
        """Predict quality scores (0-100) for face crop images (BGR).

        Args:
            images: List of BGR face crop images (np.ndarray, H×W×C).

        Returns:
            List of quality scores, one per input image, range 0-100.
            Higher = better quality.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if not self._loaded:
            raise RuntimeError("HyperIQA not loaded. Call load() first.")
        if not images:
            return []

        import torch
        from models import TargetNet  # from third_party/HyperIQA

        scores: List[float] = []
        with torch.no_grad():
            for img in images:
                # BGR → RGB, preprocess
                tensor = self._preprocess(img)
                if "cuda" in self._device:
                    tensor = tensor.cuda()

                # HyperNet forward
                paras = self._hypernet(tensor)

                # TargetNet forward → quality score
                model_target = TargetNet(paras)
                if "cuda" in self._device:
                    model_target = model_target.cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                pred = model_target(paras["target_in_vec"])
                scores.append(float(pred.item()))

        return scores

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> Any:
        """Convert a BGR face crop to a normalized PyTorch tensor.

        Follows official HyperIQA demo preprocessing:
            Resize(512, 384) → CenterCrop(224) → ToTensor → Normalize
        """
        import torch
        import torchvision.transforms as T

        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # numpy → PIL-compatible
        from PIL import Image
        pil_img = Image.fromarray(rgb)

        transform = T.Compose([
            T.Resize(self._RESIZE),
            T.CenterCrop(self._CROP_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(pil_img).unsqueeze(0)  # (1, 3, 224, 224)
        return tensor
