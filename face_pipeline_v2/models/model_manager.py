"""Model Manager — Centralized model lifecycle management.

Per AISDD 4.5.1, Stages MUST NOT load models directly.
All model loading and caching is handled by ModelManager.
"""

from pathlib import Path
from typing import Any, Dict


class ModelManager:
    """统一模型管理器 — Manages model loading, caching, and device allocation.

    Rules (AISDD 4.7):
        1. Models are loaded on first use (lazy loading).
        2. Same model must not be loaded twice (caching).
        3. Device is controlled by Runtime Configuration.
        4. Model paths come from config.yaml.

    Usage:
        manager = ModelManager(device="cpu")
        model = manager.load_model("sixdrepnet", "models/6DRepNet")
        # Later:
        model = manager.get_model("sixdrepnet")
    """

    # Registry: model_name → (module_path, class_name)
    _REGISTRY: Dict[str, tuple] = {
        "sixdrepnet": (
            "face_pipeline_v2.models.wrappers.sixdrepnet",
            "SixDRepNetWrapper",
        ),
        "scrfd": (
            "face_pipeline_v2.models.wrappers.scrfd",
            "SCRFDWrapper",
        ),
        "syncnet": (
            "face_pipeline_v2.models.wrappers.syncnet",
            "SyncNetWrapper",
        ),
        "hyperiqa": (
            "face_pipeline_v2.models.wrappers.hyperiqa",
            "HyperIQAWrapper",
        ),
        "insightface_landmark": (
            "face_pipeline_v2.models.wrappers.insightface_landmark",
            "InsightFaceLandmarkWrapper",
        ),
        "emica": (
            "face_pipeline_v2.models.wrappers.emica",
            "EMICAWrapper",
        ),
        "spectre": (
            "face_pipeline_v2.models.wrappers.spectre",
            "SPECTREWrapper",
        ),
        "hubert": (
            "face_pipeline_v2.models.wrappers.hubert",
            "HuBERTWrapper",
        ),
    }

    def __init__(self, device: str = "cpu") -> None:
        """Initialize the ModelManager.

        Args:
            device: Target device for model inference (e.g. "cpu").
        """
        self._device: str = device
        self._model_cache: Dict[str, Any] = {}

    @property
    def device(self) -> str:
        """Current target device."""
        return self._device

    def load_model(self, model_name: str, model_path: str, **extra_kwargs: Any) -> Any:
        """Load a model by name and cache it.

        Args:
            model_name: Canonical model name (e.g. "sixdrepnet", "scrfd").
            model_path: Path to model weights directory.
            **extra_kwargs: Additional keyword arguments passed to the
                           wrapper constructor (e.g. batch_size, vshift
                           for SyncNetWrapper).

        Returns:
            Loaded model wrapper instance.

        Raises:
            KeyError: If model_name is not in the registry.
            FileNotFoundError: If model_path or weights file does not exist.
            RuntimeError: If model loading fails.
        """
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        if model_name not in self._REGISTRY:
            raise KeyError(
                f"Unknown model '{model_name}'. "
                f"Registered models: {list(self._REGISTRY.keys())}"
            )

        module_path, class_name = self._REGISTRY[model_name]

        # Dynamic import
        import importlib
        module = importlib.import_module(module_path)
        wrapper_cls = getattr(module, class_name)

        # Instantiate, load, cache
        # Detect parameter name: EMICA/SPECTRE wrappers use 'checkpoint_path',
        # while other wrappers (SyncNet, SixDRepNet, etc.) use 'model_path'.
        import inspect
        try:
            init_sig = inspect.signature(wrapper_cls.__init__)
        except (ValueError, TypeError):
            init_sig = None

        if init_sig is not None and "checkpoint_path" in init_sig.parameters:
            wrapper = wrapper_cls(
                checkpoint_path=Path(model_path), device=self._device, **extra_kwargs,
            )
        else:
            wrapper = wrapper_cls(
                model_path=Path(model_path), device=self._device, **extra_kwargs,
            )
        wrapper.load()
        self._model_cache[model_name] = wrapper
        return wrapper

    def get_model(self, model_name: str) -> Any:
        """Retrieve a previously loaded model from cache.

        Args:
            model_name: Canonical model name.

        Returns:
            Cached model object.

        Raises:
            KeyError: If the model has not been loaded yet.
        """
        if model_name not in self._model_cache:
            raise KeyError(
                f"Model '{model_name}' is not loaded. "
                f"Call load_model('{model_name}', path) first."
            )
        return self._model_cache[model_name]
