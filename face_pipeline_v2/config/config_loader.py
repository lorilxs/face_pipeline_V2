"""Configuration loader for Face_Pipeline_V2.

All Pipeline parameters MUST come from configs/config.yaml.
Direct hardcoding of threshold/path/model_name/batch_size/device in
Stage code is forbidden per AISDD 4.3.1.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """统一配置加载器 — Loads and validates configs/config.yaml.

    Usage:
        loader = ConfigLoader(config_path=Path("configs/config.yaml"))
        config = loader.load()
        # Access values: config["video"]["target_fps"]
        loader.validate(config)
    """

    _REQUIRED_TOP_KEYS: tuple = (
        "project",
        "video",
        "scene",
        "face",
        "pose",
        "sync",
        "quality",
        "landmark",
        "alignment",
        "flame",
        "gaze",
        "audio",
        "runtime",
        "storage",
    )

    # Orchestration-level pipeline mode switch (Stage6-9 optional in
    # dataset mode). Backward compatible: key optional, defaults to
    # "open_domain" when absent.
    _ALLOWED_PIPELINE_MODES: tuple = ("open_domain", "dataset")

    def __init__(self, config_path: Path) -> None:
        """Initialize the ConfigLoader.

        Args:
            config_path: Absolute or relative path to config.yaml.
        """
        self._config_path: Path = Path(config_path)

    @property
    def config_path(self) -> Path:
        """Return the resolved config file path."""
        return self._config_path.resolve()

    def load(self) -> Dict[str, Any]:
        """Load configuration from the YAML file.

        Returns:
            Parsed configuration as a nested dictionary.

        Raises:
            FileNotFoundError: If config.yaml does not exist at config_path.
            yaml.YAMLError: If the YAML file is malformed.
        """
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_path.resolve()}"
            )

        with open(self._config_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        if config is None:
            raise ValueError(
                f"Configuration file is empty: {self._config_path.resolve()}"
            )

        return config

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate the loaded configuration structure.

        Checks that all required top-level category keys are present
        per AISDD 4.3.2 Configuration Hierarchy.

        Args:
            config: The loaded configuration dictionary.

        Raises:
            KeyError: If a required top-level key is missing.
        """
        missing_keys = [
            key for key in self._REQUIRED_TOP_KEYS if key not in config
        ]
        if missing_keys:
            raise KeyError(
                f"Missing required configuration categories: {missing_keys}. "
                f"Expected keys: {list(self._REQUIRED_TOP_KEYS)}"
            )
        # Optional orchestration-level switch. Only validated when present
        # so that existing configs without the key remain fully compatible.
        if "pipeline_mode" in config:
            mode = config["pipeline_mode"]
            if mode not in self._ALLOWED_PIPELINE_MODES:
                raise ValueError(
                    f"Invalid pipeline_mode={mode!r}. Allowed values: "
                    f"{list(self._ALLOWED_PIPELINE_MODES)}."
                )
