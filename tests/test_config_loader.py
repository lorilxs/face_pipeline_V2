"""Tests for ConfigLoader.

Verifies YAML loading, path management, and validation.
"""

from pathlib import Path

import pytest

from face_pipeline_v2.config.config_loader import ConfigLoader


class TestConfigLoader:
    """Tests for the ConfigLoader class."""

    def test_load_valid_config(self) -> None:
        """Verify config.yaml can be loaded and parsed."""
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        loader = ConfigLoader(config_path=config_path)
        config = loader.load()
        assert config is not None
        assert isinstance(config, dict)

    def test_validate_passes_with_all_keys(self) -> None:
        """Verify validate() passes when all required keys are present."""
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        loader = ConfigLoader(config_path=config_path)
        config = loader.load()
        # Should not raise
        loader.validate(config)

    def test_validate_raises_on_missing_keys(self) -> None:
        """Verify validate() raises KeyError for missing required keys."""
        loader = ConfigLoader(config_path=Path("/nonexistent/config.yaml"))
        incomplete_config = {"project": {}, "video": {}}
        with pytest.raises(KeyError, match="Missing required configuration categories"):
            loader.validate(incomplete_config)

    def test_load_nonexistent_file_raises(self) -> None:
        """Verify FileNotFoundError raised for missing config file."""
        loader = ConfigLoader(config_path=Path("/nonexistent/path/config.yaml"))
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_config_path_resolves(self) -> None:
        """Verify config_path property returns resolved absolute path."""
        loader = ConfigLoader(config_path=Path("configs/config.yaml"))
        resolved = loader.config_path
        assert resolved.is_absolute()
