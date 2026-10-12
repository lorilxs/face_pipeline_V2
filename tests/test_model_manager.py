"""Tests for ModelManager — EMICA + SPECTRE registration and loading.

Verifies:
- ModelManager can load EMICAWrapper via load_model("emica", ...)
- ModelManager can load SPECTREWrapper via load_model("spectre", ...)
- ModelManager handles checkpoint_path vs model_path parameter naming
- Caching: get_model() returns the same instance
- Error handling: unknown model name raises KeyError
"""

from pathlib import Path

import pytest

from face_pipeline_v2.models.model_manager import ModelManager
from face_pipeline_v2.models.wrappers.emica import EMICAWrapper
from face_pipeline_v2.models.wrappers.spectre import SPECTREWrapper


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def emica_checkpoint_path() -> Path:
    """Path to the EMICA checkpoint."""
    p = Path(__file__).resolve().parents[1] / \
        "models" / "EMICA" / "EMICA-CVT_flame2020_notexture" / \
        "checkpoints" / "model-epoch=38-val" / "loss_total=3.88806534.ckpt"
    if not p.exists():
        pytest.skip(f"EMICA checkpoint not found: {p}")
    return p


@pytest.fixture(scope="module")
def spectre_checkpoint_path() -> Path:
    """Path to the SPECTRE/DECA checkpoint."""
    p = Path(__file__).resolve().parents[1] / \
        "third_party" / "inferno" / "spectre" / "spectre" / "data" / "deca_model.tar"
    if not p.exists():
        pytest.skip(f"SPECTRE checkpoint not found: {p}")
    return p


@pytest.fixture
def manager() -> ModelManager:
    """Fresh ModelManager on CPU for each test."""
    return ModelManager(device="cpu")


# ---------------------------------------------------------------------------
# Tests: EMICA loading
# ---------------------------------------------------------------------------

class TestModelManagerEMICA:
    """Tests for ModelManager.load_model("emica")."""

    def test_load_emica_returns_emica_wrapper(
        self, manager: ModelManager, emica_checkpoint_path: Path,
    ) -> None:
        """load_model("emica") returns an EMICAWrapper instance."""
        wrapper = manager.load_model("emica", str(emica_checkpoint_path))
        assert isinstance(wrapper, EMICAWrapper)
        assert wrapper.is_loaded

    def test_emica_get_model_after_load(
        self, manager: ModelManager, emica_checkpoint_path: Path,
    ) -> None:
        """get_model("emica") returns the same cached instance."""
        loaded = manager.load_model("emica", str(emica_checkpoint_path))
        cached = manager.get_model("emica")
        assert cached is loaded

    def test_emica_checkpoint_path_preserved(
        self, manager: ModelManager, emica_checkpoint_path: Path,
    ) -> None:
        """The wrapper's checkpoint_path attribute matches the given path."""
        wrapper = manager.load_model("emica", str(emica_checkpoint_path))
        assert wrapper.checkpoint_path == emica_checkpoint_path


# ---------------------------------------------------------------------------
# Tests: SPECTRE loading
# ---------------------------------------------------------------------------

class TestModelManagerSPECTRE:
    """Tests for ModelManager.load_model("spectre")."""

    def test_load_spectre_returns_spectre_wrapper(
        self, manager: ModelManager, spectre_checkpoint_path: Path,
    ) -> None:
        """load_model("spectre") returns a SPECTREWrapper instance."""
        wrapper = manager.load_model("spectre", str(spectre_checkpoint_path))
        assert isinstance(wrapper, SPECTREWrapper)

    def test_spectre_get_model_after_load(
        self, manager: ModelManager, spectre_checkpoint_path: Path,
    ) -> None:
        """get_model("spectre") returns the same cached instance."""
        loaded = manager.load_model("spectre", str(spectre_checkpoint_path))
        cached = manager.get_model("spectre")
        assert cached is loaded

    def test_spectre_checkpoint_path_preserved(
        self, manager: ModelManager, spectre_checkpoint_path: Path,
    ) -> None:
        """The wrapper's checkpoint_path attribute matches the given path."""
        wrapper = manager.load_model("spectre", str(spectre_checkpoint_path))
        assert wrapper.checkpoint_path == spectre_checkpoint_path


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------

class TestModelManagerErrors:
    """Tests for ModelManager error handling."""

    def test_unknown_model_raises_keyerror(self, manager: ModelManager) -> None:
        """load_model with an unregistered name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown model"):
            manager.load_model("nonexistent_model", "/some/path")

    def test_get_model_before_load_raises_keyerror(
        self, manager: ModelManager,
    ) -> None:
        """get_model for a never-loaded model raises KeyError."""
        with pytest.raises(KeyError, match="not loaded"):
            manager.get_model("emica")