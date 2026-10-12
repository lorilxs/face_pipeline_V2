"""Tests for SPECTREWrapper — single-image forward pass.

Verifies:
- SPECTREWrapper can be instantiated and loaded
- encode() returns correct shapes for head_pose and jaw_pose
- preprocess_image() produces correct tensor shape and value range
- extract_parameters() returns proper dicts
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from face_pipeline_v2.models.wrappers.spectre import SPECTREWrapper


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spectre_checkpoint_path() -> Path:
    """Path to the SPECTRE/DECA checkpoint."""
    p = Path(__file__).resolve().parents[1] / \
        "third_party" / "inferno" / "spectre" / "spectre" / "data" / "deca_model.tar"
    if not p.exists():
        pytest.skip(f"SPECTRE checkpoint not found: {p}")
    return p


@pytest.fixture(scope="module")
def spectre_wrapper(spectre_checkpoint_path: Path) -> SPECTREWrapper:
    """Loaded SPECTREWrapper on CPU."""
    wrapper = SPECTREWrapper(checkpoint_path=spectre_checkpoint_path, device="cpu")
    wrapper.load()
    return wrapper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSPECTREWrapperInit:
    """Tests for SPECTREWrapper construction and loading."""

    def test_instantiate(self, spectre_checkpoint_path: Path) -> None:
        w = SPECTREWrapper(checkpoint_path=spectre_checkpoint_path, device="cpu")
        assert w.checkpoint_path == spectre_checkpoint_path
        assert w.device == "cpu"

    def test_load(self, spectre_wrapper: SPECTREWrapper) -> None:
        assert spectre_wrapper._model is not None

    def test_checkpoint_not_found(self) -> None:
        w = SPECTREWrapper(checkpoint_path=Path("/nonexistent/ckpt.tar"))
        with pytest.raises(FileNotFoundError):
            w.load()

    def test_encode_without_load(self, spectre_checkpoint_path: Path) -> None:
        w = SPECTREWrapper(checkpoint_path=spectre_checkpoint_path, device="cpu")
        dummy = torch.randn(1, 3, 224, 224)
        with pytest.raises(RuntimeError, match="not loaded"):
            w.encode(dummy)


class TestSPECTREWrapperEncode:
    """Tests for SPECTREWrapper encode() on synthetic images."""

    def test_single_image_forward(self, spectre_wrapper: SPECTREWrapper) -> None:
        """Single synthetic image forward pass returns correct shapes."""
        image = torch.randn(1, 3, 224, 224)  # synthetic, ~N(0,1)
        output = spectre_wrapper.encode(image)

        assert "expression" in output
        assert "head_pose" in output
        assert "jaw_pose" in output
        assert output["expression"].shape == (1, 100)
        assert output["head_pose"].shape == (1, 3)
        assert output["jaw_pose"].shape == (1, 3)
        assert output["expression"].dtype == torch.float32
        assert output["head_pose"].dtype == torch.float32
        assert output["jaw_pose"].dtype == torch.float32

    def test_expression_zero_padding(self, spectre_wrapper: SPECTREWrapper) -> None:
        """SPECTRE expression is 50-dim latent slice zero-padded to 100."""
        image = torch.randn(2, 3, 224, 224)
        output = spectre_wrapper.encode(image)

        exp = output["expression"]
        assert exp.shape == (2, 100)
        # Last 50 dims are the zero padding (SPECTRE raw n_exp == 50).
        assert torch.all(exp[:, 50:] == 0.0)
        # First 50 dims carry the actual latent slice (finite, may be nonzero).
        assert torch.isfinite(exp[:, :50]).all()

    def test_batch_forward(self, spectre_wrapper: SPECTREWrapper) -> None:
        """Batch of 4 synthetic images."""
        images = torch.randn(4, 3, 224, 224)
        output = spectre_wrapper.encode(images)

        assert output["expression"].shape == (4, 100)
        assert output["head_pose"].shape == (4, 3)
        assert output["jaw_pose"].shape == (4, 3)

    def test_output_values_are_finite(self, spectre_wrapper: SPECTREWrapper) -> None:
        """Output values should be finite floats."""
        image = torch.randn(1, 3, 224, 224)
        output = spectre_wrapper.encode(image)

        assert torch.isfinite(output["expression"]).all()
        assert torch.isfinite(output["head_pose"]).all()
        assert torch.isfinite(output["jaw_pose"]).all()

    def test_deterministic(self, spectre_wrapper: SPECTREWrapper) -> None:
        """Same input twice → same output."""
        torch.manual_seed(42)
        image = torch.randn(1, 3, 224, 224)

        out1 = spectre_wrapper.encode(image)
        out2 = spectre_wrapper.encode(image)

        assert torch.allclose(out1["expression"], out2["expression"])
        assert torch.allclose(out1["head_pose"], out2["head_pose"])
        assert torch.allclose(out1["jaw_pose"], out2["jaw_pose"])


class TestSPECTREWrapperPreprocess:
    """Tests for preprocess_image()."""

    def test_shape_and_range(self) -> None:
        """BGR uint8 → [1, 3, H, W] normalized to [0, 1] (float; matches SPECTRE training distribution)."""
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        tensor = SPECTREWrapper.preprocess_image(img, target_size=(224, 224))

        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32
        assert 0.0 <= tensor.min() <= tensor.max() <= 1.0

    def test_resize(self) -> None:
        """Non-224 input is resized."""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        tensor = SPECTREWrapper.preprocess_image(img, target_size=(224, 224))
        assert tensor.shape == (1, 3, 224, 224)


class TestSPECTREWrapperExtract:
    """Tests for extract_parameters()."""

    def test_extract_parameters(self, spectre_wrapper: SPECTREWrapper) -> None:
        """extract_parameters returns correct dicts."""
        image = torch.randn(2, 3, 224, 224)
        batch = spectre_wrapper.encode(image)
        results = spectre_wrapper.extract_parameters(batch, frame_indices=[7, 42])

        assert len(results) == 2
        assert results[0]["frame_index"] == 7
        assert results[1]["frame_index"] == 42
        assert len(results[0]["expression"]) == 100
        assert len(results[0]["head_pose"]) == 3
        assert len(results[0]["jaw_pose"]) == 3
        assert isinstance(results[0]["head_pose"], tuple)
        assert all(isinstance(v, float) for v in results[0]["head_pose"])