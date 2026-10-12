"""EMICA minimal loading test — single-image forward pass.

Verifies:
- EMICAWrapper can be instantiated and loaded
- encode() returns correct shapes for expcode, jawpose, globalpose
- preprocess_image() produces correct tensor shape and value range
- extract_parameters() returns proper dicts
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from face_pipeline_v2.models.wrappers.emica import EMICAWrapper


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
def emica_wrapper(emica_checkpoint_path: Path) -> EMICAWrapper:
    """Loaded EMICAWrapper on CPU."""
    wrapper = EMICAWrapper(checkpoint_path=emica_checkpoint_path, device="cpu")
    wrapper.load()
    return wrapper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEMICAWrapperInit:
    """Tests for EMICAWrapper construction and loading."""

    def test_instantiate(self, emica_checkpoint_path: Path) -> None:
        w = EMICAWrapper(checkpoint_path=emica_checkpoint_path, device="cpu")
        assert w.checkpoint_path == emica_checkpoint_path
        assert w.device == "cpu"

    def test_load(self, emica_wrapper: EMICAWrapper) -> None:
        assert emica_wrapper._deca_encoder is not None
        assert emica_wrapper._expression_encoder is not None

    def test_checkpoint_not_found(self) -> None:
        w = EMICAWrapper(checkpoint_path=Path("/nonexistent/ckpt.ckpt"))
        with pytest.raises(FileNotFoundError):
            w.load()

    def test_encode_without_load(self, emica_checkpoint_path: Path) -> None:
        w = EMICAWrapper(checkpoint_path=emica_checkpoint_path, device="cpu")
        dummy = torch.randn(1, 3, 224, 224)
        with pytest.raises(RuntimeError, match="not loaded"):
            w.encode(dummy)


class TestEMICAWrapperEncode:
    """Tests for EMICAWrapper encode() on synthetic images."""

    def test_single_image_forward(self, emica_wrapper: EMICAWrapper) -> None:
        """Single synthetic image forward pass returns correct shapes."""
        image = torch.randn(1, 3, 224, 224)
        output = emica_wrapper.encode(image)

        assert "expcode" in output
        assert "jawpose" in output
        assert "globalpose" in output
        assert output["expcode"].shape == (1, 100)
        assert output["jawpose"].shape == (1, 3)
        assert output["globalpose"].shape == (1, 3)
        assert output["expcode"].dtype == torch.float32
        assert output["jawpose"].dtype == torch.float32
        assert output["globalpose"].dtype == torch.float32

    def test_batch_forward(self, emica_wrapper: EMICAWrapper) -> None:
        """Batch of 4 synthetic images."""
        images = torch.randn(4, 3, 224, 224)
        output = emica_wrapper.encode(images)

        assert output["expcode"].shape == (4, 100)
        assert output["jawpose"].shape == (4, 3)
        assert output["globalpose"].shape == (4, 3)

    def test_output_values_are_finite(self, emica_wrapper: EMICAWrapper) -> None:
        """Output values should be finite floats."""
        image = torch.randn(1, 3, 224, 224)
        output = emica_wrapper.encode(image)

        assert torch.isfinite(output["expcode"]).all()
        assert torch.isfinite(output["jawpose"]).all()
        assert torch.isfinite(output["globalpose"]).all()

    def test_deterministic(self, emica_wrapper: EMICAWrapper) -> None:
        """Same input twice -> same output."""
        torch.manual_seed(42)
        image = torch.randn(1, 3, 224, 224)

        out1 = emica_wrapper.encode(image)
        out2 = emica_wrapper.encode(image)

        assert torch.allclose(out1["expcode"], out2["expcode"])
        assert torch.allclose(out1["jawpose"], out2["jawpose"])
        assert torch.allclose(out1["globalpose"], out2["globalpose"])


class TestEMICAWrapperPreprocess:
    """Tests for preprocess_image()."""

    def test_shape_and_range(self) -> None:
        """BGR uint8 -> [1, 3, H, W] normalized to [0, 1] (float; matches EMICA training distribution)."""
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        tensor = EMICAWrapper.preprocess_image(img, target_size=(224, 224))

        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32
        assert 0.0 <= tensor.min() <= tensor.max() <= 1.0

    def test_resize(self) -> None:
        """Non-224 input is resized."""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        tensor = EMICAWrapper.preprocess_image(img, target_size=(224, 224))
        assert tensor.shape == (1, 3, 224, 224)


class TestEMICAWrapperExtract:
    """Tests for extract_parameters()."""

    def test_extract_parameters(self, emica_wrapper: EMICAWrapper) -> None:
        """extract_parameters returns correct dicts."""
        image = torch.randn(2, 3, 224, 224)
        batch = emica_wrapper.encode(image)
        results = emica_wrapper.extract_parameters(batch, frame_indices=[7, 42])

        assert len(results) == 2
        assert results[0]["frame_index"] == 7
        assert results[1]["frame_index"] == 42
        assert len(results[0]["expcode"]) == 100
        assert len(results[0]["jawpose"]) == 3
        assert len(results[0]["globalpose"]) == 3
        assert isinstance(results[0]["expcode"], tuple)
        assert all(isinstance(v, float) for v in results[0]["expcode"])