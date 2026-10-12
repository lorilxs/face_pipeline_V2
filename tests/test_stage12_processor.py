"""End-to-end test for Stage 12 processor with mocked EMICA + SPECTRE.

Uses MockEncoder and MockCLIPPreprocessor to simulate real model behavior
without loading actual model weights.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest
import torch

from face_pipeline_v2.data.models import AlignedFaceInfo, FLAMEFeature
from face_pipeline_v2.stages.stage_12_flame.processor import (
    extract_flame_parameters,
    validate_sources,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockEncoder:
    """Mock encoder that returns deterministic outputs per batch element."""

    def __init__(
        self,
        jaw_vals: List[float],
        global_vals: List[float],
        exp_vals: List[float] = None,
        model_type: str = "emica",
    ) -> None:
        self._jaw = jaw_vals
        self._global = global_vals
        self._exp = exp_vals or [0.0] * 100
        self._type = model_type

    def __call__(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = images.shape[0]
        if self._type == "emica":
            return {
                "expcode": torch.tensor(
                    [self._exp for _ in range(B)], dtype=torch.float32,
                ),
                "jawpose": torch.tensor(
                    [self._jaw for _ in range(B)], dtype=torch.float32,
                ),
                "globalpose": torch.tensor(
                    [self._global for _ in range(B)], dtype=torch.float32,
                ),
            }
        else:
            return {
                "expression": torch.tensor(
                    [self._exp for _ in range(B)], dtype=torch.float32,
                ),
                "jaw_pose": torch.tensor(
                    [self._jaw for _ in range(B)], dtype=torch.float32,
                ),
                "head_pose": torch.tensor(
                    [self._global for _ in range(B)], dtype=torch.float32,
                ),
            }


class MockEMICAWrapper:
    """Mock EMICA wrapper with preprocess_image and encode."""

    def __init__(
        self, jaw: List[float], globalpose: List[float],
        exp: List[float] = None,
    ) -> None:
        self._encoder = MockEncoder(jaw, globalpose, exp, model_type="emica")

    def preprocess_image(
        self, img: np.ndarray, target_size: Tuple[int, int] = None,
    ) -> torch.Tensor:
        return torch.randn(1, 3, 224, 224)

    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._encoder(images)


class MockSPECTREWrapper:
    """Mock SPECTRE wrapper with encode."""

    def __init__(
        self, jaw: List[float], head_pose: List[float] = None,
        exp: List[float] = None,
    ) -> None:
        self._encoder = MockEncoder(
            jaw, head_pose or [0.0, 0.0, 0.0], exp, model_type="spectre",
        )

    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._encoder(images)


def _create_dummy_face(path: Path, size: Tuple[int, int] = (224, 224)) -> None:
    """Write a dummy BGR image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
# ===========================================================================
# Tests
# ===========================================================================

class TestProcessorEndToEnd:
    """End-to-end processor tests with mocked models."""

    def test_single_face_jaw_from_emica(self) -> None:
        """Single face → jaw from EMICA (final scheme), expcode/globalpose from EMICA."""
        emica = MockEMICAWrapper(
            jaw=[1.0, 2.0, 3.0],
            globalpose=[0.1, 0.2, 0.3],
            exp=[0.01] * 100,
        )
        spectre = MockSPECTREWrapper(jaw=[3.0, 4.0, 5.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            face_path = Path(tmpdir) / "face_0.png"
            _create_dummy_face(face_path)
            faces = [
                AlignedFaceInfo(
                    clip_id="test",
                    frame_index=0,
                    face_id="face_0",
                    timestamp=0.0,
                    aligned_face_path=face_path,
                    affine_matrix=(1., 0., 0., 0., 1., 0.),
                    original_image_size=(224, 224),
                    aligned_resolution=(224, 224),
                ),
            ]
            results = extract_flame_parameters(
                aligned_faces=faces,
                emica_model=emica,
                spectre_model=spectre,
                jaw_fusion_lambda=0.5,
                batch_size=4,
                device="cpu",
            )

        assert len(results) == 1
        f = results[0]
        np.testing.assert_array_almost_equal(
            np.array(f.jawpose), np.array([1.0, 2.0, 3.0]),
        )
        np.testing.assert_array_almost_equal(
            np.array(f.globalpose), np.array([0.1, 0.2, 0.3]),
        )
        assert len(f.expcode) == 100
        assert all(abs(v - 0.01) < 1e-6 for v in f.expcode[:10])
        assert len(f.shapecode) == 300
        assert all(v == 0.0 for v in f.shapecode)
        assert len(f.eye_pose) == 6
        assert all(v == 0.0 for v in f.eye_pose)
        np.testing.assert_array_almost_equal(
            np.array(f.spectre_jawpose), np.array([3.0, 4.0, 5.0]),
        )
        np.testing.assert_array_almost_equal(
            np.array(f.emica_jawpose), np.array([1.0, 2.0, 3.0]),
        )

    def test_multi_face_batch(self) -> None:
        """3 faces in one batch → 3 results."""
        emica = MockEMICAWrapper(
            jaw=[1.0, 0.0, 0.0],
            globalpose=[0.5, 0.5, 0.5],
            exp=[0.0] * 100,
        )
        spectre = MockSPECTREWrapper(jaw=[0.0, 1.0, 0.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            faces = []
            for i in range(3):
                path = Path(tmpdir) / f"face_{i}.png"
                _create_dummy_face(path)
                faces.append(
                    AlignedFaceInfo(
                        clip_id="test",
                        frame_index=i,
                        face_id=f"face_{i}",
                        timestamp=float(i),
                        aligned_face_path=path,
                        affine_matrix=(1., 0., 0., 0., 1., 0.),
                        original_image_size=(224, 224),
                        aligned_resolution=(224, 224),
                    ),
                )
            results = extract_flame_parameters(
                aligned_faces=faces,
                emica_model=emica,
                spectre_model=spectre,
                jaw_fusion_lambda=0.6,
                batch_size=2,
                device="cpu",
            )

        assert len(results) == 3
        for i, f in enumerate(results):
            assert f.frame_index == i
            assert len(f.expcode) == 100
            assert len(f.jawpose) == 3
            assert len(f.globalpose) == 3
            assert f.spectre_jawpose is not None

    def test_empty_input(self) -> None:
        """Empty list → empty list."""
        emica = MockEMICAWrapper(jaw=[0., 0., 0.], globalpose=[0., 0., 0.])
        spectre = MockSPECTREWrapper(jaw=[0., 0., 0.])
        results = extract_flame_parameters(
            aligned_faces=[],
            emica_model=emica,
            spectre_model=spectre,
            jaw_fusion_lambda=0.6,
            batch_size=4,
            device="cpu",
        )
    def test_jaw_emica_regardless_of_lambda(self) -> None:
        """Final scheme: jaw = pure EMICA even at λ=0.0 (fusion deprecated)."""
        emica = MockEMICAWrapper(
            jaw=[0.5, -0.3, 0.1],
            globalpose=[0.0, 0.0, 0.0],
            exp=[0.0] * 100,
        )
        spectre = MockSPECTREWrapper(jaw=[9.0, 9.0, 9.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "face.png"
            _create_dummy_face(path)
            faces = [
                AlignedFaceInfo(
                    clip_id="test",
                    frame_index=0,
                    face_id="face_0",
                    timestamp=0.0,
                    aligned_face_path=path,
                    affine_matrix=(1., 0., 0., 0., 1., 0.),
                    original_image_size=(224, 224),
                    aligned_resolution=(224, 224),
                ),
            ]
            results = extract_flame_parameters(
                aligned_faces=faces,
                emica_model=emica,
                spectre_model=spectre,
                jaw_fusion_lambda=0.0,
                batch_size=4,
                device="cpu",
            )

        assert len(results) == 1
        np.testing.assert_array_almost_equal(
            np.array(results[0].jawpose), np.array([0.5, -0.3, 0.1]),
        )
        np.testing.assert_array_almost_equal(
            np.array(results[0].emica_jawpose), np.array([0.5, -0.3, 0.1]),
        )

    def test_lambda_one_pure_emica_jaw(self) -> None:
        """λ=1.0 → jaw = pure EMICA (final scheme, λ ignored)."""
        emica = MockEMICAWrapper(
            jaw=[9.0, 9.0, 9.0],
            globalpose=[0.0, 0.0, 0.0],
            exp=[0.0] * 100,
        )
        spectre = MockSPECTREWrapper(jaw=[-0.5, 0.3, -0.1])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "face.png"
            _create_dummy_face(path)
            faces = [
                AlignedFaceInfo(
                    clip_id="test",
                    frame_index=0,
                    face_id="face_0",
                    timestamp=0.0,
                    aligned_face_path=path,
                    affine_matrix=(1., 0., 0., 0., 1., 0.),
                    original_image_size=(224, 224),
                    aligned_resolution=(224, 224),
                ),
            ]
            results = extract_flame_parameters(
                aligned_faces=faces,
                emica_model=emica,
                spectre_model=spectre,
                jaw_fusion_lambda=1.0,
                batch_size=4,
                device="cpu",
            )

        assert len(results) == 1
        np.testing.assert_array_almost_equal(
            np.array(results[0].jawpose), np.array([9.0, 9.0, 9.0]),
        )

    def test_head_pose_from_emica_not_spectre(self) -> None:
        """Globalpose comes from EMICA, SPECTRE head_pose is ignored."""
        emica = MockEMICAWrapper(
            jaw=[0.0, 0.0, 0.0],
            globalpose=[0.1, 0.2, 0.3],
            exp=[0.0] * 100,
        )
        spectre = MockSPECTREWrapper(
            jaw=[0.0, 0.0, 0.0],
            head_pose=[9.9, 9.9, 9.9],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "face.png"
            _create_dummy_face(path)
            faces = [
                AlignedFaceInfo(
                    clip_id="test",
                    frame_index=0,
                    face_id="face_0",
                    timestamp=0.0,
                    aligned_face_path=path,
                    affine_matrix=(1., 0., 0., 0., 1., 0.),
                    original_image_size=(224, 224),
                    aligned_resolution=(224, 224),
                ),
            ]
            results = extract_flame_parameters(
                aligned_faces=faces,
                emica_model=emica,
                spectre_model=spectre,
                jaw_fusion_lambda=0.6,
                batch_size=4,
                device="cpu",
            )

        assert len(results) == 1
        np.testing.assert_array_almost_equal(
            np.array(results[0].globalpose), np.array([0.1, 0.2, 0.3]),
        )


# ===========================================================================
# Stage12 EMICA vs SPECTRE full-parameter source ablation tests (AISDD)
# ===========================================================================

class TestStage12SourceSelection:
    """来源选择 — Experiment A (全 EMICA) / Experiment B (全 SPECTRE) / 禁混源。"""

    def _single_face(self, tmpdir: str) -> List[AlignedFaceInfo]:
        path = Path(tmpdir) / "face.png"
        _create_dummy_face(path)
        return [AlignedFaceInfo(
            clip_id="test",
            frame_index=0,
            face_id="face_0",
            timestamp=0.0,
            aligned_face_path=path,
            affine_matrix=(1., 0., 0., 0., 1., 0.),
            original_image_size=(224, 224),
            aligned_resolution=(224, 224),
        )]

    def test_emica_default_unchanged(self) -> None:
        """Production default (all emica) — jaw/head/expression from EMICA."""
        emica = MockEMICAWrapper(
            jaw=[5.0, 6.0, 7.0],
            globalpose=[0.1, 0.2, 0.3],
            exp=[0.5] * 100,
        )
        spectre = MockSPECTREWrapper(
            jaw=[9.0, 9.0, 9.0], head_pose=[8.0, 8.0, 8.0],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            faces = self._single_face(tmpdir)
            results = extract_flame_parameters(
                aligned_faces=faces,
                emica_model=emica,
                spectre_model=spectre,
                jaw_fusion_lambda=1.0,
                batch_size=4,
                device="cpu",
            )

        assert len(results) == 1
        f = results[0]
        np.testing.assert_array_almost_equal(
            np.array(f.jawpose), np.array([5.0, 6.0, 7.0]),
        )
        np.testing.assert_array_almost_equal(
            np.array(f.globalpose), np.array([0.1, 0.2, 0.3]),
        )
        assert len(f.expcode) == 100
        assert all(abs(v - 0.5) < 1e-6 for v in f.expcode)

    def test_spectre_full_source_mode(self) -> None:
        """Experiment B: all three sources = spectre (50-dim exp zero-padded)."""
        spectre_exp = [round(0.01 * i, 6) for i in range(1, 51)]  # 50-dim
        emica = MockEMICAWrapper(
            jaw=[0.0, 0.0, 0.0],
            globalpose=[0.0, 0.0, 0.0],
            exp=[0.0] * 100,
        )
        spectre = MockSPECTREWrapper(
            jaw=[1.0, 2.0, 3.0],
            head_pose=[0.1, 0.2, 0.3],
            exp=spectre_exp,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            faces = self._single_face(tmpdir)
            results = extract_flame_parameters(
                aligned_faces=faces,
                emica_model=emica,
                spectre_model=spectre,
                jaw_fusion_lambda=0.6,
                batch_size=4,
                device="cpu",
                expression_source="spectre",
                head_source="spectre",
                jaw_source="spectre",
            )

        assert len(results) == 1
        f = results[0]
        # expression: 50-dim SPECTRE latent slice zero-padded to 100
        np.testing.assert_array_almost_equal(
            np.array(f.expcode[:50]), np.array(spectre_exp),
        )
        np.testing.assert_array_equal(np.array(f.expcode[50:]), np.zeros(50))
        np.testing.assert_array_almost_equal(
            np.array(f.jawpose), np.array([1.0, 2.0, 3.0]),
        )
        np.testing.assert_array_almost_equal(
            np.array(f.globalpose), np.array([0.1, 0.2, 0.3]),
        )
        # audit fields always preserved (raw per-model values)
        np.testing.assert_array_almost_equal(
            np.array(f.emica_jawpose), np.array([0.0, 0.0, 0.0]),
        )
        np.testing.assert_array_almost_equal(
            np.array(f.spectre_jawpose), np.array([1.0, 2.0, 3.0]),
        )

    def test_mixed_source_forbidden(self) -> None:
        """Mixed source (expression=spectre, rest=emica) raises ValueError."""
        emica = MockEMICAWrapper(jaw=[0., 0., 0.], globalpose=[0., 0., 0.])
        spectre = MockSPECTREWrapper(jaw=[0., 0., 0.])

        with tempfile.TemporaryDirectory() as tmpdir:
            faces = self._single_face(tmpdir)
            with pytest.raises(ValueError, match="Mixed"):
                extract_flame_parameters(
                    aligned_faces=faces,
                    emica_model=emica,
                    spectre_model=spectre,
                    expression_source="spectre",
                    head_source="emica",
                    jaw_source="emica",
                    batch_size=4,
                    device="cpu",
                )

    def test_invalid_source_raises(self) -> None:
        """Unknown source value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            validate_sources("emica", "emica", "deca")
