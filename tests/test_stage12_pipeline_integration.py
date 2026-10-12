"""Stage12 Pipeline Integration Tests.

Tests the Stage12 → Pipeline integration path:
  Test1: Stage11 mock output → Stage12 FLAMEFeature generation (direct mode)
  Test2: Output dimension validation
  Test3: Stage12 input/output validation
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

from face_pipeline_v2.data.models import AlignedFaceInfo, FLAMEFeature
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_12_flame.processor import (
    extract_flame_parameters,
)
from face_pipeline_v2.stages.stage_12_flame.stage import Stage12FLAME

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockEncoder:
    def __init__(self, jaw_vals, global_vals, exp_vals=None, model_type="emica"):
        self._jaw = jaw_vals
        self._global = global_vals
        self._exp = exp_vals or [0.0] * 100
        self._type = model_type

    def __call__(self, images):
        B = images.shape[0]
        if self._type == "emica":
            return {
                "expcode": torch.tensor([self._exp] * B, dtype=torch.float32),
                "jawpose": torch.tensor([self._jaw] * B, dtype=torch.float32),
                "globalpose": torch.tensor([self._global] * B, dtype=torch.float32),
            }
        return {
            "expression": torch.tensor([self._exp] * B, dtype=torch.float32),
            "jaw_pose": torch.tensor([self._jaw] * B, dtype=torch.float32),
            "head_pose": torch.tensor([self._global] * B, dtype=torch.float32),
        }


class MockEMICAWrapper:
    def __init__(self, jaw, globalpose, exp=None):
        self._encoder = MockEncoder(jaw, globalpose, exp, "emica")
    def preprocess_image(self, img, target_size=None):
        return torch.randn(1, 3, 224, 224)
    def encode(self, images):
        return self._encoder(images)


class MockSPECTREWrapper:
    def __init__(self, jaw, head_pose=None):
        self._encoder = MockEncoder(jaw, head_pose or [0.,0.,0.], None, "spectre")
    def encode(self, images):
        return self._encoder(images)


def _create_dummy_face(path, size=(224, 224)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_context(override=None):
    cfg = {"stage12": {"enabled": True, "jaw_fusion_lambda": 0.6,
                       "batch_size": 8, "runtime_mode": "direct"}}
    if override:
        cfg["stage12"].update(override)
    return PipelineContext(config=cfg, logger=MagicMock(),
                           model_manager=MagicMock(),
                           runtime_info={"clip_id": "test_clip"})
# ---------------------------------------------------------------------------
# Test 1: Dimension checks
# ---------------------------------------------------------------------------

class TestStage12Dimensions:
    """All FLAMEFeature fields have correct dimensions."""

    def test_expression_dim(self):
        emica = MockEMICAWrapper([0.1,0.2,0.3], [0.,0.,0.], list(range(100)))
        spectre = MockSPECTREWrapper([0.,0.,0.])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"; _create_dummy_face(p)
            faces = [AlignedFaceInfo("t", 0, "f0", 0.0, p,
                     (1.,0.,0.,0.,1.,0.), (224,224), (224,224))]
            r = extract_flame_parameters(faces, emica, spectre, 0.6, 4, "cpu")
        assert len(r[0].expcode) == 100

    def test_jaw_pose_dim(self):
        emica = MockEMICAWrapper([0.1,0.2,0.3], [0.,0.,0.])
        spectre = MockSPECTREWrapper([0.,0.,0.])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"; _create_dummy_face(p)
            faces = [AlignedFaceInfo("t", 0, "f0", 0.0, p,
                     (1.,0.,0.,0.,1.,0.), (224,224), (224,224))]
            r = extract_flame_parameters(faces, emica, spectre, 0.6, 4, "cpu")
        assert len(r[0].jawpose) == 3

    def test_head_pose_dim(self):
        emica = MockEMICAWrapper([0.,0.,0.], [0.1,0.2,0.3])
        spectre = MockSPECTREWrapper([0.,0.,0.])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"; _create_dummy_face(p)
            faces = [AlignedFaceInfo("t", 0, "f0", 0.0, p,
                     (1.,0.,0.,0.,1.,0.), (224,224), (224,224))]
            r = extract_flame_parameters(faces, emica, spectre, 0.6, 4, "cpu")
        assert len(r[0].globalpose) == 3

    def test_eye_pose_dim(self):
        emica = MockEMICAWrapper([0.,0.,0.], [0.,0.,0.])
        spectre = MockSPECTREWrapper([0.,0.,0.])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"; _create_dummy_face(p)
            faces = [AlignedFaceInfo("t", 0, "f0", 0.0, p,
                     (1.,0.,0.,0.,1.,0.), (224,224), (224,224))]
            r = extract_flame_parameters(faces, emica, spectre, 0.6, 4, "cpu")
        assert len(r[0].eye_pose) == 6

    def test_shapecode_dim(self):
        emica = MockEMICAWrapper([0.,0.,0.], [0.,0.,0.])
        spectre = MockSPECTREWrapper([0.,0.,0.])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"; _create_dummy_face(p)
            faces = [AlignedFaceInfo("t", 0, "f0", 0.0, p,
                     (1.,0.,0.,0.,1.,0.), (224,224), (224,224))]
            r = extract_flame_parameters(faces, emica, spectre, 0.6, 4, "cpu")
        assert len(r[0].shapecode) == 300

    def test_spectre_jawpose_dim(self):
        emica = MockEMICAWrapper([0.,0.,0.], [0.,0.,0.])
        spectre = MockSPECTREWrapper([0.1,0.2,0.3])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"; _create_dummy_face(p)
            faces = [AlignedFaceInfo("t", 0, "f0", 0.0, p,
                     (1.,0.,0.,0.,1.,0.), (224,224), (224,224))]
            r = extract_flame_parameters(faces, emica, spectre, 0.6, 4, "cpu")
        assert r[0].spectre_jawpose is not None
        assert len(r[0].spectre_jawpose) == 3
# ---------------------------------------------------------------------------
# Test 2: Fusion correctness
# ---------------------------------------------------------------------------

class TestJawFusion:
    def test_jaw_emica_at_mid_lambda(self):
        """Final scheme: jaw = EMICA even at λ=0.6 (fusion deprecated)."""
        emica = MockEMICAWrapper([0.5, -0.3, 0.1], [0., 0., 0.])
        spectre = MockSPECTREWrapper([-0.5, 0.3, -0.1])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"; _create_dummy_face(p)
            faces = [AlignedFaceInfo("t", 0, "f0", 0.0, p,
                     (1.,0.,0.,0.,1.,0.), (224,224), (224,224))]
            r = extract_flame_parameters(faces, emica, spectre, 0.6, 4, "cpu")
        np.testing.assert_array_almost_equal(
            np.array(r[0].jawpose), np.array([0.5, -0.3, 0.1]))
        np.testing.assert_array_almost_equal(
            np.array(r[0].emica_jawpose), np.array([0.5, -0.3, 0.1]))

    def test_jaw_emica_at_zero_lambda(self):
        """Final scheme: jaw = EMICA even at λ=0.0 (fusion deprecated)."""
        emica = MockEMICAWrapper([0.5, -0.3, 0.1], [0., 0., 0.])
        spectre = MockSPECTREWrapper([9., 9., 9.])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"; _create_dummy_face(p)
            faces = [AlignedFaceInfo("t", 0, "f0", 0.0, p,
                     (1.,0.,0.,0.,1.,0.), (224,224), (224,224))]
            r = extract_flame_parameters(faces, emica, spectre, 0.0, 4, "cpu")
        np.testing.assert_array_almost_equal(
            np.array(r[0].jawpose), np.array([0.5, -0.3, 0.1]))

    def test_jaw_emica_regardless_of_lambda_1(self):
        emica = MockEMICAWrapper([9., 9., 9.], [0., 0., 0.])
        spectre = MockSPECTREWrapper([-0.5, 0.3, -0.1])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"; _create_dummy_face(p)
            faces = [AlignedFaceInfo("t", 0, "f0", 0.0, p,
                     (1.,0.,0.,0.,1.,0.), (224,224), (224,224))]
            r = extract_flame_parameters(faces, emica, spectre, 1.0, 4, "cpu")
        np.testing.assert_array_almost_equal(
            np.array(r[0].jawpose), np.array([9., 9., 9.]))

    def test_batch_processing(self):
        emica = MockEMICAWrapper([0.1,0.2,0.3], [0.,0.,0.], list(range(100)))
        spectre = MockSPECTREWrapper([0.,0.,0.])
        with tempfile.TemporaryDirectory() as d:
            faces = []
            for i in range(20):
                p = Path(d) / f"f{i:04d}.png"; _create_dummy_face(p)
                faces.append(AlignedFaceInfo("t", i, f"f{i}", i/25., p,
                             (1.,0.,0.,0.,1.,0.), (224,224), (224,224)))
            r = extract_flame_parameters(faces, emica, spectre, 0.6, 5, "cpu")
        assert len(r) == 20
        for i, f in enumerate(r):
            assert f.frame_index == i
# ---------------------------------------------------------------------------
# Test 3: Stage12 input/output validation
# ---------------------------------------------------------------------------

class TestStage12Validation:
    def test_valid_input_passes(self):
        stage = Stage12FLAME()
        ctx = _make_context()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"; _create_dummy_face(p)
            faces = [AlignedFaceInfo("t", 0, "f0", 0.0, p,
                     (1.,0.,0.,0.,1.,0.), (224,224), (224,224))]
            assert stage.validate_input(faces, ctx) is True

    def test_empty_input_passes(self):
        stage = Stage12FLAME()
        assert stage.validate_input([], _make_context()) is True

    def test_wrong_type_fails(self):
        stage = Stage12FLAME()
        assert stage.validate_input("not a list", _make_context()) is False

    def test_non_alignedface_fails(self):
        stage = Stage12FLAME()
        assert stage.validate_input(["not a face"], _make_context()) is False

    def test_valid_output_passes(self):
        stage = Stage12FLAME()
        features = [FLAMEFeature(0, tuple([0.]*300), tuple([0.]*100),
                     tuple([0.]*3), tuple([0.]*3), tuple([0.]*3),
                     tuple([0.]*6), None, None, tuple([0.]*3))]
        assert stage.validate_output(features, _make_context()) is True

    def test_wrong_output_type_fails(self):
        stage = Stage12FLAME()
        assert stage.validate_output("not a list", _make_context()) is False

    def test_empty_output_passes(self):
        stage = Stage12FLAME()
        assert stage.validate_output([], _make_context()) is True

    def test_stage12_process_empty(self):
        stage = Stage12FLAME()
        ctx = _make_context()
        assert stage.run([], ctx) == []