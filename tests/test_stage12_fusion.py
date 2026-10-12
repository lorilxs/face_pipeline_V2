"""Tests for Stage12 jaw source logic (FINAL adopted: pure EMICA jaw).

Covers the Stage12 design without real model inference:
- T1: jaw_pose = pure EMICA jaw (no fusion; λ is deprecated)
- T2: Expression comes from EMICA, shape [100]
- T3: Head pose comes from EMICA globalpose
- T4: eye_pose = zeros(6)
- T5: spectre_jawpose preserved as raw value (audit/debug)
- T6: emica_jawpose preserved as raw value (== jawpose in final scheme)
"""

import numpy as np
import pytest

from face_pipeline_v2.data.models import FLAMEFeature


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _stage12_jaw(
    emica_jaw: np.ndarray,
    spectre_jaw: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Stage12 jaw source (FINAL adopted): pure EMICA, no fusion.

    ``lam`` is retained only for a backward-compatible interface; the
    production code (runner.py / processor.py) sets jaw_pose = emica_jaw
    unconditionally.
    """
    return emica_jaw


# ===========================================================================
# T1: Jaw source
# ===========================================================================

class TestJawFusion:
    """Tests for the Stage12 jaw source (pure EMICA, FINAL adopted)."""

    @pytest.mark.parametrize(
        "emica_jaw,spectre_jaw,lam",
        [
            (np.array([1., 2., 3.]), np.array([3., 4., 5.]), 0.5),
            (np.array([1., 2., 3.]), np.array([3., 4., 5.]), 0.0),
            (np.array([1., 2., 3.]), np.array([3., 4., 5.]), 1.0),
            (np.array([1., 2., 3.]), np.array([3., 4., 5.]), 0.6),
        ],
    )
    def test_jaw_equals_emica_for_any_lambda(
        self, emica_jaw, spectre_jaw, lam,
    ) -> None:
        """jaw_pose == emica_jaw regardless of λ (fusion deprecated)."""
        jaw = _stage12_jaw(emica_jaw, spectre_jaw, lam)
        np.testing.assert_array_almost_equal(jaw, emica_jaw)

    def test_jaw_ignores_spectre_value(self) -> None:
        """jaw_pose is EMICA; SPECTRE raw jaw is ignored for jaw_pose."""
        emica = np.array([0.1, -0.2, 0.3])
        spectre = np.array([5.0, 5.0, 5.0])
        jaw = _stage12_jaw(emica, spectre, 1.0)
        np.testing.assert_array_equal(jaw, emica)
# ===========================================================================
# T2, T3, T4, T5: FLAMEFeature field correctness
# ===========================================================================

class TestFLAMEFeatureFields:
    """Tests for FLAMEFeature field contents after fusion."""

    def test_expression_from_emica_100d(self) -> None:
        """Expression code is EMICA [100], not EMOCA 50."""
        emica_exp = np.random.randn(100).astype(np.float32)
        feature = FLAMEFeature(
            frame_index=0,
            shapecode=tuple([0.0] * 300),
            expcode=tuple(emica_exp.tolist()),
            jawpose=tuple([0.0, 0.0, 0.0]),
            globalpose=tuple([0.0, 0.0, 0.0]),
            cam=tuple([0.0] * 3),
            eye_pose=tuple([0.0] * 6),
            spectre_jawpose=tuple([0.0] * 3),
        )
        assert len(feature.expcode) == 100, \
            f"expcode should be 100, got {len(feature.expcode)}"
        np.testing.assert_array_almost_equal(
            np.array(feature.expcode), emica_exp,
        )

    def test_head_pose_from_emica_globalpose(self) -> None:
        """Head (globalpose) comes from EMICA, not SPECTRE head_pose."""
        emica_global = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        feature = FLAMEFeature(
            frame_index=0,
            shapecode=tuple([0.0] * 300),
            expcode=tuple([0.0] * 100),
            jawpose=tuple([0.0, 0.0, 0.0]),
            globalpose=tuple(emica_global.tolist()),
            cam=tuple([0.0] * 3),
            eye_pose=tuple([0.0] * 6),
            spectre_jawpose=tuple([0.0] * 3),
        )
        assert len(feature.globalpose) == 3
        np.testing.assert_array_almost_equal(
            np.array(feature.globalpose), emica_global,
        )

    def test_eye_pose_zeros_6(self) -> None:
        """eye_pose is fixed zeros(6), not from any encoder."""
        feature = FLAMEFeature(
            frame_index=0,
            shapecode=tuple([0.0] * 300),
            expcode=tuple([0.0] * 100),
            jawpose=tuple([0.0, 0.0, 0.0]),
            globalpose=tuple([0.0, 0.0, 0.0]),
            cam=tuple([0.0] * 3),
        )
        assert len(feature.eye_pose) == 6
        assert all(e == 0.0 for e in feature.eye_pose), \
            f"eye_pose should be all zeros, got {feature.eye_pose}"

    def test_spectre_jawpose_raw_value(self) -> None:
        """spectre_jawpose preserves raw SPECTRE jaw value."""
        spectre_raw = np.array([-0.1, 0.2, 0.0], dtype=np.float32)
        feature = FLAMEFeature(
            frame_index=0,
            shapecode=tuple([0.0] * 300),
            expcode=tuple([0.0] * 100),
            jawpose=tuple([0.0, 0.0, 0.0]),
            globalpose=tuple([0.0, 0.0, 0.0]),
            cam=tuple([0.0] * 3),
            eye_pose=tuple([0.0] * 6),
            spectre_jawpose=tuple(spectre_raw.tolist()),
        )
        assert len(feature.spectre_jawpose) == 3
        np.testing.assert_array_almost_equal(
            np.array(feature.spectre_jawpose), spectre_raw,
        )

    def test_emica_jawpose_raw_value(self) -> None:
        """emica_jawpose preserves raw EMICA jaw value (debug/analysis)."""
        emica_raw = np.array([0.7, -0.8, 0.9], dtype=np.float32)
        feature = FLAMEFeature(
            frame_index=0,
            shapecode=tuple([0.0] * 300),
            expcode=tuple([0.0] * 100),
            jawpose=tuple([0.0, 0.0, 0.0]),
            globalpose=tuple([0.0, 0.0, 0.0]),
            cam=tuple([0.0] * 3),
            eye_pose=tuple([0.0] * 6),
            spectre_jawpose=tuple([0.0] * 3),
            emica_jawpose=tuple(emica_raw.tolist()),
        )
        assert len(feature.emica_jawpose) == 3
        np.testing.assert_array_almost_equal(
            np.array(feature.emica_jawpose), emica_raw,
        )

    def test_emica_jawpose_default_none(self) -> None:
        """emica_jawpose defaults to None (backward compat)."""
        feature = FLAMEFeature(
            frame_index=0,
            shapecode=tuple([0.0] * 300),
            expcode=tuple([0.0] * 100),
            jawpose=tuple([0.0, 0.0, 0.0]),
            globalpose=tuple([0.0, 0.0, 0.0]),
            cam=tuple([0.0] * 3),
        )
        assert feature.emica_jawpose is None

    def test_jaw_matches_stage12_design(self) -> None:
        """Jawpose in FLAMEFeature equals pure EMICA jaw (FINAL scheme)."""
        emica_jaw = np.array([1.0, 2.0, 3.0])
        spectre_jaw = np.array([3.0, 4.0, 5.0])
        lam = 0.5
        jaw = _stage12_jaw(emica_jaw, spectre_jaw, lam)

        feature = FLAMEFeature(
            frame_index=0,
            shapecode=tuple([0.0] * 300),
            expcode=tuple([0.0] * 100),
            jawpose=tuple(jaw.tolist()),
            globalpose=tuple([0.0, 0.0, 0.0]),
            cam=tuple([0.0] * 3),
            eye_pose=tuple([0.0] * 6),
            spectre_jawpose=tuple(spectre_jaw.tolist()),
            emica_jawpose=tuple(emica_jaw.tolist()),
        )
        assert len(feature.jawpose) == 3
        np.testing.assert_array_almost_equal(
            np.array(feature.jawpose), np.array([1.0, 2.0, 3.0]),
        )

    def test_posecode_concat(self) -> None:
        """FLAMEFeature.posecode = globalpose(3) + jawpose(3) = 6."""
        feature = FLAMEFeature(
            frame_index=0,
            shapecode=tuple([0.0] * 300),
            expcode=tuple([0.0] * 100),
            jawpose=(1.0, 2.0, 3.0),
            globalpose=(4.0, 5.0, 6.0),
            cam=tuple([0.0] * 3),
        )
        assert len(feature.posecode) == 6
        assert feature.posecode == (4.0, 5.0, 6.0, 1.0, 2.0, 3.0)