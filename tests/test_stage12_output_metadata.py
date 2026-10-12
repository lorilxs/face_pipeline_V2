"""Test Stage12 output metadata — timestamp, clip_id, camera_params.

Verifies downstream compatibility fixes:
  - Issue 1: timestamp & clip_id added to FLAMEFeature
  - Issue 2: read_stage12_outputs() reads cam from npz if present
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from face_pipeline_v2.data.models import FLAMEFeature
from face_pipeline_v2.stages.stage_12_flame.integration import \
    read_stage12_outputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_npz(
    npz_path: Path,
    T: int = 3,
    *,
    include_timestamp: bool = True,
    include_cam: bool = False,
    include_clip_id: bool = False,
) -> None:
    """Create a minimal npz file mimicking Stage12 runner output."""
    arrays: dict = {
        "frame_index": np.arange(T, dtype=np.int64),
        "shapecode": np.random.randn(T, 300).astype(np.float32),
        "expression_code": np.random.randn(T, 100).astype(np.float32),
        "jaw_pose": np.random.randn(T, 3).astype(np.float32),
        "head_pose": np.random.randn(T, 3).astype(np.float32),
        "eye_pose": np.zeros((T, 6), dtype=np.float32),
        "spectre_jawpose": np.random.randn(T, 3).astype(np.float32),
    }
    if include_timestamp:
        arrays["timestamp"] = np.array([0.0, 0.04, 0.08], dtype=np.float32)
    if include_cam:
        arrays["cam"] = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
    if include_clip_id:
        arrays["clip_id"] = np.array(["clip_A", "clip_A", "clip_A"])

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
# ---------------------------------------------------------------------------
# FLAMEFeature dataclass tests
# ---------------------------------------------------------------------------

class TestFLAMEFeatureMetadata:
    """Verify timestamp and clip_id fields on the FLAMEFeature dataclass."""

    def test_timestamp_default(self) -> None:
        """timestamp defaults to 0.0 when not provided (backward compat)."""
        f = FLAMEFeature(
            frame_index=0,
            shapecode=(0.0,) * 300,
            expcode=(0.0,) * 100,
            jawpose=(0.0,) * 3,
            globalpose=(0.0,) * 3,
            cam=(0.0,) * 3,
        )
        assert f.timestamp == 0.0

    def test_clip_id_default(self) -> None:
        """clip_id defaults to '' when not provided (backward compat)."""
        f = FLAMEFeature(
            frame_index=0,
            shapecode=(0.0,) * 300,
            expcode=(0.0,) * 100,
            jawpose=(0.0,) * 3,
            globalpose=(0.0,) * 3,
            cam=(0.0,) * 3,
        )
        assert f.clip_id == ""

    def test_timestamp_explicit(self) -> None:
        """timestamp can be set explicitly."""
        f = FLAMEFeature(
            frame_index=0,
            shapecode=(0.0,) * 300,
            expcode=(0.0,) * 100,
            jawpose=(0.0,) * 3,
            globalpose=(0.0,) * 3,
            cam=(0.0,) * 3,
            timestamp=1.5,
        )
        assert f.timestamp == 1.5

    def test_clip_id_explicit(self) -> None:
        """clip_id can be set explicitly."""
        f = FLAMEFeature(
            frame_index=0,
            shapecode=(0.0,) * 300,
            expcode=(0.0,) * 100,
            jawpose=(0.0,) * 3,
            globalpose=(0.0,) * 3,
            cam=(0.0,) * 3,
            clip_id="test_clip_001",
        )
        assert f.clip_id == "test_clip_001"

    def test_full_with_metadata(self) -> None:
        """Full FLAMEFeature with timestamp and clip_id."""
        f = FLAMEFeature(
            frame_index=42,
            shapecode=(0.1,) * 300,
            expcode=(0.2,) * 100,
            jawpose=(0.3, 0.4, 0.5),
            globalpose=(0.6, 0.7, 0.8),
            cam=(1.0, 2.0, 3.0),
            timestamp=2.5,
            clip_id="clip_042",
            spectre_jawpose=(-0.3, -0.4, -0.5),
        )
        assert f.frame_index == 42
        assert f.timestamp == 2.5
        assert f.clip_id == "clip_042"
        assert f.spectre_jawpose == (-0.3, -0.4, -0.5)

    def test_immutable_timestamp(self) -> None:
        """timestamp is frozen (immutable)."""
        f = FLAMEFeature(
            frame_index=0,
            shapecode=(0.0,) * 300,
            expcode=(0.0,) * 100,
            jawpose=(0.0,) * 3,
            globalpose=(0.0,) * 3,
            cam=(0.0,) * 3,
        )
# ---------------------------------------------------------------------------
# read_stage12_outputs() tests
# ---------------------------------------------------------------------------

class TestReadStage12OutputsMetadata:
    """Verify read_stage12_outputs correctly restores metadata from npz."""

    def test_timestamp_restored(self) -> None:
        """timestamp is read from npz and attached to each FLAMEFeature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            npz_path = out_dir / "test_clip" / "flame_features.npz"
            _make_npz(npz_path, T=3, include_timestamp=True)

            features = read_stage12_outputs(out_dir, "test_clip")

            assert len(features) == 3
            assert features[0].timestamp == pytest.approx(0.0)
            assert features[1].timestamp == pytest.approx(0.04)
            assert features[2].timestamp == pytest.approx(0.08)

    def test_clip_id_from_npz(self) -> None:
        """clip_id is read from npz when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            npz_path = out_dir / "test_clip" / "flame_features.npz"
            _make_npz(npz_path, T=2, include_clip_id=True)

            features = read_stage12_outputs(out_dir, "test_clip")

            assert len(features) == 2
            assert features[0].clip_id == "clip_A"
            assert features[1].clip_id == "clip_A"

    def test_clip_id_fallback(self) -> None:
        """clip_id falls back to function arg when npz has no clip_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            npz_path = out_dir / "my_clip" / "flame_features.npz"
            _make_npz(npz_path, T=2, include_clip_id=False)

            features = read_stage12_outputs(out_dir, "my_clip")

            assert len(features) == 2
            assert features[0].clip_id == "my_clip"
            assert features[1].clip_id == "my_clip"

    def test_cam_zero_fallback(self) -> None:
        """cam is [0,0,0] when npz has no cam/camera_params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            npz_path = out_dir / "c" / "flame_features.npz"
            _make_npz(npz_path, T=2, include_cam=False)

            features = read_stage12_outputs(out_dir, "c")

            assert len(features) == 2
            assert features[0].cam == (0.0, 0.0, 0.0)
            assert features[1].cam == (0.0, 0.0, 0.0)

    def test_cam_restored(self) -> None:
        """cam is read from npz when 'cam' key is present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            npz_path = out_dir / "c" / "flame_features.npz"
            _make_npz(npz_path, T=2, include_cam=True)

            features = read_stage12_outputs(out_dir, "c")

            assert len(features) == 2
            assert features[0].cam == (1.0, 2.0, 3.0)
            assert features[1].cam == (4.0, 5.0, 6.0)

    def test_camera_params_key_restored(self) -> None:
        """cam is read from npz 'camera_params' key as fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            npz_path = out_dir / "c" / "flame_features.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_path,
                frame_index=np.arange(2),
                shapecode=np.random.randn(2, 300).astype(np.float32),
                expression_code=np.random.randn(2, 100).astype(np.float32),
                jaw_pose=np.random.randn(2, 3).astype(np.float32),
                head_pose=np.random.randn(2, 3).astype(np.float32),
                eye_pose=np.zeros((2, 6), dtype=np.float32),
                spectre_jawpose=np.random.randn(2, 3).astype(np.float32),
                camera_params=np.array(
                    [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                    dtype=np.float32,
                ),
            )

            features = read_stage12_outputs(out_dir, "c")

            assert len(features) == 2
            assert features[0].cam == (10.0, 20.0, 30.0)

    def test_timestamp_missing_graceful(self) -> None:
        """timestamp defaults to 0.0 when absent from npz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            npz_path = out_dir / "c" / "flame_features.npz"
            _make_npz(npz_path, T=2, include_timestamp=False)

            features = read_stage12_outputs(out_dir, "c")

            assert features[0].timestamp == 0.0
            assert features[1].timestamp == 0.0

    def test_file_not_found(self) -> None:
        """read_stage12_outputs raises FileNotFoundError for missing npz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            with pytest.raises(FileNotFoundError):
                read_stage12_outputs(out_dir, "nonexistent")
        with pytest.raises(Exception):
            f.timestamp = 99.9  # type: ignore[misc]