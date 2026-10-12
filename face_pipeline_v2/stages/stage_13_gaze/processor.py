"""Stage 13 Gaze Processor — MediaPipe FaceLandmarker gaze extraction.

Transplanted from old pipeline:
  handle_stage3_pre_gazed_mediapipe.py → get_gaze_vector_from_landmarks()

Core algorithm:
  1. face_normal = cross(right_cheek - left_cheek, chin - forehead) → normalize
  2. pupil_offset = (iris_landmark - eye_center) / face_width
  3. gaze = normalize(face_normal + 3.0 * pupil_offset)
  4. Fallback: if iris landmarks unavailable, gaze = face_normal
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from face_pipeline_v2.data.models import AlignedFaceInfo, GazeFeature

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

# ---------------------------------------------------------------------------
# Constants — MediaPipe FaceMesh 478-point topology
# ---------------------------------------------------------------------------

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144, 163, 7]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380, 381, 362]
LEFT_PUPIL_IDX = 468
RIGHT_PUPIL_IDX = 473
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
def get_gaze_vector_from_landmarks(
    landmarks: Any,
    frame_width: int,
    frame_height: int,
) -> Dict[str, float]:
    """Compute gaze vector from MediaPipe FaceLandmarker result.

    Coordinate system aligned with OpenFace: z points toward camera (negative).

    Algorithm:
        1. face_normal = normalize(cross(cheek_LR, chin_forehead))
        2. pupil_offset = (iris - eye_center) / face_width
        3. gaze = normalize(face_normal + 3.0 * pupil_offset)

    Args:
        landmarks: MediaPipe NormalizedLandmarkList (indexable by int).
        frame_width: Image width in pixels.
        frame_height: Image height in pixels.

    Returns:
        Dict with keys: gaze_0_x/y/z, gaze_1_x/y/z  (all float).
    """

    def lm_to_3d(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([lm.x * frame_width, lm.y * frame_height, lm.z * frame_width])

    p_left_cheek = lm_to_3d(LEFT_CHEEK)
    p_right_cheek = lm_to_3d(RIGHT_CHEEK)
    p_forehead = lm_to_3d(FOREHEAD)
    p_chin = lm_to_3d(CHIN)

    v1 = p_right_cheek - p_left_cheek
    v2 = p_chin - p_forehead
    face_normal = np.cross(v1, v2)
    face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)
    if face_normal[2] > 0:
        face_normal = -face_normal

    # Iris landmarks available
    if len(landmarks) > RIGHT_PUPIL_IDX:
        left_eye_center = np.mean([lm_to_3d(i) for i in LEFT_EYE_INDICES], axis=0)
        right_eye_center = np.mean([lm_to_3d(i) for i in RIGHT_EYE_INDICES], axis=0)
        left_pupil = lm_to_3d(LEFT_PUPIL_IDX)
        right_pupil = lm_to_3d(RIGHT_PUPIL_IDX)

        face_width = np.linalg.norm(p_right_cheek - p_left_cheek)
        left_offset = (left_pupil - left_eye_center) / (face_width + 1e-8)
        right_offset = (right_pupil - right_eye_center) / (face_width + 1e-8)

        gaze_weight = 3.0
        left_gaze = face_normal + gaze_weight * left_offset
        right_gaze = face_normal + gaze_weight * right_offset

        left_gaze = left_gaze / (np.linalg.norm(left_gaze) + 1e-8)
        right_gaze = right_gaze / (np.linalg.norm(right_gaze) + 1e-8)

        return {
            "gaze_0_x": float(left_gaze[0]),
            "gaze_0_y": float(left_gaze[1]),
            "gaze_0_z": float(left_gaze[2]),
            "gaze_1_x": float(right_gaze[0]),
            "gaze_1_y": float(right_gaze[1]),
            "gaze_1_z": float(right_gaze[2]),
        }

    # Fallback: no iris landmarks → both eyes = face_normal
    return {
        "gaze_0_x": float(face_normal[0]),
        "gaze_0_y": float(face_normal[1]),
        "gaze_0_z": float(face_normal[2]),
        "gaze_1_x": float(face_normal[0]),
        "gaze_1_y": float(face_normal[1]),
        "gaze_1_z": float(face_normal[2]),
    }


def create_mediapipe_detector(
    model_path: str,
    delegate: str = "cpu",
) -> Any:
    """Create MediaPipe FaceLandmarker with optional GPU delegate.

    Args:
        model_path: Path to face_landmarker.task.
        delegate: "cpu" or "gpu".

    Returns:
        MediaPipe FaceLandmarker instance.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    mp_delegate = (
        mp_python.BaseOptions.Delegate.GPU
        if delegate == "gpu"
        else mp_python.BaseOptions.Delegate.CPU
    )
    base_options = mp_python.BaseOptions(
        model_asset_path=model_path,
        delegate=mp_delegate,
    )
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def extract_gaze_parameters(
    aligned_faces: List[AlignedFaceInfo],
    detector: Any,
    image_size: Tuple[int, int] = (256, 256),
) -> List[GazeFeature]:
    """Extract gaze vectors from aligned face images using MediaPipe.

    For each aligned face:
    1. Read aligned_face_path → BGR image
    2. Convert to RGB → MediaPipe Image
    3. Run FaceLandmarker.detect()
    4. Call get_gaze_vector_from_landmarks() → 6-dim gaze vector
    5. Build GazeFeature dataclass

    Args:
        aligned_faces: List of AlignedFaceInfo from Stage 11.
        detector: MediaPipe FaceLandmarker instance.
        image_size: Aligned face image dimensions (width, height).

    Returns:
        List of GazeFeature, one per input face.
    """

    results: List[GazeFeature] = []
    h, w = image_size

    for face_info in aligned_faces:
        img_path = Path(face_info.aligned_face_path)
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection = detector.detect(mp_image)

        if not detection.face_landmarks or len(detection.face_landmarks) == 0:
            continue

        landmarks = detection.face_landmarks[0]
        gaze_data = get_gaze_vector_from_landmarks(landmarks, w, h)

        results.append(GazeFeature(
            frame_index=face_info.frame_index,
            timestamp=face_info.timestamp,
            clip_id=face_info.clip_id,
            left_eye=(
                gaze_data["gaze_0_x"],
                gaze_data["gaze_0_y"],
                gaze_data["gaze_0_z"],
            ),
            right_eye=(
                gaze_data["gaze_1_x"],
                gaze_data["gaze_1_y"],
                gaze_data["gaze_1_z"],
            ),
            gaze_vector=(
                gaze_data["gaze_0_x"],
                gaze_data["gaze_0_y"],
                gaze_data["gaze_0_z"],
                gaze_data["gaze_1_x"],
                gaze_data["gaze_1_y"],
                gaze_data["gaze_1_z"],
            ),
        ))

    return results