"""Stage 11: Face Alignment.

Normalizes faces to 256×256 canonical space using landmark-based
affine (similarity) transform. Saves affine matrix for inverse recovery.

Uses: 2-point (eye-center) similarity transform + cv2.warpAffine (OpenCV, no GPU).
"""

from face_pipeline_v2.stages.stage_11_alignment.stage import Stage11Alignment

__all__ = ["Stage11Alignment"]
