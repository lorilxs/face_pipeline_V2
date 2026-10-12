"""Stage 10: High-Precision Landmark Extraction.

Extracts 106-point facial landmarks using InsightFace 2d106det model.
Only processes clips accepted by Stage 9 Quality Gate.
"""

from face_pipeline_v2.stages.stage_10_landmark.stage import Stage10Landmark

__all__ = ["Stage10Landmark"]
