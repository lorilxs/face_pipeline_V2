"""Stage 0.5: Resolution Filter.

Video-level resolution hard filter (rule-based, no model). Rejects videos
below a configurable minimum resolution immediately after Stage 0 metadata
extraction and before any downstream model inference (Stage 1+).
"""

from face_pipeline_v2.stages.stage_00_5_resolution_filter.stage import (
    Stage0_5ResolutionFilter,
)

__all__ = ["Stage0_5ResolutionFilter"]
