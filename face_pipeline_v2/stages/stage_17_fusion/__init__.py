"""Stage 17 — Training Sample Assembly (Parameter Alignment & Fusion).

将 Stage12 FLAME / Stage13 Gaze / Stage14 Audio 三模态特征按统一时间轴融合,
装配为 clip-level 的 TrainingSample 样本包 (见 docs/Stage17_Design_Review.md §4)。

当前状态: skeleton (DataModel + 接口定义); 融合逻辑待 Phase2 实现。
"""

from face_pipeline_v2.stages.stage_17_fusion.export import (
    export_multimodal_sample,
    read_multimodal_export,
    validate_export_metadata,
)
from face_pipeline_v2.stages.stage_17_fusion.integration import (
    read_training_sample,
    save_training_sample,
)
from face_pipeline_v2.stages.stage_17_fusion.processor import (
    build_training_sample,
)
from face_pipeline_v2.stages.stage_17_fusion.stage import Stage17Fusion

__all__ = [
    "Stage17Fusion",
    "build_training_sample",
    "read_training_sample",
    "save_training_sample",
    "export_multimodal_sample",
    "read_multimodal_export",
    "validate_export_metadata",
]


