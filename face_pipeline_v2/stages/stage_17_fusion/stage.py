"""Stage 17 — Training Sample Assembly (BaseStage skeleton).

输入: Stage12/13/14 三模态 NPZ (经各自 read_* 函数) + 溯源元数据
输出: List[TrainingSample]

Fusion 核心 (processor.build_training_sample + integration.save/read_training_sample)
已在 Phase2 实现; 本 BaseStage 的 process() 仍为占位 — 待 Phase3 接入 pipeline
时按 config (stage17.output_dir / fps / audio_tolerance) 调用 processor + integration。
"""

from __future__ import annotations

from typing import Any, List

from face_pipeline_v2.data.models import TrainingSample
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext


class Stage17Fusion(BaseStage):
    """Stage 17 — Parameter Alignment & Fusion (skeleton)."""

    _STAGE_NAME: str = "Stage 17 Training Sample Assembly"

    def __init__(self) -> None:
        super().__init__(stage_name=self._STAGE_NAME)

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """TODO(Phase2): 校验三模态输入形态与配置 (clip_id/fps/tolerance)。"""
        # skeleton: 仅记录配置快照, 不拒绝任何输入; 待 Phase2 冻结输入契约。
        stage17_cfg: dict = context.config.get("stage17", {})
        context.logger.debug(
            f"[{self._STAGE_NAME}] skeleton validate_input, "
            f"config keys={sorted(stage17_cfg.keys())}"
        )
        return True

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[TrainingSample]:
        """TODO(Phase2): 调用 processor.build_training_sample 完成三模态装配。"""
        raise NotImplementedError(
            f"[{self._STAGE_NAME}] Fusion not implemented yet (skeleton phase)."
        )

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """TODO(Phase2): 校验 TrainingSample 的 T 轴一致性 / 维度 / 溯源字段。"""
        # skeleton: 空结果视为合法占位。
        return output_data is None or isinstance(output_data, list)
