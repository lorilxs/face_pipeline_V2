"""Pipeline execution context — carries runtime environment across all Stages.

Per AISDD 2.11, PipelineContext holds the configuration, logger, model manager,
storage manager, and runtime state information used by every Stage.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PipelineContext:
    """Pipeline运行上下文 — 所有Stage共享的运行时环境。

    Immutable design ensures Stages cannot modify global state.
    Each Stage receives this context and operates within its bounds.

    Fields are defined per AISDD 2.11.
    """

    config: Dict[str, Any]
    """配置对象 — 当前运行配置 (loaded from configs/config.yaml)"""

    logger: Any
    """日志对象 — PipelineLogger instance for unified logging"""

    model_manager: Any
    """模型管理器 — ModelManager instance for model lifecycle"""

    runtime_info: Dict[str, Any] = field(default_factory=dict)
    """运行信息 — 当前任务状态 (video_id, stage progress, timestamps, etc.)"""

    stage_audit: Dict[str, Any] = field(default_factory=dict)
    """增量审计挂载 — Stage5.2 / Stage5.3 硬过滤阶段将每个被拒绝 clip 的记录
    追加到该字典（仅这两个 stage 使用）。frozen dataclass 只禁止字段重新
    赋值，允许对 dict 的就地修改（append）。该审计是 pipeline_manifest.json
    phase1 中 audit 部分的数据源：即使删除中间输出，逐 clip 拒绝原因仍保留。
    """

    def add_stage_audit(self, stage_id: str, clip_id: str, entry: Dict[str, Any]) -> None:
        """Append one per-clip audit entry for a hard-filter stage.

        Args:
            stage_id: e.g. ``stage_05_2_single_face_filter``.
            clip_id: the rejected clip id.
            entry: audit dict (reasons, metrics, ...) recorded at rejection time.
        """
        bucket = self.stage_audit.setdefault(stage_id, {}).setdefault("rejected", [])
        bucket.append({"clip_id": clip_id, **entry})

    @property
    def video_id(self) -> Optional[str]:
        """Shortcut to retrieve current video_id from runtime_info."""
        return self.runtime_info.get("video_id")

    @property
    def current_stage(self) -> Optional[str]:
        """Shortcut to retrieve current stage name from runtime_info."""
        return self.runtime_info.get("current_stage")
