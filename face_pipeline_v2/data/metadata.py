"""Pipeline metadata model for stage result tracking.

Per AISDD 2.4.4 data traceability requirements, every stage result
must be traceable back to its source video through a metadata chain.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class PipelineMetadata:
    """Pipeline运行元数据 — 用于Stage结果追踪。

    Attached to every stage output to maintain full provenance chain
    from TrainingSample back to Raw Video.
    """

    pipeline_version: str
    """Pipeline版本 — 当前Pipeline版本号 (e.g. v1.0)"""
    config_version: str
    """配置版本 — 使用的配置文件版本标识"""
    model_version: str
    """模型版本 — 使用的模型版本标识"""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    """时间戳 — Pipeline执行时间 (ISO 8601 UTC)"""


def create_metadata(
    pipeline_version: str = "v1.0",
    config_version: str = "v1.0",
    model_version: str = "v1.0",
) -> PipelineMetadata:
    """Factory function to create PipelineMetadata with defaults.

    Args:
        pipeline_version: Pipeline version string.
        config_version: Configuration version string.
        model_version: Model version string.

    Returns:
        A frozen PipelineMetadata instance with UTC timestamp.
    """
    return PipelineMetadata(
        pipeline_version=pipeline_version,
        config_version=config_version,
        model_version=model_version,
    )
