"""Unified Pipeline exception hierarchy for Face_Pipeline_V2.

All Pipeline-specific exceptions inherit from PipelineError,
enabling fine-grained error handling per AISDD error handling requirements.
"""

from typing import Any


class PipelineError(Exception):
    """Base exception for all Pipeline-specific errors.

    All custom exceptions in the pipeline inherit from this class,
    allowing callers to catch PipelineError as a blanket for any
    pipeline-originated failure.
    """


class InputValidationError(PipelineError):
    """Stage input validation failed.

    Raised when a Stage's validate_input() returns False,
    indicating the provided input data does not meet the
    stage's input contract.
    """


class StageExecutionError(PipelineError):
    """Stage output validation or stage-level execution failure.

    Raised when validate_output() returns False after processing,
    or when a stage encounters an error that prevents completion.
    """


class VideoReadError(PipelineError):
    """Video file cannot be read or decoded.

    Raised when:
        - The video file does not exist on disk.
        - FFprobe returns a non-zero exit code.
        - The video file is corrupted or unreadable.
    """


class MissingDependencyError(PipelineError):
    """Required external dependency is not available.

    Raised when:
        - FFprobe (or other CLI tools) are not found on PATH.
        - Required system packages are missing.
    """


class MetadataExtractionError(PipelineError):
    """Failed to extract or parse metadata from a media file.

    Raised when:
        - FFprobe JSON output cannot be parsed.
        - Required metadata fields are absent or malformed.
    """


class MissingStreamError(PipelineError):
    """Required media stream (video or audio) is absent.

    Raised when:
        - A video file has no video stream.
        - An expected audio stream is missing.
    """


class DataProcessingError(PipelineError):
    """Data processing or conversion failed.

    Raised when:
        - FFmpeg conversion exits with non-zero code.
        - Output file is not produced after processing.
        - Post-processing verification fails.
    """


class InvalidBboxError(DataProcessingError):
    """Face bounding box missing or invalid.

    Raised when a detected face's tight bbox is degenerate
    (``x2 <= x1`` or ``y2 <= y1``) or contains non-finite coordinates.
    Subclasses DataProcessingError for backward compatibility with existing
    ``except DataProcessingError`` handlers.
    """


class FaceCropError(DataProcessingError):
    """Face crop failed after successful detection.

    Raised when the expanded-bbox crop cannot be produced or written
    (invalid crop region or I/O failure).  Subclasses DataProcessingError
    for backward compatibility.
    """


class FaceResolutionFilterError(StageExecutionError):
    """Stage 5 face-resolution pre-filter rejected the video.

    Hard video-level FAIL gate raised after Stage 5 output when the configured
    tight-bbox resolution threshold is not met.  Carries the complete
    VideoResolutionResult for audit and reporting.
    """

    def __init__(self, message: str, result: Any = None) -> None:
        super().__init__(message)
        self.result = result
