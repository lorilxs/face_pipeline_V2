"""Unified logging system for Face_Pipeline_V2.

Provides a single Logger class that outputs to both console and file,
supporting INFO, WARNING, ERROR, and DEBUG levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class PipelineLogger:
    """统一Logger — Pipeline全局日志系统。

    Outputs to:
        - Console (stdout): human-readable log messages.
        - File (outputs/logs/): persistent log storage for traceability.

    Usage:
        logger = PipelineLogger(name="Face_Pipeline_V2", log_dir=Path("outputs/logs"))
        logger.info("Pipeline started")
        logger.warning("Low confidence detection")
        logger.error("Model loading failed")
        logger.debug("Frame 42: bbox=(10, 20, 100, 120)")
    """

    _LOGGER_NAME: str = "Face_Pipeline_V2"
    _LOG_FORMAT: str = "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s"
    _DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        name: str = _LOGGER_NAME,
        log_dir: Optional[Path] = None,
        level: int = logging.INFO,
    ) -> None:
        """Initialize the PipelineLogger.

        Args:
            name: Logger name identifier.
            log_dir: Directory for log file output. If None, file logging is disabled.
            level: Minimum logging level (default: INFO).
        """
        self._name: str = name
        self._log_dir: Optional[Path] = log_dir
        self._level: int = level

        self._logger: logging.Logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers.clear()

        formatter = logging.Formatter(
            fmt=self._LOG_FORMAT,
            datefmt=self._DATE_FORMAT,
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # File handler (optional)
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                filename=log_dir / "pipeline.log",
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    @property
    def name(self) -> str:
        """Logger name."""
        return self._name

    def info(self, message: str) -> None:
        """Log an INFO level message.

        Args:
            message: The log message.
        """
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """Log a WARNING level message.

        Args:
            message: The log message.
        """
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Log an ERROR level message.

        Args:
            message: The log message.
        """
        self._logger.error(message)

    def debug(self, message: str) -> None:
        """Log a DEBUG level message.

        Args:
            message: The log message.
        """
        self._logger.debug(message)

    def set_level(self, level: int) -> None:
        """Update the logging level at runtime.

        Args:
            level: New logging level (e.g. logging.DEBUG).
        """
        self._level = level
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)
