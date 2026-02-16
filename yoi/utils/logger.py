"""Logging utilities for YOI Vision Engine."""

import json
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _env_positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return max(1, default)
    try:
        return max(1, int(value))
    except ValueError:
        return max(1, default)


class ContextFilter(logging.Filter):
    """Inject runtime logging context (config tag) into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        config_tag = os.getenv("YOI_LOG_CONFIG_TAG", "global")
        setattr(record, "config_tag", config_tag)
        return True


class ColorFormatter(logging.Formatter):
    """Console formatter with per-config and per-level ANSI colors."""

    RESET = "\033[0m"
    LEVEL_COLORS = {
        "DEBUG": "\033[37m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    TAG_COLORS = ["\033[34m", "\033[35m", "\033[96m", "\033[92m", "\033[93m", "\033[94m"]

    def _color_for_tag(self, tag: str) -> str:
        if not tag:
            return "\033[90m"
        return self.TAG_COLORS[sum(ord(char) for char in tag) % len(self.TAG_COLORS)]

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        level = getattr(record, "levelname", "INFO")
        level_color = self.LEVEL_COLORS.get(level, "\033[37m")
        tag = getattr(record, "config_tag", "global")
        tag_color = self._color_for_tag(tag)

        colored_base = base.replace(level, f"{level_color}{level}{self.RESET}", 1)
        colored_base = colored_base.replace(
            f"[cfg:{tag}]",
            f"[cfg:{tag_color}{tag}{self.RESET}]",
            1,
        )
        return colored_base


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data, ensure_ascii=False)


class YOILogger:
    """Singleton logger service for YOI Engine."""

    _instance = None
    _loggers: Dict[str, logging.Logger] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # File log directory priority:
        # 1) YOI_LOG_DIR (explicit)
        # 2) LOGS_PATH/engine
        # 3) package-local yoi/logs
        package_root = Path(__file__).resolve().parents[1]
        explicit_log_dir = os.getenv("YOI_LOG_DIR")
        logs_path = os.getenv("LOGS_PATH")

        if explicit_log_dir:
            self.log_dir = Path(explicit_log_dir)
        elif logs_path:
            self.log_dir = Path(logs_path) / "engine"
        else:
            self.log_dir = package_root / "logs"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def get_logger(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        json_format: bool = False,
    ) -> logging.Logger:
        """Get or create a logger."""
        if name in self._loggers:
            return self._loggers[name]

        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        effective_level = getattr(logging, level_name, level)

        logger = logging.getLogger(name)
        logger.setLevel(effective_level)
        logger.propagate = False
        logger.handlers.clear()
        logger.filters.clear()
        logger.addFilter(ContextFilter())

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(effective_level)

        base_format = "%(asctime)s - %(name)s - [cfg:%(config_tag)s] - %(levelname)s - %(message)s"

        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(fmt=base_format, datefmt="%Y-%m-%d %H:%M:%S")

        console_formatter: logging.Formatter = formatter
        if not json_format and _env_enabled("YOI_LOG_COLOR", default=True):
            console_formatter = ColorFormatter(fmt=base_format, datefmt="%Y-%m-%d %H:%M:%S")

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Enable file handler when requested and allowed by env.
        log_to_file = _env_enabled("YOI_LOG_TO_FILE", default=True)
        max_mb = _env_positive_int("YOI_LOG_MAX_MB", default=5)
        backup_count = _env_positive_int("YOI_LOG_BACKUP_COUNT", default=3)

        if log_file and log_to_file:
            log_suffix = os.getenv("YOI_LOG_FILE_SUFFIX", "").strip()
            effective_log_file = log_file
            if log_suffix:
                base_name = Path(log_file).stem
                ext = Path(log_file).suffix
                effective_log_file = f"{base_name}.{log_suffix}{ext}"

            log_path = self.log_dir / effective_log_file
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max(1, max_mb) * 1024 * 1024,
                backupCount=max(1, backup_count),
                encoding="utf-8",
            )
            file_handler.setLevel(effective_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self._loggers[name] = logger
        return logger

    def get_engine_logger(self) -> logging.Logger:
        """Logger for main engine."""
        return self.get_logger("yoi.engine", "engine.log")

    def get_inference_logger(self) -> logging.Logger:
        """Logger for inference."""
        return self.get_logger("yoi.inference", "inference.log")

    def get_analytics_logger(self) -> logging.Logger:
        """Logger for analytics in JSON format."""
        return self.get_logger("yoi.analytics", "analytics.json", json_format=True)

    def get_output_logger(self) -> logging.Logger:
        """Logger for output generation."""
        return self.get_logger("yoi.output", "output.log")

    def get_video_logger(self) -> logging.Logger:
        """Logger for video processing."""
        return self.get_logger("yoi.video", "video.log")

    def get_rtsp_logger(self) -> logging.Logger:
        """Logger for RTSP stream reader/pusher status."""
        return self.get_logger("yoi.rtsp", "rtsp.log")

    def get_dashboard_logger(self) -> logging.Logger:
        """Logger for dashboard delivery activity."""
        return self.get_logger("yoi.dashboard", "dashboard.log")


# Singleton instance
logger_service = YOILogger()
