"""Runtime pipeline helpers."""

from __future__ import annotations

from yoi.components.engine import VisionEngine
from yoi.config import YOIConfig


def run_pipeline(config: YOIConfig) -> VisionEngine:
    """Create and run the vision pipeline from config."""
    engine = VisionEngine(config)
    engine.process()
    return engine
