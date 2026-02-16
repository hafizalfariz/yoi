"""Base Feature Class

Abstract base class for all detection features.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Detection:
    """Detection result from YOLO"""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] normalized
    centroid: tuple  # (x, y) normalized


@dataclass
class FeatureResult:
    """Result from feature processing"""

    feature_type: str
    detections: List[Detection]
    metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []


class BaseFeature(ABC):
    """Abstract base class for all features

    All features must implement:
    - process(): Process detections for this frame
    - get_metrics(): Get current metrics
    - reset(): Reset feature state
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize feature with config

        Args:
            config: Feature configuration dict from YAML
        """
        self.config = config
        self.frame_count = 0
        self.alerts = []

    @abstractmethod
    def process(self, detections: List[Detection], frame_idx: int) -> FeatureResult:
        """Process detections for current frame

        Args:
            detections: List of detections from tracker
            frame_idx: Current frame index

        Returns:
            FeatureResult with metrics and alerts
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current feature metrics

        Returns:
            Dictionary of current metrics
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset feature state"""
        pass

    def _check_point_in_polygon(self, point: tuple, polygon: List[tuple]) -> bool:
        """Check if point is inside polygon using ray casting

        Args:
            point: (x, y) point to check
            polygon: List of (x, y) vertices

        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _check_line_crossing(
        self, pt1: tuple, pt2: tuple, line_start: tuple, line_end: tuple
    ) -> Optional[str]:
        """Check if trajectory crosses line and return direction

        Args:
            pt1: Previous point (x, y)
            pt2: Current point (x, y)
            line_start: Line start point (x, y)
            line_end: Line end point (x, y)

        Returns:
            'in', 'out', or None
        """
        # Check if trajectory crosses line using line intersection
        x1, y1 = pt1
        x2, y2 = pt2
        x3, y3 = line_start
        x4, y4 = line_end

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            # Lines intersect, determine direction
            # Use cross product to determine which side
            cross = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
            return "in" if cross > 0 else "out"

        return None
