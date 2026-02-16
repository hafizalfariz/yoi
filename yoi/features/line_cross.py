"""Line Crossing Feature

Detect when objects cross defined lines for in/out counting.
"""

from collections import defaultdict
from typing import Any, Dict, List

from yoi.utils.logger import logger_service

from .base import BaseFeature, Detection, FeatureResult


class LineCrossFeature(BaseFeature):
    """Line crossing detection feature

    Tracks objects crossing lines and counts in/out movements.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize line crossing feature

        Config expected:
            lines: List of line definitions with coords
            centroid: 'head', 'bottom', 'mid_centre'
            lost_threshold: Max frames before object considered lost
            allow_recounting: Allow same object to be recounted
            alerts: Alert thresholds for in/out counts
        """
        super().__init__(config)

        self.logger = logger_service.get_analytics_logger()
        self.lines = config.get("lines", [])
        self.centroid_mode = config.get("centroid", "mid_centre")
        self.lost_threshold = config.get("lost_threshold", 30)
        self.allow_recounting = config.get("allow_recounting", False)

        # Maximum allowed position jump between frames (normalized units 0-1).
        # If a track's centroid jumps more than this, we treat it as a new path
        # to reduce ID-switch artifacts where an ID "moves" from one person to another.
        self.max_position_jump = config.get("max_position_jump", 0.25)

        # Alert config
        alert_config = config.get("alerts", {})
        self.in_threshold = alert_config.get("in_warning_threshold", 5)
        self.out_threshold = alert_config.get("out_warning_threshold", 5)

        # Tracking state
        self.track_positions = {}  # track_id -> [(x, y), ...]
        self.track_last_seen = {}  # track_id -> frame_idx
        self.crossed_tracks = defaultdict(set)  # line_id -> set of track_ids

        # Counters per line
        self.in_counts = defaultdict(int)
        self.out_counts = defaultdict(int)
        self.total_in = 0
        self.total_out = 0

    def _get_centroid(self, detection: Detection) -> tuple:
        """Get centroid point based on mode

        Args:
            detection: Detection object

        Returns:
            (x, y) centroid point normalized [0, 1]
        """
        x1, y1, x2, y2 = detection.bbox

        if self.centroid_mode == "head":
            # Top center of bbox
            return ((x1 + x2) / 2, y1)
        elif self.centroid_mode == "bottom":
            # Bottom center of bbox
            return ((x1 + x2) / 2, y2)
        else:  # mid_centre
            # Center of bbox
            return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _get_line_attr(self, line, attr_name: str, default=None):
        """Get attribute from line config (dict or dataclass)

        Args:
            line: Line config (dict or dataclass)
            attr_name: Attribute name
            default: Default value if not found

        Returns:
            Attribute value or default
        """
        if isinstance(line, dict):
            return line.get(attr_name, default)
        else:
            return getattr(line, attr_name, default)

    def _check_line_crossing(
        self,
        prev_point: tuple,
        curr_point: tuple,
        line_start: tuple,
        line_end: tuple,
        line_config: Dict = None,
    ) -> str:
        """Check if trajectory crosses the line and determine direction

        Args:
            prev_point: Previous position (x, y) normalized [0, 1]
            curr_point: Current position (x, y) normalized [0, 1]
            line_start: Line start point (x, y)
            line_end: Line end point (x, y)
            line_config: Line configuration dict with direction info

        Returns:
            'in', 'out', or None if no crossing
        """
        # Check if line segment from prev to curr crosses the line
        if self._segments_intersect(prev_point, curr_point, line_start, line_end):
            # Determine direction based on line orientation and motion
            direction = (
                self._get_line_attr(line_config, "direction", "downward")
                if line_config
                else "downward"
            )
            orientation = (
                self._get_line_attr(line_config, "orientation", "horizontal")
                if line_config
                else "horizontal"
            )

            # Get line normal vector to determine direction
            line_vec = (line_end[0] - line_start[0], line_end[1] - line_start[1])
            # Normal perpendicular to line (rotate 90 degrees)
            normal = (-line_vec[1], line_vec[0])

            # Motion vector
            motion = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])

            # Dot product: positive = motion in normal direction, negative = opposite
            dot_product = motion[0] * normal[0] + motion[1] * normal[1]

            # Map direction based on line orientation
            if orientation == "horizontal":
                if direction == "downward":
                    return "in" if dot_product > 0 else "out"
                elif direction == "upward":
                    return "in" if dot_product < 0 else "out"
                else:  # rightward / leftward - need to check x component
                    return "in" if dot_product > 0 else "out"
            else:  # vertical
                if direction == "rightward":
                    return "in" if dot_product > 0 else "out"
                elif direction == "leftward":
                    return "in" if dot_product < 0 else "out"
                else:  # upward / downward - need to check y component
                    return "in" if dot_product > 0 else "out"

        return None

    def _segments_intersect(self, p1: tuple, p2: tuple, p3: tuple, p4: tuple) -> bool:
        """Check if line segment p1-p2 intersects with line segment p3-p4

        Uses CCW (counter-clockwise) method for robust intersection detection

        Args:
            p1, p2: First segment endpoints
            p3, p4: Second segment endpoints

        Returns:
            True if segments intersect
        """

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def process(self, detections: List[Detection], frame_idx: int) -> FeatureResult:
        """Process detections for line crossing

        Args:
            detections: List of tracked detections
            frame_idx: Current frame index

        Returns:
            FeatureResult with crossing metrics
        """
        self.frame_count += 1
        current_alerts = []

        # Clean up lost tracks
        lost_tracks = [
            tid
            for tid, last_seen in self.track_last_seen.items()
            if frame_idx - last_seen > self.lost_threshold
        ]
        for tid in lost_tracks:
            del self.track_positions[tid]
            del self.track_last_seen[tid]

        # Process each detection
        for det in detections:
            track_id = det.track_id
            centroid = self._get_centroid(det)

            # Update tracking
            if track_id not in self.track_positions:
                self.track_positions[track_id] = []

            # IMPORTANT: Save previous position BEFORE appending new one
            has_prev = len(self.track_positions[track_id]) > 0
            prev_point = self.track_positions[track_id][-1] if has_prev else None

            # If the track appears to "teleport" (large jump), reset its history
            # so we do not connect two different physical people with one ID.
            if has_prev and prev_point is not None:
                dx = centroid[0] - prev_point[0]
                dy = centroid[1] - prev_point[1]
                jump_distance = (dx * dx + dy * dy) ** 0.5
                if jump_distance > self.max_position_jump:
                    # Clear previous positions for this track
                    self.track_positions[track_id] = []
                    prev_point = None
                    has_prev = False
                    # Also remove this track from any "already crossed" sets so
                    # it can be counted correctly as a fresh path.
                    for line_id in list(self.crossed_tracks.keys()):
                        if track_id in self.crossed_tracks[line_id]:
                            self.crossed_tracks[line_id].discard(track_id)

            # Now append new position
            self.track_positions[track_id].append(centroid)
            curr_point = centroid
            self.track_last_seen[track_id] = frame_idx

            # Keep only last N positions
            if len(self.track_positions[track_id]) > 10:
                self.track_positions[track_id] = self.track_positions[track_id][-10:]

            # Check line crossings
            if has_prev and prev_point is not None:
                for line_idx, line in enumerate(self.lines):
                    line_id = self._get_line_attr(line, "id", line_idx)
                    coords = self._get_line_attr(line, "coords", [])
                    if len(coords) < 2:
                        continue

                    # Handle both dict coords and CoordPoint objects
                    if isinstance(coords[0], dict):
                        line_start = (coords[0]["x"], coords[0]["y"])
                        line_end = (coords[1]["x"], coords[1]["y"])
                    else:
                        # CoordPoint dataclass
                        line_start = (coords[0].x, coords[0].y)
                        line_end = (coords[1].x, coords[1].y)

                    # Check if recounting is disabled and already counted
                    if not self.allow_recounting and track_id in self.crossed_tracks[line_id]:
                        continue

                    # Check crossing (pass line config for direction info)
                    direction = self._check_line_crossing(
                        prev_point, curr_point, line_start, line_end, line
                    )

                    if direction:
                        # Mark as crossed
                        self.crossed_tracks[line_id].add(track_id)

                        # Update counters
                        if direction == "in":
                            self.in_counts[line_id] += 1
                            self.total_in += 1

                            # Check alert threshold
                            if self.in_counts[line_id] >= self.in_threshold:
                                current_alerts.append(
                                    {
                                        "type": "line_crossing_in",
                                        "line_id": line_id,
                                        "count": self.in_counts[line_id],
                                        "threshold": self.in_threshold,
                                        "frame": frame_idx,
                                        "track_id": track_id,
                                    }
                                )

                        elif direction == "out":
                            self.out_counts[line_id] += 1
                            self.total_out += 1

                            # Check alert threshold
                            if self.out_counts[line_id] >= self.out_threshold:
                                current_alerts.append(
                                    {
                                        "type": "line_crossing_out",
                                        "line_id": line_id,
                                        "count": self.out_counts[line_id],
                                        "threshold": self.out_threshold,
                                        "frame": frame_idx,
                                        "track_id": track_id,
                                    }
                                )

        self.alerts.extend(current_alerts)

        return FeatureResult(
            feature_type="line_cross",
            detections=detections,
            metrics=self.get_metrics(),
            alerts=current_alerts,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get line crossing metrics

        Returns:
            Dict with in/out counts per line and totals
        """
        line_metrics = {}
        for line_idx, line in enumerate(self.lines):
            line_id = self._get_line_attr(line, "id", line_idx)
            line_metrics[f"line_{line_id}"] = {
                "in_count": self.in_counts[line_id],
                "out_count": self.out_counts[line_id],
                "net_count": self.in_counts[line_id] - self.out_counts[line_id],
            }

        return {
            "feature": "line_cross",
            "total_in": self.total_in,
            "total_out": self.total_out,
            "net_count": self.total_in - self.total_out,
            "lines": line_metrics,
            "active_tracks": len(self.track_positions),
            "alerts_count": len(self.alerts),
        }

    def reset(self):
        """Reset all counters and state"""
        self.track_positions.clear()
        self.track_last_seen.clear()
        self.crossed_tracks.clear()
        self.in_counts.clear()
        self.out_counts.clear()
        self.total_in = 0
        self.total_out = 0
        self.alerts.clear()
        self.frame_count = 0
