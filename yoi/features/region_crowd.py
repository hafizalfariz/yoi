"""Region Crowd Feature

Count objects within defined regions (people counting in areas).
"""

from collections import defaultdict
from typing import Any, Dict, List

from .base import BaseFeature, Detection, FeatureResult


class RegionCrowdFeature(BaseFeature):
    """Region crowd counting feature

    Counts objects inside defined polygonal regions.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize region crowd feature

        Config expected:
            regions: List of region definitions with coords
            centroid: 'head', 'bottom', 'mid_centre'
            alert_threshold: Number of people to trigger alert
            cooldown_seconds: Time between alerts
        """
        super().__init__(config)

        self.regions = config.get("regions", [])
        self.centroid_mode = config.get("centroid", "mid_centre")
        self.alert_threshold = config.get("alert_threshold", 10)
        self.warning_threshold = config.get("warning_threshold", self.alert_threshold)
        self.critical_threshold = config.get(
            "critical_threshold",
            max(self.warning_threshold + 1, self.alert_threshold),
        )
        self.cooldown_frames = config.get("cooldown_seconds", 5) * config.get("fps", 30)

        # State
        self.current_counts = defaultdict(int)
        self.max_counts = defaultdict(int)
        self.last_alert_frame = defaultdict(int)
        self.tracks_in_region = defaultdict(set)  # region_id -> set of track_ids

    @staticmethod
    def _get_region_attr(region: Any, attr_name: str, default=None):
        if isinstance(region, dict):
            return region.get(attr_name, default)
        return getattr(region, attr_name, default)

    def _get_centroid(self, detection: Detection) -> tuple:
        """Get centroid point based on mode"""
        x1, y1, x2, y2 = detection.bbox

        if self.centroid_mode == "head":
            return ((x1 + x2) / 2, y1)
        elif self.centroid_mode == "bottom":
            return ((x1 + x2) / 2, y2)
        else:  # mid_centre
            return ((x1 + x2) / 2, (y1 + y2) / 2)

    def process(self, detections: List[Detection], frame_idx: int) -> FeatureResult:
        """Process detections for region counting

        Args:
            detections: List of tracked detections
            frame_idx: Current frame index

        Returns:
            FeatureResult with region counts
        """
        self.frame_count += 1
        current_alerts = []

        # Reset current counts
        for region_idx in range(len(self.regions)):
            self.current_counts[region_idx] = 0
            self.tracks_in_region[region_idx] = set()

        # Count detections in each region
        for det in detections:
            centroid = self._get_centroid(det)

            for region_idx, region in enumerate(self.regions):
                coords = self._get_region_attr(region, "coords", [])
                if len(coords) < 3:
                    continue

                # Convert coords to polygon
                if isinstance(coords[0], dict):
                    polygon = [(c["x"], c["y"]) for c in coords]
                else:
                    polygon = [(c.x, c.y) for c in coords]

                # Check if centroid is in region
                if self._check_point_in_polygon(centroid, polygon):
                    self.current_counts[region_idx] += 1
                    self.tracks_in_region[region_idx].add(det.track_id)

                    # Update max count
                    if self.current_counts[region_idx] > self.max_counts[region_idx]:
                        self.max_counts[region_idx] = self.current_counts[region_idx]

                    # Check alert threshold with cooldown
                    if (
                        self.current_counts[region_idx] >= self.alert_threshold
                        and frame_idx - self.last_alert_frame[region_idx] >= self.cooldown_frames
                    ):
                        current_alerts.append(
                            {
                                "type": "region_crowd_alert",
                                "region_id": self._get_region_attr(region, "id", region_idx),
                                "count": self.current_counts[region_idx],
                                "threshold": self.alert_threshold,
                                "frame": frame_idx,
                            }
                        )
                        self.last_alert_frame[region_idx] = frame_idx

        self.alerts.extend(current_alerts)

        return FeatureResult(
            feature_type="region_crowd",
            detections=detections,
            metrics=self.get_metrics(),
            alerts=current_alerts,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get region crowd metrics

        Returns:
            Dict with current and max counts per region
        """
        region_metrics = {}
        for region_idx, region in enumerate(self.regions):
            region_id = self._get_region_attr(region, "id", region_idx)
            current_count = self.current_counts[region_idx]
            status = "normal"
            if current_count >= self.critical_threshold:
                status = "critical"
            elif current_count >= self.warning_threshold:
                status = "warning"

            region_metrics[f"region_{region_id}"] = {
                "current_count": current_count,
                "max_count": self.max_counts[region_idx],
                "active_tracks": len(self.tracks_in_region[region_idx]),
                "status": status,
            }

        total_current = sum(self.current_counts.values())
        total_max = sum(self.max_counts.values())
        inside_track_ids = sorted(
            {
                int(track_id)
                for track_set in self.tracks_in_region.values()
                for track_id in track_set
            }
        )

        return {
            "feature": "region_crowd",
            "total_current": total_current,
            "total_max": total_max,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "inside_track_ids": inside_track_ids,
            "regions": region_metrics,
            "alerts_count": len(self.alerts),
        }

    def reset(self):
        """Reset all counters and state"""
        self.current_counts.clear()
        self.max_counts.clear()
        self.last_alert_frame.clear()
        self.tracks_in_region.clear()
        self.alerts.clear()
        self.frame_count = 0
