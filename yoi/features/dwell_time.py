"""Dwell Time Feature

Track how long objects stay in defined regions.
"""

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from .base import BaseFeature, Detection, FeatureResult


class DwellTimeFeature(BaseFeature):
    """Dwell time tracking feature

    Tracks how long objects remain in each region.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize dwell time feature

        Config expected:
            regions: List of region definitions with coords
            centroid: 'head', 'bottom', 'mid_centre'
            min_dwell_time_seconds: Minimum time to count as dwelling
            alert_threshold: Dwell time to trigger alert
            fps: Frames per second for time calculation
        """
        super().__init__(config)

        self.regions = config.get("regions", [])
        self.centroid_mode = config.get("centroid", "mid_centre")
        self.fps = config.get("fps", 30)

        min_dwell_seconds = (
            config.get("min_dwell_time_seconds")
            or config.get("min_dwell_seconds")
            or config.get("min_dwelltime")
            or config.get("min_dwell_time")
            or 3
        )
        alert_threshold_seconds = (
            config.get("alert_threshold_seconds")
            or config.get("alert_threshold")
            or config.get("warning_seconds")
            or 10
        )

        self.min_dwell_frames = float(min_dwell_seconds) * float(self.fps)
        self.alert_threshold_frames = float(alert_threshold_seconds) * float(self.fps)

        # State tracking
        self.track_entry_frame = defaultdict(dict)  # track_id -> {region_id: entry_frame}
        self.track_exit_frame = defaultdict(dict)  # track_id -> {region_id: exit_frame}
        self.dwell_times = defaultdict(list)  # region_id -> [dwell_time_seconds, ...]
        self.current_dwelling = defaultdict(set)  # region_id -> set of track_ids
        self.alerted_tracks = defaultdict(set)  # region_id -> set of track_ids (already alerted)

    @staticmethod
    def _region_id(region: Any, fallback: int) -> Any:
        if isinstance(region, dict):
            return region.get("id", fallback)
        return getattr(region, "id", fallback)

    @staticmethod
    def _region_coords(region: Any) -> List[Any]:
        if isinstance(region, dict):
            return region.get("coords", [])
        return getattr(region, "coords", []) or []

    @staticmethod
    def _point_xy(point: Any) -> tuple[float, float]:
        if isinstance(point, dict):
            return (float(point.get("x", 0.0)), float(point.get("y", 0.0)))
        if is_dataclass(point):
            raw = asdict(point)
            return (float(raw.get("x", 0.0)), float(raw.get("y", 0.0)))
        return (float(getattr(point, "x", 0.0)), float(getattr(point, "y", 0.0)))

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
        """Process detections for dwell time tracking

        Args:
            detections: List of tracked detections
            frame_idx: Current frame index

        Returns:
            FeatureResult with dwell time metrics
        """
        self.frame_count += 1
        current_alerts = []

        # Track which tracks are currently in each region
        current_in_region = defaultdict(set)

        # Process each detection
        for det in detections:
            track_id = det.track_id
            centroid = self._get_centroid(det)

            for region_idx, region in enumerate(self.regions):
                region_id = self._region_id(region, region_idx)
                coords = self._region_coords(region)
                if len(coords) < 3:
                    continue

                polygon = [self._point_xy(coord) for coord in coords]

                if self._check_point_in_polygon(centroid, polygon):
                    current_in_region[region_id].add(track_id)

                    # Track entry
                    if (
                        track_id not in self.track_entry_frame
                        or region_id not in self.track_entry_frame[track_id]
                    ):
                        self.track_entry_frame[track_id][region_id] = frame_idx
                        self.current_dwelling[region_id].add(track_id)

                    # Calculate current dwell time
                    entry_frame = self.track_entry_frame[track_id][region_id]
                    dwell_frames = frame_idx - entry_frame
                    dwell_seconds = dwell_frames / self.fps

                    # Check alert threshold
                    if (
                        dwell_frames >= self.alert_threshold_frames
                        and track_id not in self.alerted_tracks[region_id]
                    ):
                        current_alerts.append(
                            {
                                "type": "dwell_time_alert",
                                "region_id": region_id,
                                "track_id": track_id,
                                "dwell_time_seconds": dwell_seconds,
                                "threshold_seconds": self.alert_threshold_frames / self.fps,
                                "frame": frame_idx,
                            }
                        )
                        self.alerted_tracks[region_id].add(track_id)

        # Check for exits (tracks that were in region but now aren't)
        for region_id in self.current_dwelling.keys():
            exited_tracks = self.current_dwelling[region_id] - current_in_region[region_id]

            for track_id in exited_tracks:
                # Calculate final dwell time
                if (
                    track_id in self.track_entry_frame
                    and region_id in self.track_entry_frame[track_id]
                ):
                    entry_frame = self.track_entry_frame[track_id][region_id]
                    dwell_frames = frame_idx - entry_frame
                    dwell_seconds = dwell_frames / self.fps

                    # Only record if above minimum
                    if dwell_frames >= self.min_dwell_frames:
                        self.dwell_times[region_id].append(dwell_seconds)

                    # Clean up tracking
                    del self.track_entry_frame[track_id][region_id]
                    if track_id in self.alerted_tracks[region_id]:
                        self.alerted_tracks[region_id].remove(track_id)

        # Update current dwelling
        for region_id in self.current_dwelling.keys():
            self.current_dwelling[region_id] = current_in_region[region_id]

        self.alerts.extend(current_alerts)

        return FeatureResult(
            feature_type="dwell_time",
            detections=detections,
            metrics=self.get_metrics(),
            alerts=current_alerts,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get dwell time metrics

        Returns:
            Dict with dwell time statistics per region
        """
        region_metrics = {}
        inside_track_ids = set()
        for region_idx, region in enumerate(self.regions):
            region_id = self._region_id(region, region_idx)
            dwell_list = self.dwell_times[region_id]

            if len(dwell_list) > 0:
                avg_dwell = sum(dwell_list) / len(dwell_list)
                max_dwell = max(dwell_list)
                min_dwell = min(dwell_list)
            else:
                avg_dwell = max_dwell = min_dwell = 0

            # Calculate current dwell times for active tracks
            current_dwells = []
            for track_id in self.current_dwelling[region_id]:
                if (
                    track_id in self.track_entry_frame
                    and region_id in self.track_entry_frame[track_id]
                ):
                    entry_frame = self.track_entry_frame[track_id][region_id]
                    dwell_frames = self.frame_count - entry_frame
                    current_dwells.append(dwell_frames / self.fps)
            inside_track_ids.update(self.current_dwelling[region_id])

            region_metrics[f"region_{region_id}"] = {
                "current_dwelling": len(self.current_dwelling[region_id]),
                "current_dwell_times": current_dwells,
                "total_completed": len(dwell_list),
                "avg_dwell_seconds": round(avg_dwell, 2),
                "max_dwell_seconds": round(max_dwell, 2),
                "min_dwell_seconds": round(min_dwell, 2),
            }

        all_dwells = []
        for dwell_list in self.dwell_times.values():
            all_dwells.extend(dwell_list)

        alerted_track_ids = sorted(
            int(track_id) for track_set in self.alerted_tracks.values() for track_id in track_set
        )

        if len(all_dwells) > 0:
            overall_avg = sum(all_dwells) / len(all_dwells)
            overall_max = max(all_dwells)
        else:
            overall_avg = overall_max = 0

        return {
            "feature": "dwell_time",
            "regions": region_metrics,
            "inside_track_ids": sorted(int(track_id) for track_id in inside_track_ids),
            "alerted_track_ids": alerted_track_ids,
            "overall_avg_dwell_seconds": round(overall_avg, 2),
            "overall_max_dwell_seconds": round(overall_max, 2),
            "total_dwells_recorded": len(all_dwells),
            "alerts_count": len(self.alerts),
        }

    def reset(self):
        """Reset all tracking state"""
        self.track_entry_frame.clear()
        self.track_exit_frame.clear()
        self.dwell_times.clear()
        self.current_dwelling.clear()
        self.alerted_tracks.clear()
        self.alerts.clear()
        self.frame_count = 0
