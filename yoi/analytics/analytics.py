"""Analytics engine for dwell-time and object counting."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from yoi.utils.logger import logger_service


@dataclass
class DwellTimeAnalytics:
    """Dwell-time analytics for a single object."""

    track_id: int
    class_name: str
    entry_frame: int
    exit_frame: Optional[int] = None
    dwell_time_frames: int = 0
    dwell_time_sec: float = 0.0
    entry_position: Optional[Tuple[float, float]] = None
    exit_position: Optional[Tuple[float, float]] = None
    max_confidence: float = 0.0
    avg_confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "entry_frame": self.entry_frame,
            "exit_frame": self.exit_frame,
            "dwell_time_frames": self.dwell_time_frames,
            "dwell_time_sec": self.dwell_time_sec,
            "entry_position": self.entry_position,
            "exit_position": self.exit_position,
            "max_confidence": self.max_confidence,
            "avg_confidence": self.avg_confidence,
        }


@dataclass
class FrameAnalytics:
    """Analytics results for a single frame."""

    frame_idx: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    object_count: int = 0
    object_count_by_class: Dict[str, int] = field(default_factory=dict)
    active_tracks: Dict[int, Tuple] = field(default_factory=dict)
    dwell_events: List[DwellTimeAnalytics] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp.isoformat(),
            "object_count": self.object_count,
            "object_count_by_class": self.object_count_by_class,
            "active_tracks": {str(k): v for k, v in self.active_tracks.items()},
            "dwell_events": [e.to_dict() for e in self.dwell_events],
        }


class AnalyticsEngine:
    """Main analytics engine for detections and tracks."""

    def __init__(self, fps: float = 30, min_dwell_sec: float = 5.0):
        """
        Initialize analytics engine

        Args:
            fps: Frames per second
            min_dwell_sec: Minimum dwell time threshold
        """
        self.fps = fps
        self.min_dwell_sec = min_dwell_sec
        self.logger = logger_service.get_analytics_logger()

        self.completed_tracks: Dict[int, DwellTimeAnalytics] = {}
        self.frame_idx = 0

    def process_frame(
        self,
        frame_idx: int,
        tracked_objects: Dict[int, Tuple],
        previous_tracks: Dict[int, Tuple],
        tracker_obj,
    ) -> FrameAnalytics:
        """
        Process a single frame for analytics.

        Args:
            frame_idx: Frame index
            tracked_objects: Current tracked objects {id: (x, y, class)}
            previous_tracks: Previous frame tracks
            tracker_obj: Tracker object for detailed info

        Returns:
            FrameAnalytics object
        """
        analytics = FrameAnalytics(frame_idx=frame_idx)

        # Count objects by class
        analytics.object_count = len(tracked_objects)
        for track_id, (x, y, class_name) in tracked_objects.items():
            analytics.object_count_by_class[class_name] = (
                analytics.object_count_by_class.get(class_name, 0) + 1
            )
            analytics.active_tracks[track_id] = (x, y, class_name)

        # Check completed tracks (dwell-time events).
        if hasattr(tracker_obj, "tracks"):
            for track_id, track_obj in tracker_obj.tracks.items():
                if track_id not in previous_tracks and track_id in tracked_objects:
                    # New track detected
                    pass

            # Check tracks that disappeared.
            for prev_track_id in previous_tracks:
                if prev_track_id not in tracked_objects:
                    # Track lost - calculate dwell time
                    if hasattr(tracker_obj, "get_track"):
                        track = tracker_obj.get_track(prev_track_id)
                        if track and track.frames_alive >= (self.min_dwell_sec * self.fps):
                            dwell_analytics = self._create_dwell_analytics(track, frame_idx)
                            analytics.dwell_events.append(dwell_analytics)
                            self.completed_tracks[prev_track_id] = dwell_analytics

        self.frame_idx = frame_idx
        return analytics

    def _create_dwell_analytics(self, tracked_obj, end_frame: int) -> DwellTimeAnalytics:
        """Create dwell-time analytics from a tracked object."""
        dwell_frames = tracked_obj.frames_alive
        dwell_sec = dwell_frames / self.fps

        entry_pos = tracked_obj.history[0] if tracked_obj.history else None
        exit_pos = tracked_obj.history[-1] if tracked_obj.history else None

        max_conf = max(tracked_obj.confidence_history) if tracked_obj.confidence_history else 0
        avg_conf = (
            sum(tracked_obj.confidence_history) / len(tracked_obj.confidence_history)
            if tracked_obj.confidence_history
            else 0
        )

        return DwellTimeAnalytics(
            track_id=tracked_obj.track_id,
            class_name=tracked_obj.class_name,
            entry_frame=tracked_obj.frame_indices[0],
            exit_frame=end_frame,
            dwell_time_frames=dwell_frames,
            dwell_time_sec=dwell_sec,
            entry_position=entry_pos,
            exit_position=exit_pos,
            max_confidence=max_conf,
            avg_confidence=avg_conf,
        )

    def get_dwell_time_summary(self) -> List[DwellTimeAnalytics]:
        """Get summary of completed dwell-time tracks."""
        return list(self.completed_tracks.values())

    def get_long_dwellers(self, threshold_sec: float) -> List[DwellTimeAnalytics]:
        """Get objects with dwell time above threshold."""
        return [dt for dt in self.completed_tracks.values() if dt.dwell_time_sec >= threshold_sec]

    def export_summaries(self, output_dir: str) -> Dict[str, str]:
        """
        Export analytics summaries

        Args:
            output_dir: Output directory

        Returns:
            Dict with exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export JSON
        json_file = output_path / "analytics_summary.json"
        json_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_objects": len(self.completed_tracks),
            "dwell_time_analytics": [dt.to_dict() for dt in self.completed_tracks.values()],
        }
        json_file.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))

        # Export CSV
        csv_file = output_path / "analytics_summary.csv"
        csv_lines = ["track_id,class_name,entry_frame,exit_frame,dwell_time_sec,avg_confidence"]
        for dt in self.completed_tracks.values():
            csv_lines.append(
                f"{dt.track_id},{dt.class_name},{dt.entry_frame},"
                f"{dt.exit_frame},{dt.dwell_time_sec:.2f},{dt.avg_confidence:.3f}"
            )
        csv_file.write_text("\n".join(csv_lines))

        self.logger.info(f"Analytics exported to {output_dir}")

        return {
            "json": str(json_file),
            "csv": str(csv_file),
        }
