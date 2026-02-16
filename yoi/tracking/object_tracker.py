"""Object tracker for detections across frames.

Uses ByteTrack when available, with automatic fallback to centroid matching.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from yoi.tracking.reid_service import LightweightReIDService
from yoi.utils.logger import logger_service

try:
    from ultralytics.trackers.byte_tracker import BYTETracker as _BYTETrackerClass

    HAS_BYTETRACK = True
except Exception:
    _BYTETrackerClass = None  # type: ignore[assignment]
    HAS_BYTETRACK = False


class _ByteTrackDetections:
    """Minimal detection container compatible with Ultralytics BYTETracker."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    @property
    def xywh(self) -> np.ndarray:
        if len(self.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        x1 = self.xyxy[:, 0]
        y1 = self.xyxy[:, 1]
        x2 = self.xyxy[:, 2]
        y2 = self.xyxy[:, 3]
        return np.stack(
            [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                x2 - x1,
                y2 - y1,
            ],
            axis=1,
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask):
        return _ByteTrackDetections(self.xyxy[mask], self.conf[mask], self.cls[mask])


@dataclass
class TrackedObject:
    """Single tracked object."""

    track_id: int
    class_name: str
    history: List[Tuple[float, float]]
    frame_indices: List[int]
    last_frame_idx: int
    confidence_history: List[float]

    @property
    def current_center(self) -> Tuple[float, float]:
        """Get latest position."""
        return self.history[-1] if self.history else (0, 0)

    @property
    def frames_alive(self) -> int:
        """Number of frames object has been tracked."""
        return len(self.frame_indices)

    @property
    def dwell_time_sec(self, fps: float = 30) -> float:
        """Estimated dwell time in seconds."""
        if self.frames_alive == 0:
            return 0
        return self.frames_alive / fps

    def is_active(self, current_frame: int, max_lost_frames: int) -> bool:
        """Check if track is still active."""
        return (current_frame - self.last_frame_idx) <= max_lost_frames


class ObjectTracker:
    """Object tracker with ByteTrack-first strategy and centroid fallback."""

    def __init__(
        self,
        max_lost_frames: int = 30,
        max_distance: float = 50.0,
        tracker_impl: Optional[str] = None,
        bt_track_high_thresh: Optional[float] = None,
        bt_track_low_thresh: Optional[float] = None,
        bt_new_track_thresh: Optional[float] = None,
        bt_match_thresh: Optional[float] = None,
        bt_track_buffer: Optional[int] = None,
        bt_fuse_score: Optional[bool] = None,
        reid_enabled: Optional[bool] = None,
        reid_similarity_thresh: Optional[float] = None,
        reid_momentum: Optional[float] = None,
    ):
        """Initialize tracker."""
        self.max_lost_frames = max_lost_frames
        self.max_distance = max_distance
        self.logger = logger_service.get_analytics_logger()

        self.tracks: Dict[int, TrackedObject] = {}
        self._track_embeddings: Dict[int, np.ndarray] = {}
        self._byte_to_stable_track: Dict[int, int] = {}
        self.next_track_id = 1
        self.frame_idx = 0

        self._reid_enabled = (
            bool(reid_enabled)
            if reid_enabled is not None
            else (os.getenv("YOI_REID_ENABLED", "0").strip().lower() in {"1", "true", "on", "yes"})
        )
        self._reid_similarity_thresh = (
            float(reid_similarity_thresh)
            if reid_similarity_thresh is not None
            else float(os.getenv("YOI_REID_SIMILARITY_THRESH", "0.82"))
        )
        self._reid_momentum = (
            float(reid_momentum)
            if reid_momentum is not None
            else float(os.getenv("YOI_REID_MOMENTUM", "0.35"))
        )
        self._reid_service = LightweightReIDService() if self._reid_enabled else None

        if tracker_impl:
            self._tracker_impl = str(tracker_impl).strip().lower()
        else:
            self._tracker_impl = (
                (os.getenv("YOI_TRACKER_IMPL", "bytetrack") or "bytetrack").strip().lower()
            )
        self._byte_tracker = None

        if self._tracker_impl == "bytetrack" and HAS_BYTETRACK:
            track_high_thresh = (
                float(bt_track_high_thresh)
                if bt_track_high_thresh is not None
                else float(os.getenv("YOI_BT_TRACK_HIGH_THRESH", "0.5"))
            )
            track_low_thresh = (
                float(bt_track_low_thresh)
                if bt_track_low_thresh is not None
                else float(os.getenv("YOI_BT_TRACK_LOW_THRESH", "0.1"))
            )
            new_track_thresh = (
                float(bt_new_track_thresh)
                if bt_new_track_thresh is not None
                else float(os.getenv("YOI_BT_NEW_TRACK_THRESH", "0.6"))
            )
            match_thresh = (
                float(bt_match_thresh)
                if bt_match_thresh is not None
                else float(os.getenv("YOI_BT_MATCH_THRESH", "0.8"))
            )
            track_buffer = (
                max(1, int(bt_track_buffer))
                if bt_track_buffer is not None
                else max(1, self.max_lost_frames)
            )
            fuse_score = (
                bool(bt_fuse_score)
                if bt_fuse_score is not None
                else (
                    os.getenv("YOI_BT_FUSE_SCORE", "1").strip().lower()
                    not in {"0", "false", "off", "no"}
                )
            )

            bt_args = SimpleNamespace(
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                fuse_score=fuse_score,
            )
            try:
                if _BYTETrackerClass is None:
                    raise RuntimeError("BYTETracker class not available")
                self._byte_tracker = _BYTETrackerClass(args=bt_args, frame_rate=30)
                self.logger.info("Tracker initialized with ByteTrack")
                if self._reid_service is not None:
                    self.logger.info(
                        "ReID enabled (similarity>=%.2f, momentum=%.2f)",
                        self._reid_similarity_thresh,
                        self._reid_momentum,
                    )
            except Exception as exc:
                self._byte_tracker = None
                self.logger.warning(
                    f"Failed to initialize ByteTrack, falling back to centroid tracker: {exc}"
                )
        else:
            if self._tracker_impl == "bytetrack" and not HAS_BYTETRACK:
                self.logger.warning("ByteTrack requested but unavailable, using centroid tracker")
            self.logger.info("Tracker initialized with centroid fallback")
            if self._reid_service is not None:
                self.logger.info(
                    "ReID enabled (similarity>=%.2f, momentum=%.2f)",
                    self._reid_similarity_thresh,
                    self._reid_momentum,
                )

    def _extract_reid_embedding(self, frame: Optional[np.ndarray], det) -> Optional[np.ndarray]:
        if self._reid_service is None or frame is None:
            return None
        return self._reid_service.extract_embedding(
            frame,
            float(det.x1),
            float(det.y1),
            float(det.x2),
            float(det.y2),
        )

    def _update_track_embedding(
        self, stable_track_id: int, embedding: Optional[np.ndarray]
    ) -> None:
        if self._reid_service is None or embedding is None:
            return
        current = self._track_embeddings.get(stable_track_id)
        self._track_embeddings[stable_track_id] = self._reid_service.update_running_embedding(
            current=current,
            new_value=embedding,
            momentum=self._reid_momentum,
        )

    def _assign_stable_track_id(
        self,
        byte_track_id: int,
        class_name: str,
        center_x: float,
        center_y: float,
        embedding: Optional[np.ndarray],
    ) -> int:
        mapped = self._byte_to_stable_track.get(byte_track_id)
        if mapped is not None:
            return mapped

        best_track_id = None
        best_similarity = -1.0

        if self._reid_service is not None and embedding is not None:
            for track_id, track in self.tracks.items():
                if track.class_name != class_name:
                    continue
                if track.is_active(self.frame_idx, self.max_lost_frames):
                    continue
                existing_embedding = self._track_embeddings.get(track_id)
                if existing_embedding is None:
                    continue
                similarity = self._reid_service.cosine_similarity(embedding, existing_embedding)
                if similarity >= self._reid_similarity_thresh and similarity > best_similarity:
                    best_similarity = similarity
                    best_track_id = track_id

        if best_track_id is not None:
            self._byte_to_stable_track[byte_track_id] = best_track_id
            self.logger.info(
                "ReID remap: byte_track=%s -> stable_track=%s (sim=%.3f)",
                byte_track_id,
                best_track_id,
                best_similarity,
            )
            return best_track_id

        stable_track_id = self.next_track_id
        self.next_track_id += 1
        self._byte_to_stable_track[byte_track_id] = stable_track_id
        self.tracks[stable_track_id] = TrackedObject(
            track_id=stable_track_id,
            class_name=class_name,
            history=[(center_x, center_y)],
            frame_indices=[self.frame_idx],
            last_frame_idx=self.frame_idx,
            confidence_history=[],
        )
        return stable_track_id

    def _to_bytetrack_results(self, detections: List) -> _ByteTrackDetections:
        xyxy = []
        conf = []
        cls = []
        for det in detections:
            xyxy.append([float(det.x1), float(det.y1), float(det.x2), float(det.y2)])
            conf.append(float(det.confidence))
            cls.append(float(det.class_id))

        if not xyxy:
            return _ByteTrackDetections(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        return _ByteTrackDetections(
            np.asarray(xyxy, dtype=np.float32),
            np.asarray(conf, dtype=np.float32),
            np.asarray(cls, dtype=np.float32),
        )

    def _update_with_bytetrack(
        self, detections: List, frame: Optional[np.ndarray] = None
    ) -> Dict[int, Tuple]:
        if self._byte_tracker is None:
            return {}

        bt_results = self._to_bytetrack_results(detections)
        tracked = self._byte_tracker.update(bt_results)

        class_id_to_name = {int(det.class_id): det.class_name for det in detections}

        active_stable_ids = set()
        active_byte_ids = set()
        for row in tracked:
            row = np.asarray(row, dtype=np.float32)
            det_idx = None
            if row.shape[0] == 8:
                x1, y1, x2, y2, track_id, score, cls_id, det_idx = row.tolist()
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
            elif row.shape[0] == 9:
                center_x, center_y, _w, _h, _angle, track_id, score, cls_id, det_idx = row.tolist()
            else:
                continue

            normalized_byte_track_id = int(track_id)
            class_id = int(cls_id)
            class_name = class_id_to_name.get(class_id, str(class_id))
            active_byte_ids.add(normalized_byte_track_id)

            embedding = None
            if det_idx is not None and self._reid_service is not None and frame is not None:
                det_index = int(det_idx)
                if 0 <= det_index < len(detections):
                    embedding = self._extract_reid_embedding(frame, detections[det_index])

            stable_track_id = self._assign_stable_track_id(
                byte_track_id=normalized_byte_track_id,
                class_name=class_name,
                center_x=center_x,
                center_y=center_y,
                embedding=embedding,
            )
            active_stable_ids.add(stable_track_id)

            self._update_track_embedding(stable_track_id, embedding)

            existing = self.tracks.get(stable_track_id)
            if existing is None:
                self.tracks[stable_track_id] = TrackedObject(
                    track_id=stable_track_id,
                    class_name=class_name,
                    history=[(center_x, center_y)],
                    frame_indices=[self.frame_idx],
                    last_frame_idx=self.frame_idx,
                    confidence_history=[float(score)],
                )
            else:
                existing.class_name = class_name
                existing.history.append((center_x, center_y))
                existing.frame_indices.append(self.frame_idx)
                existing.last_frame_idx = self.frame_idx
                existing.confidence_history.append(float(score))

        # Drop stale tracks from local cache to keep interface behavior consistent.
        for track_id, track in list(self.tracks.items()):
            if track_id not in active_stable_ids and not track.is_active(
                self.frame_idx, self.max_lost_frames
            ):
                del self.tracks[track_id]
                self._track_embeddings.pop(track_id, None)

        for byte_track_id, stable_track_id in list(self._byte_to_stable_track.items()):
            if byte_track_id not in active_byte_ids and stable_track_id not in self.tracks:
                del self._byte_to_stable_track[byte_track_id]

        if self.tracks:
            self.next_track_id = max(self.next_track_id, max(self.tracks.keys()) + 1)

        result = {}
        for track_id, track in self.tracks.items():
            if track.last_frame_idx == self.frame_idx:
                x, y = track.current_center
                result[track_id] = (x, y, track.class_name)

        return result

    def _update_with_centroid(
        self, detections: List, frame: Optional[np.ndarray] = None
    ) -> Dict[int, Tuple]:
        """Legacy centroid tracking update."""
        centroids = {}
        for det in detections:
            if det.class_name not in centroids:
                centroids[det.class_name] = []
            embedding = self._extract_reid_embedding(frame, det)
            centroids[det.class_name].append((det.centroid_x, det.centroid_y, det, embedding))

        for class_name, class_detections in centroids.items():
            active_tracks = {
                tid: track
                for tid, track in self.tracks.items()
                if (
                    track.class_name == class_name
                    and track.is_active(self.frame_idx, self.max_lost_frames)
                )
            }

            matched_tracks = set()
            used_detections = set()

            for detection_idx, (det_x, det_y, det_obj, det_embedding) in enumerate(
                class_detections
            ):
                best_track_id = None
                best_match_score = -1.0

                for track_id, track in active_tracks.items():
                    if track_id in matched_tracks:
                        continue

                    track_x, track_y = track.current_center
                    distance = np.sqrt((det_x - track_x) ** 2 + (det_y - track_y) ** 2)

                    within_distance_gate = distance <= self.max_distance
                    distance_score = max(0.0, 1.0 - (distance / max(self.max_distance, 1e-6)))

                    reid_score = 0.0
                    if self._reid_service is not None and det_embedding is not None:
                        track_embedding = self._track_embeddings.get(track_id)
                        if track_embedding is not None:
                            reid_score = self._reid_service.cosine_similarity(
                                det_embedding, track_embedding
                            )

                    if not within_distance_gate:
                        if self._reid_service is None or det_embedding is None:
                            continue
                        # Allow wider gate only when appearance strongly matches.
                        if (
                            distance > (self.max_distance * 2.0)
                            or reid_score < self._reid_similarity_thresh
                        ):
                            continue

                    match_score = (0.65 * distance_score) + (0.35 * reid_score)
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_track_id = track_id

                if best_track_id is not None:
                    track = self.tracks[best_track_id]
                    track.history.append((det_x, det_y))
                    track.frame_indices.append(self.frame_idx)
                    track.last_frame_idx = self.frame_idx
                    track.confidence_history.append(det_obj.confidence)
                    self._update_track_embedding(best_track_id, det_embedding)
                    matched_tracks.add(best_track_id)
                    used_detections.add(detection_idx)

            for detection_idx, (det_x, det_y, det_obj, det_embedding) in enumerate(
                class_detections
            ):
                if detection_idx not in used_detections:
                    new_track = TrackedObject(
                        track_id=self.next_track_id,
                        class_name=class_name,
                        history=[(det_x, det_y)],
                        frame_indices=[self.frame_idx],
                        last_frame_idx=self.frame_idx,
                        confidence_history=[det_obj.confidence],
                    )
                    self.tracks[self.next_track_id] = new_track
                    self._update_track_embedding(self.next_track_id, det_embedding)
                    self.next_track_id += 1

        for track_id, track in list(self.tracks.items()):
            if not track.is_active(self.frame_idx, self.max_lost_frames):
                del self.tracks[track_id]
                self._track_embeddings.pop(track_id, None)

        result = {}
        for track_id, track in self.tracks.items():
            x, y = track.current_center
            result[track_id] = (x, y, track.class_name)

        return result

    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> Dict[int, Tuple]:
        """Update tracker with current frame detections."""
        if self._byte_tracker is not None:
            result = self._update_with_bytetrack(detections, frame)
        else:
            result = self._update_with_centroid(detections, frame)

        self.frame_idx += 1
        return result

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Get specific track by ID."""
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> Dict[int, TrackedObject]:
        """Get all active tracks."""
        return self.tracks.copy()

    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            "total_tracks": self.next_track_id - 1,
            "active_tracks": len(self.tracks),
            "current_frame": self.frame_idx,
            "max_distance": self.max_distance,
            "max_lost_frames": self.max_lost_frames,
            "reid_enabled": self._reid_service is not None,
        }
