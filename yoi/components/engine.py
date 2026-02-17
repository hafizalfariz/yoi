"""Main Vision Engine orchestrating video processing and analytics."""

import os
import signal
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from yoi.analytics.analytics import AnalyticsEngine
from yoi.components.engine_feature_mapping import build_feature_detections
from yoi.components.engine_output_lifecycle import (
    cleanup_engine,
    handle_feature_alert_events,
    initialize_output_engines,
)
from yoi.components.video_reader import VideoReader
from yoi.config import YOIConfig
from yoi.features import get_feature
from yoi.inference.yolo import YOLOInferencer
from yoi.stream import RTSPPusher
from yoi.tracking.object_tracker import ObjectTracker
from yoi.utils.logger import logger_service


class VisionEngine:
    """Main YOI Vision AI Engine"""

    def __init__(self, config: YOIConfig):
        """
        Initialize vision engine with config.

        Args:
            config: YOIConfig instance
        """
        self.config = config
        self.logger = logger_service.get_engine_logger()

        self.logger.info(f"Initializing YOI Vision Engine: {config.config_name}")
        self.logger.info(f"CCTV ID: {config.cctv_id}")
        self.logger.info(f"Feature: {config.feature}")
        self.logger.info("STAGE 1/6 - ENGINE INIT")

        # Optional RTSP pusher for live annotated stream
        self.rtsp_pusher: Optional[RTSPPusher] = None
        self._rtsp_url: Optional[str] = None
        self.annotator: Any = None
        self.video_writer: Any = None
        self.data_exporter: Any = None
        self.alert_manager: Any = None

        # Initialize components
        self._init_video_reader()
        self.logger.info("STAGE 2/6 - INPUT READY")
        self._init_inference_engine()
        self.logger.info("STAGE 3/6 - MODEL READY")
        self._init_tracker()
        self._init_feature_engine()
        self._init_analytics()
        self._init_output_engines()
        self.logger.info("STAGE 4/6 - OUTPUT/RTSP READY")

        self.frame_count = 0
        self.start_time = None
        self._init_runtime_tunables()
        self._init_runtime_state()
        self._init_rtsp_state()
        self._init_input_loop_mode()
        self._stop_requested = False
        self._stop_reason: Optional[str] = None

    def request_stop(self, reason: str = "external request") -> None:
        """Request graceful stop; processing loop will exit and cleanup will run."""
        if self._stop_requested:
            return
        self._stop_requested = True
        self._stop_reason = reason
        self.logger.warning("Graceful stop requested: %s", reason)

    def _init_runtime_tunables(self) -> None:
        """Initialize non-RTSP runtime tuning parameters."""
        self._log_every_n_frames = 60
        try:
            configured_log_every_n_frames = (
                getattr(self.config.output, "log_every_n_frames", None)
                if self.config.output
                else None
            )
            if configured_log_every_n_frames is not None:
                configured = int(configured_log_every_n_frames)
                if configured > 0:
                    self._log_every_n_frames = configured
        except Exception:
            self._log_every_n_frames = 60

        self._feature_log_every_n_frames = self._env_int(
            "YOI_FEATURE_LOG_EVERY_N_FRAMES",
            default=10,
            min_value=1,
        )
        self._infer_every_n_frames: int = self._env_int(
            "YOI_INFER_EVERY_N_FRAMES",
            default=1,
            min_value=1,
        )
        self._max_inference_seconds: float = self._env_float(
            "YOI_MAX_INFERENCE_SECONDS",
            default=0.0,
            min_value=0.0,
        )
        self._max_inference_runtime_seconds: Optional[float] = None
        if self._max_inference_seconds > 0:
            self._max_inference_runtime_seconds = self._max_inference_seconds
            self.logger.warning(
                "Inference runtime limit enabled: %.2f second(s)",
                self._max_inference_seconds,
            )

        if self._infer_every_n_frames > 1:
            self.logger.warning(
                "Performance mode enabled: inference every %s frame(s)",
                self._infer_every_n_frames,
            )

    def _init_runtime_state(self) -> None:
        """Initialize per-event runtime state."""
        # State for per-event outputs (images / status / CSV)
        self._event_counter: int = 0
        self._last_line_cross_counts: Optional[tuple[int, int, int]] = None
        self._last_feature_signature: Optional[str] = None
        self._track_visual_states: Dict[int, str] = {}

    def _init_rtsp_state(self) -> None:
        """Initialize RTSP health and recovery state."""
        self._rtsp_drop_warn_seconds = self._env_float(
            "YOI_RTSP_DROP_WARN_SECONDS",
            default=10.0,
            min_value=0.1,
        )

        self._rtsp_first_fail_ts: Optional[float] = None
        self._rtsp_drop_warned = False
        self._rtsp_last_health_log_ts: Optional[float] = None
        self._rtsp_push_success_count = 0
        self._rtsp_push_fail_count = 0
        self._rtsp_recover_count = 0
        self._rtsp_last_recover_attempt_ts: Optional[float] = None

        self._rtsp_health_log_interval_seconds = self._env_float(
            "YOI_RTSP_HEALTH_LOG_INTERVAL_SECONDS",
            default=10.0,
            min_value=0.1,
        )

        self._rtsp_auto_recover_enabled = self._env_enabled(
            "YOI_RTSP_AUTO_RECOVER",
            default=True,
        )
        self._rtsp_recover_cooldown_seconds = self._env_float(
            "YOI_RTSP_RECOVER_COOLDOWN_SECONDS",
            default=2.0,
            min_value=0.0,
        )

    def _init_input_loop_mode(self) -> None:
        """Initialize file-input loop behavior."""
        self._loop_file_input = False
        try:
            self._loop_file_input = self._env_enabled("YOI_LOOP_FILE_INPUT", default=False)
        except Exception:
            self._loop_file_input = False

        if (
            self._loop_file_input
            and self.config.input
            and not self.config.input.get_source_path().startswith("rtsp://")
        ):
            self.logger.info("File input loop enabled (YOI_LOOP_FILE_INPUT=1)")

    def _init_video_reader(self):
        """Initialize video reader"""
        if not self.config.input:
            raise ValueError("Input configuration required")

        try:
            source_path = self.config.input.get_source_path()
            self.video_reader = VideoReader.create(
                source=source_path,
                max_fps=self.config.input.max_fps,
                buffer_size=self.config.input.buffer_size,
            )
            self.logger.info(f"Video reader ready: {source_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize video reader: {e}")
            raise

    @staticmethod
    def _env_enabled(name: str, default: bool = True) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "off", "no"}

    @staticmethod
    def _env_int(name: str, default: int, *, min_value: Optional[int] = None) -> int:
        """Read integer env var with optional lower bound."""
        try:
            value = int(os.getenv(name, str(default)))
        except Exception:
            value = default
        if min_value is not None:
            value = max(min_value, value)
        return value

    @staticmethod
    def _env_float(name: str, default: float, *, min_value: Optional[float] = None) -> float:
        """Read float env var with optional lower bound."""
        try:
            value = float(os.getenv(name, str(default)))
        except Exception:
            value = default
        if min_value is not None:
            value = max(min_value, value)
        return value

    def _init_inference_engine(self):
        """Initialize YOLO inference engine"""
        try:
            self.inferencer = YOLOInferencer(
                model_name=self.config.model.name,
                device=self.config.model.device,
                conf=self.config.model.conf,
                iou=self.config.model.iou,
                classes=self.config.model.classes,
            )
            self.logger.info("YOLO inference engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize inference engine: {e}")
            raise

    def _init_tracker(self):
        """Initialize object tracker"""
        # Use feature_params tracking config if available
        tracking_cfg = self.config.tracking
        if self.config.feature_params and self.config.feature_params.tracking:
            tracking_cfg = self.config.feature_params.tracking

        max_lost_frames = int(getattr(tracking_cfg, "max_lost_frames", 30))
        max_distance = float(getattr(tracking_cfg, "max_distance", 50.0))
        tracker_impl = str(getattr(tracking_cfg, "tracker_impl", "bytetrack"))
        bt_track_high_thresh = float(getattr(tracking_cfg, "bt_track_high_thresh", 0.5))
        bt_track_low_thresh = float(getattr(tracking_cfg, "bt_track_low_thresh", 0.1))
        bt_new_track_thresh = float(getattr(tracking_cfg, "bt_new_track_thresh", 0.6))
        bt_match_thresh = float(getattr(tracking_cfg, "bt_match_thresh", 0.8))
        bt_track_buffer_raw = getattr(tracking_cfg, "bt_track_buffer", None)
        bt_track_buffer = int(bt_track_buffer_raw) if bt_track_buffer_raw is not None else None
        bt_fuse_score = bool(getattr(tracking_cfg, "bt_fuse_score", True))
        reid_enabled = bool(getattr(tracking_cfg, "reid_enabled", False))
        reid_similarity_thresh = float(getattr(tracking_cfg, "reid_similarity_thresh", 0.82))
        reid_momentum = float(getattr(tracking_cfg, "reid_momentum", 0.35))

        self.tracker = ObjectTracker(
            max_lost_frames=max_lost_frames,
            max_distance=max_distance,
            tracker_impl=tracker_impl,
            bt_track_high_thresh=bt_track_high_thresh,
            bt_track_low_thresh=bt_track_low_thresh,
            bt_new_track_thresh=bt_new_track_thresh,
            bt_match_thresh=bt_match_thresh,
            bt_track_buffer=bt_track_buffer,
            bt_fuse_score=bt_fuse_score,
            reid_enabled=reid_enabled,
            reid_similarity_thresh=reid_similarity_thresh,
            reid_momentum=reid_momentum,
        )
        self.logger.info("Object tracker initialized")

    def _init_feature_engine(self):
        """Initialize feature engine (line_cross, region_crowd, etc.)"""
        if not self.config.feature:
            self.feature_engine = None
            self.logger.info("No feature specified, running basic detection only")
            return

        # Build feature config
        feature_config = {
            "feature_name": self.config.feature,
            "lines": self.config.lines or [],
            "regions": self.config.regions or [],
        }

        # Add feature params if available
        if self.config.feature_params:
            raw_params: Dict[str, Any] = {}
            if is_dataclass(self.config.feature_params):
                raw_params = asdict(self.config.feature_params)
            elif isinstance(self.config.feature_params, dict):
                raw_params = dict(self.config.feature_params)
            elif hasattr(self.config.feature_params, "to_dict"):
                raw_params = self.config.feature_params.to_dict()  # type: ignore[assignment]

            extra_params = raw_params.pop("extra", {}) if isinstance(raw_params, dict) else {}
            filtered_params = {key: value for key, value in raw_params.items() if value is not None}

            feature_config.update(filtered_params)
            if isinstance(extra_params, dict):
                feature_config.update(extra_params)

        try:
            self.feature_engine = get_feature(self.config.feature, feature_config)
            self.logger.info(f"Feature engine initialized: {self.config.feature}")
        except Exception as e:
            self.logger.error(f"Failed to initialize feature engine: {e}", exc_info=True)
            self.feature_engine = None

    def _init_analytics(self):
        """Initialize analytics engine"""
        fps = self.video_reader.get_fps()

        # Use default min dwell or feature params
        min_dwell = 5.0
        if self.config.feature_params and self.config.feature_params.lost_threshold:
            min_dwell = self.config.feature_params.lost_threshold

        self.analytics_engine = AnalyticsEngine(fps=fps, min_dwell_sec=min_dwell)
        self.logger.info(f"Analytics engine initialized (FPS: {fps}, Min dwell: {min_dwell}s)")

    def _init_output_engines(self):
        """Initialize output engines."""
        initialize_output_engines(self)

    def process(self):
        """Process video stream and generate outputs."""
        self.logger.info("Starting video processing...")
        self.logger.info("STAGE 5/6 - PRE-INFERENCE CHECKS")

        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        previous_sigint_handler = signal.getsignal(signal.SIGINT)

        def _engine_signal_handler(signum, _frame):
            try:
                signal_name = signal.Signals(signum).name
            except Exception:
                signal_name = str(signum)
            self.request_stop(f"signal {signal_name}")

        signal.signal(signal.SIGTERM, _engine_signal_handler)
        signal.signal(signal.SIGINT, _engine_signal_handler)

        # Optional cooldown before starting inference so RTSP clients
        # have time to connect to the output stream.
        try:
            rtsp_cooldown = None
            if self.config.output and hasattr(self.config.output, "rtsp_cooldown_seconds"):
                rtsp_cooldown = self.config.output.rtsp_cooldown_seconds

            if self.rtsp_pusher is not None and getattr(self.rtsp_pusher, "is_running", False):
                wait_rtsp_ready = self._env_enabled("YOI_WAIT_RTSP_READY", default=False)
                if wait_rtsp_ready:
                    ready = self._is_rtsp_publisher_ready()
                    if ready:
                        self.logger.info("RTSP OUTPUT: publisher process is ready")
                    else:
                        self.logger.warning(
                            "RTSP OUTPUT: publisher process not ready yet, continuing with cooldown"
                        )

                if rtsp_cooldown is None:
                    # Default delay if RTSP is enabled but cooldown not set
                    rtsp_cooldown = 10

                wait_sec = max(0, int(rtsp_cooldown))
                if wait_sec > 0:
                    self.logger.info(
                        f"RTSP OUTPUT: cooldown {wait_sec}s before inference "
                        "(allow publisher/clients stabilize)"
                    )
                    for _ in range(wait_sec):
                        if self._stop_requested:
                            self.logger.warning(
                                "Stop requested during RTSP cooldown - skipping inference"
                            )
                            break
                        time.sleep(1)
        except Exception as e:
            self.logger.warning(f"Error during RTSP cooldown before processing: {e}")

        self.logger.info("STAGE 6/6 - INFERENCE STARTED")

        self.start_time = time.time()

        frame_generator = VideoReader.create_frame_generator(
            self.video_reader,
            self.config.input.max_fps if self.config.input else None,
            loop_file=self._loop_file_input,
        )

        previous_tracks = {}
        cached_detections = []

        try:
            for frame_idx, frame in frame_generator:
                if self._stop_requested:
                    self.logger.warning(
                        "Stopping processing loop at frame %s (reason: %s)",
                        frame_idx,
                        self._stop_reason or "requested",
                    )
                    break

                if self._max_inference_runtime_seconds is not None and self.start_time is not None:
                    elapsed_seconds = time.time() - self.start_time
                    if elapsed_seconds >= self._max_inference_runtime_seconds:
                        self.request_stop(
                            f"max inference runtime reached ({self._max_inference_seconds:.2f} second(s))"
                        )
                        self.logger.warning(
                            "Stopping processing loop at frame %s due to runtime limit",
                            frame_idx,
                        )
                        break

                # Run inference (optionally frame-skipped for higher throughput)
                should_infer = frame_idx % self._infer_every_n_frames == 0 or not cached_detections
                if should_infer:
                    inference_result = self.inferencer.infer(frame)
                    detections = inference_result.detections
                    cached_detections = detections
                else:
                    detections = cached_detections

                # Update tracker
                tracked_objects = self.tracker.update(detections, frame)

                # Process feature (line-cross, region-crowd, etc.) if configured
                feature_result = None
                track_bbox_map = None
                active_track_ids = set(tracked_objects.keys())
                self._track_visual_states = {
                    track_id: state
                    for track_id, state in self._track_visual_states.items()
                    if track_id in active_track_ids
                }
                track_alert_states: Dict[int, str] = dict(self._track_visual_states)
                if self.feature_engine:
                    feature_detections, track_bbox_map = build_feature_detections(
                        tracker=self.tracker,
                        detections=detections,
                        tracked_objects=tracked_objects,
                        frame_shape=frame.shape,
                    )

                    feature_result = self.feature_engine.process(feature_detections, frame_idx)

                    if feature_result and getattr(feature_result, "alerts", None):
                        for alert in feature_result.alerts:
                            if not isinstance(alert, dict):
                                continue
                            track_id = alert.get("track_id")
                            alert_type = str(alert.get("type", "")).lower()
                            if track_id is None:
                                continue
                            if alert_type.endswith("_in"):
                                normalized_track_id = int(track_id)
                                track_alert_states[normalized_track_id] = "in"
                                self._track_visual_states[normalized_track_id] = "in"
                            elif alert_type.endswith("_out"):
                                normalized_track_id = int(track_id)
                                track_alert_states[normalized_track_id] = "out"
                                self._track_visual_states[normalized_track_id] = "out"

                    # Log feature events (throttled): only when changed or periodic.
                    if feature_result and feature_result.metrics:
                        metrics = feature_result.metrics
                        if metrics.get("feature") == "line_cross":
                            current_counts = (
                                int(metrics.get("total_in", 0)),
                                int(metrics.get("total_out", 0)),
                                int(metrics.get("net_count", 0)),
                            )
                            should_log = (
                                self._last_line_cross_counts != current_counts
                                or frame_idx % self._log_every_n_frames == 0
                            )
                            if should_log:
                                self.logger.info(
                                    "Frame %s - line_cross in=%s out=%s net=%s active=%s",
                                    frame_idx,
                                    current_counts[0],
                                    current_counts[1],
                                    current_counts[2],
                                    int(metrics.get("active_tracks", 0)),
                                )
                                self._last_line_cross_counts = current_counts
                        elif metrics.get("feature") == "region_crowd":
                            inside_track_ids = {
                                int(track_id) for track_id in metrics.get("inside_track_ids", [])
                            }
                            for track_id in active_track_ids:
                                track_alert_states[track_id] = (
                                    "inside" if track_id in inside_track_ids else "outside"
                                )
                            signature = str(
                                (
                                    metrics.get("total_current"),
                                    metrics.get("total_max"),
                                    tuple(sorted(inside_track_ids)),
                                    str(metrics.get("regions", {})),
                                )
                            )
                            should_log = frame_idx % self._feature_log_every_n_frames == 0
                            if should_log:
                                self.logger.info(
                                    "Frame %s - region_crowd current=%s max=%s inside=%s",
                                    frame_idx,
                                    int(metrics.get("total_current", 0)),
                                    int(metrics.get("total_max", 0)),
                                    len(inside_track_ids),
                                )
                                self._last_feature_signature = signature
                        elif metrics.get("feature") == "dwell_time":
                            inside_track_ids = {
                                int(track_id) for track_id in metrics.get("inside_track_ids", [])
                            }
                            alerted_track_ids = {
                                int(track_id) for track_id in metrics.get("alerted_track_ids", [])
                            }
                            for track_id in active_track_ids:
                                if track_id in alerted_track_ids:
                                    track_alert_states[track_id] = "dwell_alert"
                                else:
                                    track_alert_states[track_id] = (
                                        "dwell_inside"
                                        if track_id in inside_track_ids
                                        else "dwell_outside"
                                    )
                            signature = str(
                                (
                                    tuple(sorted(inside_track_ids)),
                                    tuple(sorted(alerted_track_ids)),
                                    str(metrics.get("regions", {})),
                                )
                            )
                            should_log = frame_idx % self._feature_log_every_n_frames == 0
                            if should_log:
                                self.logger.info(
                                    "Frame %s - dwell_time inside=%s alerted=%s max=%.2fs",
                                    frame_idx,
                                    len(inside_track_ids),
                                    len(alerted_track_ids),
                                    float(metrics.get("overall_max_dwell_seconds", 0.0) or 0.0),
                                )
                                self._last_feature_signature = signature
                        else:
                            signature = str(metrics)
                            should_log = frame_idx % self._feature_log_every_n_frames == 0
                            if should_log:
                                self.logger.info(
                                    "Frame %s - feature=%s metrics_update",
                                    frame_idx,
                                    metrics.get("feature", "unknown"),
                                )
                                self._last_feature_signature = signature

                # Run analytics
                analytics_result = self.analytics_engine.process_frame(
                    frame_idx=frame_idx,
                    tracked_objects=tracked_objects,
                    previous_tracks=previous_tracks,
                    tracker_obj=self.tracker,
                )

                annotated_frame = frame.copy()

                # Draw detections and tracking
                if tracked_objects:
                    annotated_frame = self.annotator.draw_tracks(
                        annotated_frame,
                        tracked_objects,
                        track_bbox_map,
                        track_alert_states,
                    )
                else:
                    annotated_frame = self.annotator.draw_boxes(
                        annotated_frame,
                        detections,
                    )

                # Draw lines and feature results (line-cross counts)
                if self.config.lines:
                    annotated_frame = self.annotator.draw_lines(
                        annotated_frame, self.config.lines, feature_result
                    )

                if self.config.regions:
                    annotated_frame = self.annotator.draw_regions(
                        annotated_frame,
                        self.config.regions,
                        feature_result,
                    )

                annotated_frame = self.annotator.draw_analytics(
                    annotated_frame, analytics_result.to_dict()
                )

                # Render current FPS on the annotated frame.
                self.frame_count = frame_idx + 1
                elapsed = time.time() - self.start_time if self.start_time else 0
                current_fps = self.frame_count / elapsed if elapsed > 0 else 0

                annotated_frame = self.annotator.draw_fps(
                    annotated_frame,
                    current_fps,
                )

                # Push annotated frame to RTSP stream if enabled
                if self.rtsp_pusher is not None:
                    if not self.rtsp_pusher.is_running:
                        try:
                            self.logger.warning(
                                "RTSP pusher not running during processing; trying restart"
                            )
                            self.rtsp_pusher.restart()
                            self._rtsp_recover_count += 1
                            self._rtsp_last_recover_attempt_ts = time.time()
                        except Exception as e:
                            self.logger.warning(f"Failed to restart RTSP pusher: {e}")

                    pushed = self.rtsp_pusher.push_frame(annotated_frame)
                    now_ts = time.time()
                    if pushed:
                        self._rtsp_push_success_count += 1
                        if self._rtsp_first_fail_ts is not None and self._rtsp_drop_warned:
                            down_for = now_ts - self._rtsp_first_fail_ts
                            self.logger.info(
                                "RTSP stream recovered after %.1fs downtime (%s)",
                                down_for,
                                self._rtsp_url or "unknown",
                            )
                        self._rtsp_first_fail_ts = None
                        self._rtsp_drop_warned = False
                    else:
                        self._rtsp_push_fail_count += 1
                        if self._rtsp_first_fail_ts is None:
                            self._rtsp_first_fail_ts = now_ts
                        down_for = now_ts - self._rtsp_first_fail_ts
                        if down_for >= self._rtsp_drop_warn_seconds and not self._rtsp_drop_warned:
                            self.logger.warning(
                                "RTSP stream appears down for %.1fs (%s)",
                                down_for,
                                self._rtsp_url or "unknown",
                            )
                            self._rtsp_drop_warned = True

                        if self._rtsp_auto_recover_enabled:
                            should_attempt_recover = (
                                self._rtsp_last_recover_attempt_ts is None
                                or (now_ts - self._rtsp_last_recover_attempt_ts)
                                >= self._rtsp_recover_cooldown_seconds
                            )
                            if should_attempt_recover:
                                try:
                                    self.logger.warning(
                                        "RTSP push failed; attempting pusher recovery (url=%s)",
                                        self._rtsp_url or "unknown",
                                    )
                                    recovered = self.rtsp_pusher.restart()
                                    self._rtsp_last_recover_attempt_ts = now_ts
                                    if recovered:
                                        self._rtsp_recover_count += 1
                                        self.logger.info(
                                            "RTSP pusher recovery succeeded (count=%s, url=%s)",
                                            self._rtsp_recover_count,
                                            self._rtsp_url or "unknown",
                                        )
                                    else:
                                        self.logger.warning(
                                            "RTSP pusher recovery attempt failed (url=%s)",
                                            self._rtsp_url or "unknown",
                                        )
                                except Exception as recover_error:
                                    self._rtsp_last_recover_attempt_ts = now_ts
                                    self.logger.warning(
                                        "RTSP pusher recovery error: %s",
                                        recover_error,
                                    )

                    should_log_health = (
                        self._rtsp_last_health_log_ts is None
                        or (now_ts - self._rtsp_last_health_log_ts)
                        >= self._rtsp_health_log_interval_seconds
                    )
                    if should_log_health:
                        stream_status = "up" if pushed else "down"
                        self.logger.info(
                            "RTSP health [%s]: status=%s success=%s fail=%s recover=%s url=%s",
                            self.config.config_name,
                            stream_status,
                            self._rtsp_push_success_count,
                            self._rtsp_push_fail_count,
                            self._rtsp_recover_count,
                            self._rtsp_url or "unknown",
                        )
                        self._rtsp_last_health_log_ts = now_ts

                    if not pushed and frame_idx < 10:
                        self.logger.warning(
                            f"RTSP push failed at frame {frame_idx}; check yoi.rtsp logs"
                        )

                if feature_result is not None and getattr(feature_result, "metrics", None):
                    metrics: Dict[str, Any] = feature_result.metrics

                    if self.alert_manager is not None and getattr(feature_result, "alerts", None):
                        self.alert_manager.record(
                            frame_idx=frame_idx,
                            feature=str(metrics.get("feature", self.config.feature or "unknown")),
                            cctv_id=self.config.cctv_id,
                            alerts=feature_result.alerts,
                            metrics=metrics,
                        )

                    handle_feature_alert_events(
                        engine=self,
                        frame_idx=frame_idx,
                        frame=frame,
                        annotated_frame=annotated_frame,
                        feature_result=feature_result,
                        track_bbox_map=track_bbox_map,
                    )

                # Write to video
                if self.video_writer:
                    self.video_writer.write_frame(annotated_frame)

                # Export frame data
                self.data_exporter.add_frame(
                    frame_idx=frame_idx,
                    detections=detections,
                    tracked_objects=tracked_objects,
                    analytics=analytics_result.to_dict(),
                )

                # Log progress (throttled)
                if (frame_idx + 1) % self._log_every_n_frames == 0:
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    self.logger.info(
                        f"Processed {self.frame_count} frames "
                        f"({fps:.1f} FPS) - Objects: {len(tracked_objects)}"
                    )

                previous_tracks = tracked_objects.copy()

        except Exception as e:
            self.logger.error(f"Error during processing: {e}", exc_info=True)
            raise

        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
            signal.signal(signal.SIGINT, previous_sigint_handler)
            self._cleanup()

    def _cleanup(self):
        """Cleanup and persist final outputs."""
        cleanup_engine(self)

    def _is_rtsp_publisher_ready(self) -> bool:
        """Check RTSP publisher process state without probing RTSP path as a client."""
        pusher = getattr(self, "rtsp_pusher", None)
        if pusher is None:
            return False

        process = getattr(pusher, "process", None)
        if not getattr(pusher, "is_running", False) or process is None:
            return False

        return process.poll() is None
