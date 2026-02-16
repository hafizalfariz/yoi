"""Output lifecycle helpers for VisionEngine."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2

from yoi.alert import AlertManager
from yoi.annotate.video_annotator import VideoAnnotator
from yoi.output.exporters import DataExporter, VideoWriter
from yoi.stream import RTSPPushConfig, RTSPPusher


def _flag_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value and hasattr(value, "enabled"):
        return bool(value.enabled)
    return False


def _resolve_annotated_output_path(engine, base_output_dir: Path) -> Path:
    """Resolve output path as: base/config_name/video_name_timestamp."""
    run_timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    config_name = "default"
    metadata = getattr(engine.config, "metadata", None) or {}
    if isinstance(metadata, dict):
        config_name = (
            str(metadata.get("_active_config_stem") or "").strip()
            or str(getattr(engine.config, "config_name", "default") or "default").strip()
        )
    else:
        config_name = str(getattr(engine.config, "config_name", "default") or "default").strip()

    config_name = Path(config_name).name or "default"
    output_path = base_output_dir / config_name

    source_path = engine.config.input.get_source_path() if engine.config.input else ""
    video_name = Path(source_path).stem if source_path else ""
    if video_name:
        output_path = output_path / f"{video_name}_{run_timestamp}"
    else:
        output_path = output_path / run_timestamp

    return output_path


def initialize_output_engines(engine) -> None:
    """Initialize output directory, writers, exporters, and RTSP pusher."""
    source_type = ""
    if engine.config.input and getattr(engine.config.input, "source_type", None):
        source_type = str(engine.config.input.source_type).lower().strip()

    is_production = False
    if engine.config.output and getattr(engine.config.output, "mode", None):
        is_production = str(engine.config.output.mode).lower() == "production"

    if source_type == "rtsp":
        base_output_dir = (
            engine.config.logs.base_dir
            if engine.config.logs and engine.config.logs.base_dir
            else "logs"
        )
    elif is_production:
        base_output_dir = (
            engine.config.logs.base_dir
            if engine.config.logs and engine.config.logs.base_dir
            else "logs"
        )
    else:
        base_output_dir = (
            engine.config.output.get_output_dir() if engine.config.output else "output"
        )

    output_path = Path(base_output_dir)
    logs_cfg = engine.config.logs
    data_folder = getattr(logs_cfg, "data_folder", None) or "data"
    image_folder = getattr(logs_cfg, "image_folder", None) or "image"
    status_folder = getattr(logs_cfg, "status_folder", None) or "status"
    csv_filename = getattr(logs_cfg, "csv_file", None) or "data.csv"
    save_annotations = (
        _flag_enabled(engine.config.output.save_annotations) if engine.config.output else False
    )

    if save_annotations and engine.config.output.output_format == "annotated_video":
        try:
            output_path = _resolve_annotated_output_path(engine, output_path)
        except Exception as exc:
            engine.logger.warning(
                f"Failed to derive per-video output folder from input source: {exc}"
            )

    engine.output_dir = output_path
    engine.output_dir.mkdir(parents=True, exist_ok=True)

    for sub in (image_folder, data_folder, status_folder):
        (engine.output_dir / sub).mkdir(parents=True, exist_ok=True)

    engine.image_dir = engine.output_dir / image_folder
    engine.data_dir = engine.output_dir / data_folder
    engine.status_dir = engine.output_dir / status_folder

    for filename in (
        "detections.json",
        "detections.csv",
        "processing.log",
        "analytics_summary.json",
        "analytics_summary.csv",
        csv_filename,
    ):
        file_path = engine.output_dir / filename
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as exc:
                engine.logger.warning(f"Failed to remove old file {file_path}: {exc}")

    engine.data_csv_path = engine.output_dir / csv_filename
    try:
        engine.data_csv_path.write_text(
            "image_id,timestamp,feature,status,data_path,image_path\n",
            encoding="utf-8",
        )
    except Exception as exc:
        engine.logger.warning(f"Failed to initialize data CSV {engine.data_csv_path}: {exc}")

    engine.logs_dir = Path(engine.config.logs.base_dir) if engine.config.logs else Path("logs")
    engine.logs_dir.mkdir(parents=True, exist_ok=True)

    save_video = _flag_enabled(engine.config.output.save_video) if engine.config.output else False
    if save_video:
        frame_size = engine.video_reader.get_frame_size()
        video_dir = engine.output_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / "output_annotated.mp4"

        for old_path in (engine.output_dir / "output_annotated.mp4", video_path):
            if old_path.exists():
                try:
                    old_path.unlink()
                except Exception as exc:
                    engine.logger.warning(f"Failed to remove old video file {old_path}: {exc}")

        try:
            engine.video_writer = VideoWriter(
                output_path=str(video_path),
                fps=engine.video_reader.get_fps(),
                frame_size=frame_size,
                codec="mp4v",
            )
        except Exception as exc:
            engine.logger.error(f"Failed to initialize video writer: {exc}")
            engine.video_writer = None
    else:
        engine.video_writer = None

    engine.annotator = VideoAnnotator()
    engine.data_exporter = DataExporter(str(engine.output_dir))
    engine.alert_manager = AlertManager(output_dir=engine.output_dir, logger=engine.logger)

    _initialize_rtsp(engine)
    engine.logger.info(f"Output engines initialized: {engine.output_dir}")


def _initialize_rtsp(engine) -> None:
    try:
        rtsp_url = None
        if engine.config.output and getattr(engine.config.output, "rtsp_url", None):
            rtsp_url = engine.config.output.rtsp_url

        if not rtsp_url:
            stream_name = None
            metadata = getattr(engine.config, "metadata", None) or {}
            if isinstance(metadata, dict):
                stream_name = metadata.get("_active_config_stem")

            if not stream_name:
                stream_name = getattr(engine.config, "config_name", None)
            if not stream_name or stream_name == "default":
                source_path = engine.config.input.get_source_path() if engine.config.input else None
                if source_path:
                    stream_name = Path(source_path).stem
            if not stream_name:
                stream_name = "stream"
            rtsp_url = f"rtsp://mediamtx:8554/{stream_name}"

        engine._rtsp_url = rtsp_url

        frame_size = engine.video_reader.get_frame_size()
        fps = engine.video_reader.get_fps() or (
            engine.config.input.max_fps if engine.config.input else 25
        )
        push_cfg = RTSPPushConfig(
            server_url=rtsp_url,
            fps=int(fps) if fps else 25,
            width=frame_size[0],
            height=frame_size[1],
        )
        engine.rtsp_pusher = RTSPPusher(push_cfg)

        started = False
        try:
            started = engine.rtsp_pusher.start()
        except Exception as exc:
            engine.logger.exception(f"Unexpected error while starting RTSP pusher: {exc}")

        if started:
            engine.logger.info(f"RTSP OUTPUT: streaming annotated video to {rtsp_url}")
        else:
            engine.logger.warning("Failed to start RTSP pusher; RTSP output disabled")
            if getattr(engine.rtsp_pusher, "last_startup_output", None):
                for index, line in enumerate(engine.rtsp_pusher.last_startup_output[:50], start=1):
                    engine.logger.warning(f"FFMPEG_STARTUP[{index}]: {line}")
            engine.rtsp_pusher = None

        internal = rtsp_url
        external = internal.replace("mediamtx:8554", "localhost:6554") if internal else None
        config_name = getattr(engine.config, "config_name", "unknown")
        if internal:
            engine.logger.info(f"RTSP endpoint internal for '{config_name}': {internal}")
        if external:
            engine.logger.info(f"RTSP endpoint external for '{config_name}': {external}")

    except Exception as exc:
        engine.logger.warning(f"Error initializing RTSP pusher: {exc}")


def handle_line_cross_events(
    engine, frame_idx: int, annotated_frame, metrics: Dict[str, Any]
) -> None:
    """Write event snapshots and metadata for line-cross increments."""
    try:
        curr_in = int(metrics.get("total_in", 0))
        curr_out = int(metrics.get("total_out", 0))
    except Exception:
        return

    delta_in = max(curr_in - getattr(engine, "_prev_total_in", 0), 0)
    delta_out = max(curr_out - getattr(engine, "_prev_total_out", 0), 0)

    engine._prev_total_in = curr_in
    engine._prev_total_out = curr_out

    if delta_in == 0 and delta_out == 0:
        return

    for directory in (
        getattr(engine, "image_dir", None),
        getattr(engine, "data_dir", None),
        getattr(engine, "status_dir", None),
    ):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def _save_event(direction: str) -> None:
        try:
            ts = datetime.utcnow().isoformat()
            engine._event_counter += 1
            image_id = f"{frame_idx:06d}_{engine._event_counter:04d}_{direction}"

            image_rel = f"image/{image_id}.jpg"
            data_rel = f"data/{image_id}.json"

            image_path = engine.image_dir / f"{image_id}.jpg"
            data_path = engine.data_dir / f"{image_id}.json"
            status_path = engine.status_dir / f"{image_id}.json"

            try:
                cv2.imwrite(str(image_path), annotated_frame)
            except Exception as exc:
                engine.logger.warning(f"Failed to save event image {image_path}: {exc}")

            event_payload: Dict[str, Any] = {
                "image_id": image_id,
                "timestamp": ts,
                "feature": "line_cross",
                "status": direction,
                "frame_idx": frame_idx,
                "metrics": metrics,
            }
            try:
                import json as _json

                data_path.write_text(
                    _json.dumps(event_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as exc:
                engine.logger.warning(f"Failed to write event data {data_path}: {exc}")

            status_payload: Dict[str, Any] = {
                "image_id": image_id,
                "timestamp": ts,
                "feature": "line_cross",
                "status": direction,
                "data_path": data_rel,
                "image_path": image_rel,
                "sent_to_dashboard": False,
            }
            try:
                import json as _json

                status_path.write_text(
                    _json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as exc:
                engine.logger.warning(f"Failed to write status file {status_path}: {exc}")

            try:
                if hasattr(engine, "data_csv_path"):
                    line = f"{image_id},{ts},line_cross,{direction},{data_rel},{image_rel}\n"
                    with engine.data_csv_path.open("a", encoding="utf-8") as file_obj:
                        file_obj.write(line)
            except Exception as exc:
                engine.logger.warning(
                    f"Failed to append event row to {engine.data_csv_path}: {exc}"
                )

        except Exception as exc:
            engine.logger.warning(f"Error while handling line_cross event: {exc}")

    for _ in range(delta_in):
        _save_event("in")
    for _ in range(delta_out):
        _save_event("out")


def cleanup_engine(engine) -> None:
    """Release resources and export final artifacts."""
    engine.logger.info("Cleaning up and saving outputs...")

    if engine.feature_engine:
        final_metrics = engine.feature_engine.get_metrics()
        engine.logger.info(f"Final feature metrics: {final_metrics}")

    if engine.video_writer:
        engine.video_writer.close()

    if getattr(engine, "rtsp_pusher", None) is not None:
        try:
            engine.rtsp_pusher.stop()
        except Exception as exc:
            engine.logger.warning(f"Error while stopping RTSP pusher: {exc}")

    if engine.video_reader:
        engine.video_reader.close()

    save_annotations = (
        _flag_enabled(engine.config.output.save_annotations) if engine.config.output else False
    )
    if save_annotations:
        export_debug_artifacts = engine._env_enabled("YOI_EXPORT_DEBUG_ARTIFACTS", True)
        if export_debug_artifacts:
            engine.data_exporter.export_json()
            engine.data_exporter.export_csv()
            engine.data_exporter.export_logs()

            if engine.analytics_engine:
                try:
                    engine.analytics_engine.export_summaries(str(engine.output_dir))
                except Exception as exc:
                    engine.logger.error(f"Failed to export analytics summaries: {exc}")
        else:
            engine.logger.info("Skipping debug artifact exports (YOI_EXPORT_DEBUG_ARTIFACTS=0)")

    total_time = time.time() - engine.start_time
    engine.logger.info("=" * 70)
    if getattr(engine, "_stop_requested", False):
        engine.logger.info("Processing stopped gracefully (interrupted)")
    else:
        engine.logger.info("Processing completed!")
    engine.logger.info(f"Frames processed: {engine.frame_count}")
    engine.logger.info(f"Total time: {total_time:.2f} seconds")
    engine.logger.info(f"Average FPS: {engine.frame_count / total_time:.2f}")
    engine.logger.info(f"Output directory: {engine.output_dir}")
    engine.logger.info("=" * 70)
