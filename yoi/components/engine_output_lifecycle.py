"""Output lifecycle helpers for VisionEngine."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2

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


def _is_rtsp_input(engine) -> bool:
    if engine.config.input and getattr(engine.config.input, "source_type", None):
        source_type = str(engine.config.input.source_type).lower().strip()
        if source_type == "rtsp":
            return True

    source_path = ""
    if engine.config.input:
        source_path = engine.config.input.get_source_path() or ""

    return source_path.startswith("rtsp://")


def initialize_output_engines(engine) -> None:
    """Initialize output directory, writers, exporters, and RTSP pusher."""
    is_rtsp = _is_rtsp_input(engine)

    if is_rtsp:
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
    engine._event_status_enabled = is_rtsp

    for filename in (csv_filename,):
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
    engine.alert_manager = None

    _initialize_rtsp(engine)
    engine.logger.info(f"Output engines initialized: {engine.output_dir}")


def _source_name_from_input(engine) -> str:
    source_path = ""
    if engine.config.input:
        source_path = engine.config.input.get_source_path() or ""

    if not source_path:
        return "unknown"

    if source_path.startswith("rtsp://"):
        stem = Path(source_path).stem
        return stem or "rtsp"

    return Path(source_path).stem or "unknown"


def _safe_token(raw: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in raw)
    cleaned = cleaned.strip("_")
    return cleaned or "event"


def _resolve_track_crop(frame, track_bbox_map: Dict[int, Any] | None, track_id: int | None):
    if frame is None or track_bbox_map is None or track_id is None:
        return None

    det = track_bbox_map.get(int(track_id))
    if det is None:
        return None

    try:
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(frame_w - 1, int(det.x1)))
        y1 = max(0, min(frame_h - 1, int(det.y1)))
        x2 = max(x1 + 1, min(frame_w, int(det.x2)))
        y2 = max(y1 + 1, min(frame_h, int(det.y2)))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop
    except Exception:
        return None


def _append_event_csv_row(
    engine,
    image_id: str,
    timestamp: str,
    feature_name: str,
    warning_label: str,
    data_rel: str,
    image_rel: str,
) -> None:
    try:
        if hasattr(engine, "data_csv_path"):
            row = f"{image_id},{timestamp},{feature_name},{warning_label},{data_rel},{image_rel}\n"
            with engine.data_csv_path.open("a", encoding="utf-8") as file_obj:
                file_obj.write(row)
    except Exception as exc:
        engine.logger.warning(f"Failed to append event row to {engine.data_csv_path}: {exc}")


def handle_feature_alert_events(
    engine,
    frame_idx: int,
    frame,
    annotated_frame,
    feature_result,
    track_bbox_map: Dict[int, Any] | None = None,
) -> None:
    alerts = getattr(feature_result, "alerts", None) or []
    if not alerts:
        return

    metrics: Dict[str, Any] = getattr(feature_result, "metrics", {}) or {}
    feature_name = str(metrics.get("feature") or getattr(feature_result, "feature_type", "unknown"))
    source_name = _source_name_from_input(engine)
    config_name = str(getattr(engine.config, "config_name", "default") or "default")

    for alert in alerts:
        if not isinstance(alert, dict):
            continue

        try:
            ts = datetime.utcnow().isoformat()
            engine._event_counter += 1

            warning_label = _safe_token(str(alert.get("type", "warning")).lower())
            image_id = (
                f"{frame_idx:06d}_{engine._event_counter:04d}_"
                f"{_safe_token(feature_name)}_{warning_label}"
            )

            image_rel = f"image/{image_id}.jpg"
            data_rel = f"data/{image_id}.json"

            image_path = engine.image_dir / f"{image_id}.jpg"
            data_path = engine.data_dir / f"{image_id}.json"
            status_path = engine.status_dir / f"{image_id}.json"

            track_id_raw = alert.get("track_id")
            track_id = None
            if track_id_raw is not None:
                try:
                    track_id = int(track_id_raw)
                except Exception:
                    track_id = None

            cropped = _resolve_track_crop(frame, track_bbox_map, track_id)
            capture_frame = cropped if cropped is not None else annotated_frame

            try:
                cv2.imwrite(str(image_path), capture_frame)
            except Exception as exc:
                engine.logger.warning(f"Failed to save alert image {image_path}: {exc}")

            event_payload: Dict[str, Any] = {
                "image_id": image_id,
                "timestamp": ts,
                "config_name": config_name,
                "source_name": source_name,
                "feature": feature_name,
                "warning": str(alert.get("type", "warning")),
                "frame_idx": frame_idx,
                "track_id": track_id,
                "cctv_id": getattr(engine.config, "cctv_id", ""),
                "alert": alert,
                "metrics": metrics,
                "image_path": image_rel,
            }

            try:
                data_path.write_text(
                    json.dumps(event_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                engine.logger.warning(f"Failed to write alert data {data_path}: {exc}")

            if bool(getattr(engine, "_event_status_enabled", False)):
                status_payload: Dict[str, Any] = {
                    "image_id": image_id,
                    "timestamp": ts,
                    "feature": feature_name,
                    "status": str(alert.get("type", "warning")),
                    "data_path": data_rel,
                    "image_path": image_rel,
                    "sent_to_dashboard": False,
                }
                try:
                    status_path.write_text(
                        json.dumps(status_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    engine.logger.warning(f"Failed to write status file {status_path}: {exc}")
            elif status_path.exists():
                try:
                    status_path.unlink()
                except Exception:
                    pass

            _append_event_csv_row(
                engine=engine,
                image_id=image_id,
                timestamp=ts,
                feature_name=feature_name,
                warning_label=warning_label,
                data_rel=data_rel,
                image_rel=image_rel,
            )

            engine.logger.info(
                "ALERT_EVENT feature=%s warning=%s track_id=%s image=%s",
                feature_name,
                alert.get("type", "warning"),
                track_id if track_id is not None else "-",
                image_rel,
            )

        except Exception as exc:
            engine.logger.warning(f"Error while handling feature alert event: {exc}")


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
            base_url = os.getenv("YOI_RTSP_BASE_URL", "").strip().rstrip("/")
            if base_url:
                rtsp_url = f"{base_url}/{stream_name}"
            else:
                in_container = Path("/.dockerenv").exists()
                default_host = "mediamtx" if in_container else "localhost"
                default_port = "8554" if in_container else "6554"
                host = os.getenv("YOI_RTSP_HOST", default_host).strip() or default_host
                port = os.getenv("YOI_RTSP_PORT", default_port).strip() or default_port
                rtsp_url = f"rtsp://{host}:{port}/{stream_name}"

        engine._rtsp_url = rtsp_url

        frame_size = engine.video_reader.get_frame_size()
        source_fps = engine.video_reader.get_fps() or (
            engine.config.input.max_fps if engine.config.input else 25
        )
        fps = source_fps
        forced_output_fps = os.getenv("YOI_RTSP_OUTPUT_FPS", "").strip()
        if forced_output_fps:
            try:
                forced = int(forced_output_fps)
                if forced > 0:
                    fps = forced
                else:
                    engine.logger.warning(
                        "Ignoring YOI_RTSP_OUTPUT_FPS=%s (must be > 0)",
                        forced_output_fps,
                    )
            except ValueError:
                engine.logger.warning(
                    "Ignoring invalid YOI_RTSP_OUTPUT_FPS=%s",
                    forced_output_fps,
                )

        bitrate = os.getenv("YOI_RTSP_BITRATE", "").strip() or "2M"
        preset = os.getenv("YOI_RTSP_PRESET", "").strip() or "ultrafast"
        push_cfg = RTSPPushConfig(
            server_url=rtsp_url,
            fps=int(fps) if fps else 25,
            width=frame_size[0],
            height=frame_size[1],
            bitrate=bitrate,
            preset=preset,
        )
        engine.rtsp_pusher = RTSPPusher(push_cfg)

        started = False
        try:
            started = engine.rtsp_pusher.start()
        except Exception as exc:
            engine.logger.exception(f"Unexpected error while starting RTSP pusher: {exc}")

        if started:
            engine.logger.info(f"RTSP OUTPUT: streaming annotated video to {rtsp_url}")
            engine.logger.info(
                "RTSP encoder settings: fps=%s (source=%.2f), bitrate=%s, preset=%s",
                push_cfg.fps,
                float(source_fps or 0),
                push_cfg.bitrate,
                push_cfg.preset,
            )
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
        if external and external != internal:
            engine.logger.info(f"RTSP endpoint external for '{config_name}': {external}")

    except Exception as exc:
        engine.logger.warning(f"Error initializing RTSP pusher: {exc}")

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
        engine.logger.info("Skipping debug artifact exports to keep minimal output layout")

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
