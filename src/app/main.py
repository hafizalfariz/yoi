"""
YOI Vision Engine - Main Entry Point
"""

import argparse
import copy
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]

# Ensure project root is in path for imports
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from yoi import __version__ as yoi_version  # noqa: E402
from yoi.components.engine import VisionEngine  # noqa: E402
from yoi.config import VideoInputConfig, YOIConfig  # noqa: E402
from yoi.utils.logger import logger_service  # noqa: E402

_ACTIVE_ENGINE: VisionEngine | None = None
_SHUTDOWN_REQUESTED = False
_ACTIVE_CHILD_PROCESSES: list[subprocess.Popen] = []


def _parse_percent_value(raw: str) -> float | None:
    value = str(raw or "").strip().lower()
    if not value:
        return None
    if value == "max":
        return 100.0
    if value.endswith("%"):
        value = value[:-1].strip()
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed < 10 or parsed > 100:
        return None
    return parsed


def _apply_auto_percent_thread_limits(logger) -> None:
    runtime_profile, _ = _resolve_runtime_profile()
    percent_env_key = "GPU_CPU_LIMIT_PERCENT" if runtime_profile == "gpu" else "CPU_LIMIT_PERCENT"
    percent_value = _parse_percent_value(os.getenv(percent_env_key, ""))

    if percent_value is None:
        return

    logical_cores = os.cpu_count() or 1
    target_threads = max(1, int(math.floor((logical_cores * percent_value) / 100.0)))

    os.environ["OMP_NUM_THREADS"] = str(target_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(target_threads)
    os.environ["MKL_NUM_THREADS"] = str(target_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(target_threads)
    os.environ["OPENCV_FOR_THREADS_NUM"] = str(max(1, min(target_threads, 8)))

    logger.info(
        "Auto thread sizing: %s=%s -> logical_cores=%s, target_threads=%s",
        percent_env_key,
        os.getenv(percent_env_key),
        logical_cores,
        target_threads,
    )


def _resolve_runtime_profile() -> Tuple[str, str]:
    """Resolve runtime profile label and source for logging."""
    runtime_profile = os.getenv("YOI_RUNTIME_PROFILE", "").strip().lower()
    if runtime_profile in {"cpu", "gpu"}:
        return runtime_profile, "YOI_RUNTIME_PROFILE"

    target_device = os.getenv("YOI_TARGET_DEVICE", "").strip().lower()
    if target_device == "gpu":
        target_device = "cuda"

    if target_device == "cuda":
        return "gpu", "YOI_TARGET_DEVICE"
    if target_device in {"cpu", "mps"}:
        return "cpu", "YOI_TARGET_DEVICE"

    return "unknown", "default"


def _normalize_device_label(raw_device: Optional[str]) -> str:
    """Normalize device label to canonical runtime values."""
    normalized = str(raw_device or "").strip().lower()
    if normalized == "gpu":
        return "cuda"
    if normalized in {"cpu", "cuda", "mps"}:
        return normalized
    return ""


def _resolve_runtime_target_device() -> Tuple[str, str]:
    """Resolve canonical runtime target device and source env label."""
    target_device = _normalize_device_label(os.getenv("YOI_TARGET_DEVICE", ""))
    if target_device:
        return target_device, "YOI_TARGET_DEVICE"

    runtime_profile = os.getenv("YOI_RUNTIME_PROFILE", "").strip().lower()
    if runtime_profile == "gpu":
        return "cuda", "YOI_RUNTIME_PROFILE"
    if runtime_profile == "cpu":
        return "cpu", "YOI_RUNTIME_PROFILE"

    return "", "default"


def _get_config_model_device(config_path: Path, logger) -> str:
    """Load config and return canonical model device value (or empty when unavailable)."""
    try:
        config = _load_config_from_path(config_path)
    except Exception as exc:
        logger.warning("Failed reading model.device from %s: %s", config_path, exc)
        return ""

    raw_device = getattr(getattr(config, "model", None), "device", "")
    normalized = _normalize_device_label(raw_device)
    if not normalized:
        logger.warning(
            "Config %s has unsupported model.device '%s'",
            config_path,
            raw_device,
        )
    return normalized


def _ensure_config_runtime_device_match(config_path: Path, logger) -> bool:
    """Return False when config model.device does not match runtime target device."""
    runtime_device, runtime_source = _resolve_runtime_target_device()
    if not runtime_device:
        return True

    config_device = _get_config_model_device(config_path, logger)
    if not config_device:
        logger.error(
            "Cannot determine model.device for config %s while runtime device is '%s'",
            config_path,
            runtime_device,
        )
        return False

    if config_device != runtime_device:
        logger.warning(
            "Config device mismatch: config=%s (from %s), runtime=%s (from %s)",
            config_device,
            config_path,
            runtime_device,
            runtime_source,
        )
        logger.error(
            "Refusing to run config %s because model.device '%s' does not match runtime '%s'",
            config_path,
            config_device,
            runtime_device,
        )
        return False

    return True


def _signal_name(signum: int) -> str:
    """Resolve OS signal name for logs."""
    try:
        return signal.Signals(signum).name
    except Exception:
        return str(signum)


def _terminate_active_child_processes(
    logger, signal_name: str, grace_seconds: float = 55.0
) -> None:
    """Terminate active parallel child processes gracefully, then force-kill if needed."""
    global _ACTIVE_CHILD_PROCESSES

    if not _ACTIVE_CHILD_PROCESSES:
        return

    logger.warning(
        "Forwarding %s to %s active child process(es)",
        signal_name,
        len(_ACTIVE_CHILD_PROCESSES),
    )

    for proc in list(_ACTIVE_CHILD_PROCESSES):
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception as exc:
            logger.warning("Failed to terminate child process: %s", exc)

    deadline = time.time() + max(0.0, grace_seconds)
    while time.time() < deadline:
        alive = [proc for proc in _ACTIVE_CHILD_PROCESSES if proc.poll() is None]
        if not alive:
            break
        time.sleep(0.2)

    for proc in list(_ACTIVE_CHILD_PROCESSES):
        try:
            if proc.poll() is None:
                logger.warning("Force-killing stubborn child process PID=%s", proc.pid)
                proc.kill()
        except Exception:
            pass


def _register_signal_handlers(logger) -> None:
    """Register signal handlers for graceful shutdown."""

    def _handler(signum, _frame):
        global _SHUTDOWN_REQUESTED
        global _ACTIVE_ENGINE

        _SHUTDOWN_REQUESTED = True
        sig_name = _signal_name(signum)
        logger.warning("Received %s - initiating graceful shutdown", sig_name)

        if _ACTIVE_ENGINE is not None:
            try:
                _ACTIVE_ENGINE.request_stop(f"signal {sig_name}")
            except Exception as exc:
                logger.warning("Failed to request engine stop: %s", exc)

        _terminate_active_child_processes(logger, sig_name)

        # Force controlled interruption so processing unwinds immediately
        # and VisionEngine.process() executes its finally cleanup path.
        raise KeyboardInterrupt(f"Signal {sig_name}")

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def validate_models(logger):
    """Validate and log available models

    Uses MODELS_DIR from environment if available, otherwise defaults to ./models
    relative to the project root, so local run and docker behave the same.
    """
    models_env = os.getenv("MODELS_DIR")
    if models_env:
        models_path = Path(models_env)
    else:
        models_path = ROOT_DIR / "models"
    logger.info("=" * 70)
    logger.info("MODEL VALIDATION")
    logger.info("=" * 70)

    if not models_path.exists():
        logger.warning(f"Models directory not found: {models_path.absolute()}")
        return False

    logger.info(f"Models directory found: {models_path.absolute()}")

    model_dirs = [d for d in models_path.iterdir() if d.is_dir()]
    if not model_dirs:
        logger.warning("No model directories found")
        return False

    logger.info(f"Found {len(model_dirs)} model directory(ies):")
    for model_dir in model_dirs:
        # Check for version subdirectories (e.g., models/person_detection/1/)
        version_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        files = []

        if version_dirs:
            # If version directories exist, check those
            for version_dir in version_dirs:
                files.extend(version_dir.iterdir())
                logger.info(f"  {model_dir.name}")
                for f in version_dir.iterdir():
                    logger.info(f"     - {f.name}")
        else:
            # Otherwise check the model directory directly
            files = list(model_dir.iterdir())
            logger.info(f"  {model_dir.name}")
            for f in files:
                logger.info(f"     - {f.name}")

        has_weight = any(f.suffix in [".onnx", ".pt", ".pth"] for f in files)
        has_metadata = any(f.name == "metadata.yaml" for f in files)

        if not has_weight:
            logger.warning("     No weight file found (.onnx, .pt, .pth)")
        if not has_metadata:
            logger.warning("     No metadata.yaml found")

    logger.info("=" * 70)
    return len(model_dirs) > 0


def validate_config(config_path: str, logger):
    """Validate and log config file"""
    logger.info("=" * 70)
    logger.info("CONFIG VALIDATION")
    logger.info("=" * 70)

    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"Config file not found: {config_file.absolute()}")
        return False

    logger.info(f"Config file found: {config_file.absolute()}")
    logger.info(f"  - File size: {config_file.stat().st_size} bytes")
    logger.info(f"  - File type: {config_file.suffix}")

    try:
        if config_file.suffix == ".yaml" or config_file.suffix == ".yml":
            config = YOIConfig.from_yaml(str(config_path))
        elif config_file.suffix == ".json":
            config = YOIConfig.from_json(str(config_path))
        else:
            logger.error(f"Unsupported config format: {config_file.suffix}")
            return False

        logger.info("Config parsed successfully")
        logger.info(f"  - Config name: {config.config_name}")
        logger.info(f"  - CCTV ID: {config.cctv_id}")
        logger.info(f"  - Feature: {getattr(config, 'feature', 'N/A')}")
        logger.info(
            f"  - Input source: {config.input.source if hasattr(config, 'input') else 'N/A'}"
        )
        logger.info("=" * 70)
        return True

    except Exception as e:
        logger.error(f"Failed to parse config: {e}")
        logger.exception("Config parsing error:")
        logger.info("=" * 70)
        return False


def create_sample_config(output_path: str) -> YOIConfig:
    """Create sample configuration"""
    config = YOIConfig(
        config_name="sample_config",
        cctv_id="camera_office",
        input=VideoInputConfig(
            source="input/sample.mp4", max_fps=30, frame_size=(1280, 720), buffer_size=30
        ),
    )
    return config


def resolve_default_configs(logger) -> List[Path]:
    """Resolve default config paths for local run.

    Priority:
    1. YOI_CONFIG_PATH env (absolute or relative) -> single file
    2. CONFIG_DIR + YOI_CONFIG_FILE (envs) -> single file
    3. All *.yaml / *.yml files in CONFIG_DIR (sorted by name)
    """
    # 1. Explicit config path
    env_config = os.getenv("YOI_CONFIG_PATH")
    if env_config:
        cfg = Path(env_config)
        if not cfg.is_absolute():
            cfg = ROOT_DIR / cfg
        logger.info(f"Using YOI_CONFIG_PATH from env: {cfg}")
        return [cfg]

    # 2. Base config directory (default: ./configs/app)
    config_dir_env = os.getenv("CONFIG_DIR")
    if config_dir_env:
        config_dir = Path(config_dir_env)
        if not config_dir.is_absolute():
            config_dir = ROOT_DIR / config_dir
    else:
        config_dir = ROOT_DIR / "configs" / "app"

    # If CONFIG_DIR accidentally points to a file, just use it directly
    if config_dir.is_file():
        logger.info(f"Using CONFIG_DIR file as config: {config_dir}")
        return [config_dir]

    # 3. CONFIG_DIR + file name from env (if provided)
    file_name = os.getenv("YOI_CONFIG_FILE")
    if file_name:
        candidate = config_dir / file_name
        if candidate.exists():
            logger.info(f"Using YOI_CONFIG_FILE in CONFIG_DIR: {candidate}")
            return [candidate]
        logger.warning("YOI_CONFIG_FILE not found in CONFIG_DIR: %s", candidate)
    else:
        candidate = config_dir / "config.yaml"

    # 4. Fallback to all YAML in directory (alphabetical for deterministic order)
    yaml_files = sorted(
        list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml")),
        key=lambda path: path.name.lower(),
    )

    if yaml_files:
        runtime_device, runtime_source = _resolve_runtime_target_device()

        if len(yaml_files) == 1:
            logger.info(f"Using single YAML in {config_dir}: {yaml_files[0]}")
            return yaml_files

        logger.info(
            "Multiple YAML configs found in %s; detected (%s)",
            config_dir,
            ", ".join(path.name for path in yaml_files),
        )

        if not runtime_device:
            logger.info(
                "No runtime device filter set; running all detected configs (%s)",
                ", ".join(path.name for path in yaml_files),
            )
            return yaml_files

        logger.info(
            "Auto-detect device filter active: runtime=%s (from %s)",
            runtime_device,
            runtime_source,
        )

        selected_configs: List[Path] = []
        skipped_configs: List[str] = []
        for cfg_path in yaml_files:
            cfg_device = _get_config_model_device(cfg_path, logger)
            if not cfg_device:
                skipped_configs.append(f"{cfg_path.name}(unknown)")
                continue
            if cfg_device == runtime_device:
                selected_configs.append(cfg_path)
            else:
                skipped_configs.append(f"{cfg_path.name}({cfg_device})")

        if selected_configs:
            logger.info(
                "Auto-detect selected configs (%s): %s",
                len(selected_configs),
                ", ".join(path.name for path in selected_configs),
            )
        if skipped_configs:
            logger.warning(
                "Auto-detect skipped configs due to device mismatch (%s): %s",
                len(skipped_configs),
                ", ".join(skipped_configs),
            )

        if not selected_configs:
            logger.error(
                "No compatible configs found for runtime device '%s' in %s",
                runtime_device,
                config_dir,
            )
            return []

        return selected_configs

    # Nothing found; return path where we expected the convention file
    logger.error(f"No config YAML found in {config_dir}")
    return [candidate]


def _load_config_from_path(config_path: Path) -> YOIConfig:
    """Load config object from supported file path."""
    if config_path.suffix in (".yaml", ".yml"):
        return YOIConfig.from_yaml(str(config_path))
    if config_path.suffix == ".json":
        return YOIConfig.from_json(str(config_path))
    raise ValueError(f"Unsupported config format: {config_path.suffix}")


def _apply_runtime_config_context(config: YOIConfig, config_path: Path) -> None:
    """Attach active config metadata and per-config logging context."""
    if config.metadata is None:
        config.metadata = {}
    config.metadata["_active_config_path"] = str(config_path)
    config.metadata["_active_config_stem"] = config_path.stem
    os.environ["YOI_LOG_CONFIG_TAG"] = config_path.stem
    os.environ["YOI_LOG_FILE_SUFFIX"] = config_path.stem


def _expand_run_queue_multi_video(
    run_queue: List[Tuple[YOIConfig, str]],
    logger,
) -> List[Tuple[YOIConfig, str]]:
    expanded_queue: List[Tuple[YOIConfig, str]] = []

    for run_cfg, label in run_queue:
        input_cfg = getattr(run_cfg, "input", None)
        if input_cfg is None:
            expanded_queue.append((run_cfg, label))
            continue

        source_type = str(getattr(input_cfg, "source_type", "video") or "video").lower()
        if source_type != "video":
            expanded_queue.append((run_cfg, label))
            continue

        source_paths = input_cfg.get_source_paths()
        if len(source_paths) <= 1:
            expanded_queue.append((run_cfg, label))
            continue

        logger.info(
            "Multi-video config detected: %s sources will run sequentially",
            len(source_paths),
        )

        for idx, source_path in enumerate(source_paths, start=1):
            cfg_copy = copy.deepcopy(run_cfg)
            cfg_copy.input.source = source_path
            cfg_copy.input.video_source = source_path
            cfg_copy.input.video_files = [source_path]

            if cfg_copy.metadata is None:
                cfg_copy.metadata = {}
            cfg_copy.metadata["video_source_index"] = idx
            cfg_copy.metadata["video_source_total"] = len(source_paths)

            source_name = Path(source_path).name or source_path
            expanded_queue.append(
                (
                    cfg_copy,
                    f"{label} [video {idx}/{len(source_paths)}: {source_name}]",
                )
            )

    return expanded_queue


def run_configs_in_parallel(config_paths: List[Path], logger) -> int:
    """Run multiple config files in parallel subprocesses (with optional worker limit)."""
    worker_limit_raw = os.getenv("YOI_MAX_PARALLEL_CONFIGS", "0").strip()
    worker_limit = 0
    try:
        worker_limit = int(worker_limit_raw)
    except ValueError:
        worker_limit = 0

    if worker_limit <= 0:
        worker_limit = len(config_paths)

    worker_limit = max(1, min(worker_limit, len(config_paths)))

    auto_tune_parallel_threads = (
        os.getenv("YOI_AUTO_TUNE_PARALLEL_THREADS", "1").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    omp_total_raw = os.getenv("OMP_NUM_THREADS", "").strip()
    try:
        omp_total = int(omp_total_raw) if omp_total_raw else int(os.cpu_count() or worker_limit)
    except ValueError:
        omp_total = int(os.cpu_count() or worker_limit)

    opencv_total_raw = os.getenv("OPENCV_FOR_THREADS_NUM", "").strip()
    try:
        opencv_total = int(opencv_total_raw) if opencv_total_raw else worker_limit
    except ValueError:
        opencv_total = worker_limit

    per_worker_omp = max(1, omp_total // worker_limit)
    per_worker_opencv = max(1, opencv_total // worker_limit)

    logger.info(
        "Parallel config runner using up to %s worker(s) for %s config(s)",
        worker_limit,
        len(config_paths),
    )
    if auto_tune_parallel_threads and worker_limit > 1:
        logger.info(
            "Parallel thread auto-tune enabled: OMP %s->%s per worker, OpenCV %s->%s per worker",
            omp_total,
            per_worker_omp,
            opencv_total,
            per_worker_opencv,
        )

    remaining = list(config_paths)
    failed: List[Tuple[Path, int]] = []

    global _ACTIVE_CHILD_PROCESSES
    try:
        while remaining:
            batch = remaining[:worker_limit]
            remaining = remaining[worker_limit:]

            processes: List[Tuple[Path, subprocess.Popen]] = []
            for config_path in batch:
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(config_path),
                ]
                proc_env = os.environ.copy()
                proc_env["YOI_LOG_CONFIG_TAG"] = config_path.stem
                proc_env["YOI_LOG_FILE_SUFFIX"] = config_path.stem
                if auto_tune_parallel_threads and worker_limit > 1:
                    proc_env["OMP_NUM_THREADS"] = str(per_worker_omp)
                    proc_env["OPENBLAS_NUM_THREADS"] = str(per_worker_omp)
                    proc_env["MKL_NUM_THREADS"] = str(per_worker_omp)
                    proc_env["NUMEXPR_NUM_THREADS"] = str(per_worker_omp)
                    proc_env["OPENCV_FOR_THREADS_NUM"] = str(per_worker_opencv)
                logger.info(f"Launching parallel run for config: {config_path}")
                proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=proc_env)
                processes.append((config_path, proc))

            _ACTIVE_CHILD_PROCESSES = [proc for _, proc in processes]

            for config_path, proc in processes:
                exit_code = proc.wait()
                if exit_code != 0:
                    failed.append((config_path, exit_code))
    finally:
        _ACTIVE_CHILD_PROCESSES = []

    if failed:
        for config_path, exit_code in failed:
            logger.error(f"Parallel run failed for {config_path} (exit={exit_code})")
        return 1

    logger.info("All parallel config runs completed successfully")
    return 0


def main():
    global _SHUTDOWN_REQUESTED
    global _ACTIVE_ENGINE

    parser = argparse.ArgumentParser(description="YOI Vision AI Engine")
    parser.add_argument("--config", type=str, help="Config file path (YAML or JSON)")
    parser.add_argument("--video", type=str, help="Video file or RTSP URL")
    parser.add_argument("--model", type=str, default="yolov8n", help="YOLO model name")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Max FPS for processing")
    parser.add_argument("--sample", action="store_true", help="Generate sample config")
    parser.add_argument("--validate", action="store_true", help="Validate models and config only")

    args = parser.parse_args()

    logger = logger_service.get_engine_logger()
    _SHUTDOWN_REQUESTED = False
    _ACTIVE_ENGINE = None
    _register_signal_handlers(logger)
    _apply_auto_percent_thread_limits(logger)

    try:
        # Log startup information
        runtime_profile, runtime_profile_source = _resolve_runtime_profile()
        target_device = os.getenv("YOI_TARGET_DEVICE", "").strip() or "<unset>"
        strict_device = os.getenv("YOI_STRICT_DEVICE", "0").strip() or "0"
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip() or "<unset>"

        logger.info("\n" + "=" * 70)
        logger.info("ENGINE: STARTUP")
        logger.info("=" * 70)
        logger.info(f"Working dir : {os.getcwd()}")
        logger.info(f"Python      : {sys.version.split()[0]}")
        logger.info(f"YOI Version : {yoi_version}")
        logger.info(
            "Runtime     : profile=%s (source=%s), target_device=%s, strict=%s",
            runtime_profile,
            runtime_profile_source,
            target_device,
            strict_device,
        )
        logger.info(f"CUDA Visible: {cuda_visible_devices}")
        logger.info("Credits     : by hafizalfariz")
        logger.info("=" * 70 + "\n")

        # 1) Validate models
        logger.info("STEP 1/4 - Validating models...")
        models_ok = validate_models(logger)
        if models_ok:
            logger.info("STEP 1/4 - MODELS OK")
        else:
            logger.warning("STEP 1/4 - MODELS CHECK FAILED (see details above)")

        # 2) Load / create config(s)
        logger.info("STEP 2/4 - Loading configuration...")
        run_queue: List[Tuple[YOIConfig, str]] = []
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                return 1

            if not _ensure_config_runtime_device_match(config_path, logger):
                return 1

            # Validate config
            config_ok = validate_config(str(config_path), logger)
            if not config_ok:
                return 1

            config = _load_config_from_path(config_path)
            _apply_runtime_config_context(config, config_path)

            logger.info(f"Config loaded successfully: {args.config}")
            logger.info("STEP 2/4 - CONFIG OK (from --config)")
            run_queue.append((config, str(config_path)))

        elif args.video:
            config = create_sample_config(".")
            config.input.source = args.video
            config.input.max_fps = args.fps
            config.model.name = args.model
            config.model.device = args.device
            config.output.output_dir = args.output_dir
            logger.info("Config created from command line arguments")
            logger.info("STEP 2/4 - CONFIG OK (from --video arguments)")
            run_queue.append((config, f"video:{args.video}"))

        elif args.sample:
            config = create_sample_config(".")
            config.save_yaml("sample_config.yaml")
            config.save_json("sample_config.json")
            logger.info("Sample config generated: sample_config.yaml, sample_config.json")
            return 0

        elif args.validate:
            logger.info("Validation completed successfully")
            return 0

        else:
            # No explicit arguments: run all default YAML configs
            config_paths = resolve_default_configs(logger)

            if not config_paths:
                logger.error("STEP 2/4 - CONFIG FAILED (no compatible config for active runtime)")
                return 1

            if len(config_paths) > 1:
                logger.info("STEP 2/4 - CONFIG OK (default resolver)")
                logger.info(
                    "Launching %s default configs in parallel",
                    len(config_paths),
                )
                return run_configs_in_parallel(config_paths, logger)

            for config_path in config_paths:
                if not config_path.exists():
                    logger.error(f"Default config file not found: {config_path}")
                    return 1

                if not _ensure_config_runtime_device_match(config_path, logger):
                    return 1

                config_ok = validate_config(str(config_path), logger)
                if not config_ok:
                    return 1

                try:
                    config = _load_config_from_path(config_path)
                except ValueError as exc:
                    logger.error(str(exc))
                    return 1

                _apply_runtime_config_context(config, config_path)

                logger.info(f"Config loaded successfully (default): {config_path}")
                run_queue.append((config, str(config_path)))

            logger.info("STEP 2/4 - CONFIG OK (default resolver)")

        run_queue = _expand_run_queue_multi_video(run_queue, logger)

        # 3) & 4) Summarize input/output and run engine per config
        for idx, (config, config_label) in enumerate(run_queue, start=1):
            if _SHUTDOWN_REQUESTED:
                logger.warning("Shutdown requested before run start - skipping remaining configs")
                break

            logger.info("\n" + "=" * 70)
            logger.info(
                "RUN %s/%s - ACTIVE CONFIG: %s",
                idx,
                len(run_queue),
                config_label,
            )
            logger.info("=" * 70)

            if hasattr(config, "input") and config.input is not None:
                try:
                    source_type = getattr(config.input, "source_type", None) or "video"
                    source_path = config.input.get_source_path()
                    logger.info("STEP 3/4 - INPUT SUMMARY")
                    logger.info(f"  - Type : {source_type}")
                    logger.info(f"  - Source: {source_path}")
                    if isinstance(source_path, str) and source_path.startswith("rtsp://"):
                        logger.info(f"  - RTSP MODE: {source_path}")
                except Exception as e:
                    logger.warning(f"Failed to summarize input source: {e}")

            if hasattr(config, "output") and config.output is not None:
                url_dash = getattr(config.output, "url_dashboard", None)
                if url_dash:
                    logger.info("OUTPUT: dashboard URL configured")
                    logger.info(f"  - Dashboard: {url_dash}")
                rtsp_url = getattr(config.output, "rtsp_url", None)
                if rtsp_url:
                    logger.info("OUTPUT: RTSP stream configured")
                    logger.info(f"  - RTSP OUT: {rtsp_url}")

            logger.info("\n" + "=" * 70)
            logger.info("STEP 4/4 - STARTING INFERENCE ENGINE")
            logger.info("=" * 70 + "\n")
            engine = VisionEngine(config)
            _ACTIVE_ENGINE = engine
            try:
                engine.process()
            finally:
                _ACTIVE_ENGINE = None

            if _SHUTDOWN_REQUESTED:
                logger.warning("Graceful shutdown completed for active run")
                break

        logger.info("\n" + "=" * 70)
        logger.info("ENGINE: PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70 + "\n")
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        logger.info("=" * 70 + "\n")
        return 1
    except KeyboardInterrupt:
        logger.warning("Shutdown interrupted by signal; exiting gracefully")
        logger.info("=" * 70 + "\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
