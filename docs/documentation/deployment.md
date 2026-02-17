# Deployment Guide

This guide provides a focused deployment flow for production-like CPU/GPU runtime operations.

## Audience

- DevOps engineers
- Release owners
- Operators handling runtime rollout

## 1) Environment Setup

1. Copy template file:
   - `.env.example` -> `.env`
2. Adjust deployment values in `.env`:
  - Section 2 (Resource Policy): `CPU_LIMIT_PERCENT`, `GPU_CPU_LIMIT_PERCENT`, `MEMORY_LIMIT_PERCENT`
  - Section 3 (Paths/Config): `INPUT_PATH`, `OUTPUT_PATH`, `LOGS_PATH`, `MODELS_PATH`, `CONFIGS_PATH`, `CONFIG_DIR`
  - Section 4 (Device Policy): `YOI_STRICT_DEVICE`
  - Section 5 (Inference): `YOI_MAX_INFERENCE_SECONDS`, `YOI_INFER_EVERY_N_FRAMES`
  - Section 6 (RTSP): `YOI_RTSP_OUTPUT_FPS`, `YOI_RTSP_BITRATE`, `YOI_RTSP_PRESET`

Notes:

- Runtime limit uses seconds (`YOI_MAX_INFERENCE_SECONDS`), not minutes.
- Keep `YOI_MAX_INFERENCE_SECONDS=0` to disable auto-stop.
- Percent policy supports `10..100` or `max` and runtime auto-calculates thread/core usage.
- Memory policy uses `MEMORY_LIMIT_PERCENT` (`10..100` or `max`) when using `yoi/devtools/dev.ps1` (Windows) or `yoi/devtools/dev.sh` (Linux/macOS).
- If `MEMORY_LIMIT_PERCENT` is empty, compose service default `mem_limit` is used.
- `.env.example` is grouped by numbered sections for faster editing and safer review.

## 2) Runtime Assets

Prepare all runtime assets before startup:

- Input videos in `input/`
- Config file in `configs/app/`
- Model files in `models/<model_name>/1/`

Recommended model format by target runtime:

- CPU target: `best.onnx`
- GPU target: `best.pt`

Supported extensions: `.onnx`, `.pt`, `.pth`

## 3) Model Selection Behavior

Runtime resolves model files according to target device/profile:

- Target `cpu` -> prefers `.onnx`, then `.pt/.pth`
- Target `cuda/gpu` -> prefers `.pt/.pth`, then `.onnx`

Strict GPU mode:

- `YOI_STRICT_DEVICE=1` rejects ONNX to avoid silent CPU fallback.

## 4) Start Services

First deployment run (includes build):

- CPU:
  - `docker compose --profile cpu up -d --build`
- GPU:
  - `docker compose --profile gpu up -d --build`

Repeat run without rebuild:

- CPU: `docker compose --profile cpu up -d`
- GPU: `docker compose --profile gpu up -d`

No-build with latest local code mounts (development override):

- CPU:
  - `docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile cpu up -d --no-build`
- GPU:
  - `docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile gpu up -d --no-build`

## 5) Validation Checks

Follow logs:

- CPU: `docker compose logs -f app-yoi-cpu`
- GPU: `docker compose logs -f app-yoi-gpu`

Confirm startup line includes expected profile:

- `Runtime     : profile=cpu ...`
- `Runtime     : profile=gpu ...`

Optional error-only filtering (PowerShell):

- `docker compose logs -f app-yoi-gpu | Select-String -Pattern "error|exception|traceback"`

## 6) Stop / Teardown

Profile-specific stop:

- CPU: `docker compose --profile cpu down`
- GPU: `docker compose --profile gpu down`

Global stop (all active services in project):

- `docker compose down`

## 7) Clean Run Checklist

Before final clean run:

- Ensure desired config file is finalized in `configs/app/`
- Ensure model format matches target runtime policy
- Run first startup with `--build`
- Verify profile line in logs
- Verify output artifacts in `output/<config_name>/<video_name>_<timestamp>/`

## Related Docs

- [Main README](../../README.md)
- [Operations](operations.md)
- [Before Run Checklist](before-run-checklist.md)
- [Runtime Flow](runtime-flow.md)
- [Configuration Builder](configuration-builder.md)
