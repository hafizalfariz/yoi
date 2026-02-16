# Operations

This document provides practical runbooks for starting, monitoring, validating, and stopping YOI services.

## Audience and Use

- Audience: DevOps engineers, operators, and release owners.
- Use this document for day-to-day service lifecycle, quality checks, and operational handoff.

## Purpose

- Standardize runtime lifecycle commands across profiles.
- Provide a repeatable quality validation routine.
- Define clear stop/down scope for service-level and project-level operations.

## Standard Policy

- First run uses `--build`.
- Daily/repeat runs use normal `up -d` without `--build`.
- If code changes must be picked up without rebuild, run with dev override file (`docker-compose.dev.yml`).
- Default log follow is app service only (`app-yoi-cpu` or `app-yoi-gpu`).
- MediaMTX log checks are optional and executed separately.
- Optional runtime limit is available via `YOI_MAX_INFERENCE_SECONDS`.
- Runtime may gracefully fall back from GPU to CPU when `YOI_STRICT_DEVICE=0` and GPU initialization fails.

## Pre-Run Checklist

- Place input videos in `input/` (example: `input/sample.mp4`).
- Place model files in `models/<model_name>/1/`.
- CPU profile: prefer `best.onnx`.
- GPU profile: prefer `best.pt`.
- Supported model extensions: `.onnx`, `.pt`, `.pth`.
- Warning: if GPU config points to a model folder that only has `best.onnx` (without `best.pt`), runtime can fall back to CPU and FPS can drop significantly.
- Warning: do not run `app-yoi-cpu` and `app-yoi-gpu` simultaneously on the same RTSP path (for example `dwelltime`), because publishers will replace each other.
- Ensure active config exists in `configs/app/` and references correct model/input.
- Confirm Docker Engine is running before compose commands.

Model resolution policy:

- Target `cpu` resolves `.onnx` first, then `.pt/.pth`.
- Target `cuda/gpu` resolves `.pt/.pth` first, then `.onnx`.
- Strict GPU (`YOI_STRICT_DEVICE=1`) rejects ONNX to avoid silent CPU fallback.
- Recommended production setting: `YOI_STRICT_DEVICE=1` to fail fast on GPU misconfiguration.
- For RTSP input source, runtime artifacts are written under `logs/<config_name>/<stream_name>_<timestamp>/...`.

## End-to-End Playbook

### CPU Playbook

1. Place input video in `input/` (example: `input/sample.mp4`).
2. Place model in `models/<model_name>/1/` (example: `models/person_general_detection/1/best.onnx`).
3. Ensure active config is ready (example: `configs/app/kluis-line.yaml`).
4. Run first startup with build: `docker compose --profile cpu up -d --build`.
5. Optional no-build with latest code mounts: `docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile cpu up -d --no-build`.
6. Follow app logs: `docker compose logs -f app-yoi-cpu`.
7. Confirm startup profile line: `Runtime     : profile=cpu ...`.
8. Optional stream logs: `docker compose logs -f mediamtx`.
9. Stop when complete: `docker compose --profile cpu down`.

### GPU Playbook

1. Place input video in `input/` (example: `input/sample.mp4`).
2. Place model in `models/<model_name>/1/` (example: `models/person_general_detection/1/best.pt`).
  - Warning: for GPU runtime, ensure `best.pt` exists in the selected model folder.
3. Ensure active config is ready (example: `configs/app/kluis-line.yaml`).
4. Run first startup with build: `docker compose --profile gpu up -d --build`.
5. Optional no-build with latest code mounts: `docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile gpu up -d --no-build`.
6. Follow app logs: `docker compose logs -f app-yoi-gpu`.
7. Confirm startup profile line: `Runtime     : profile=gpu ...`.
8. Optional stream logs: `docker compose logs -f mediamtx`.
9. Stop when complete: `docker compose --profile gpu down`.

## Service Lifecycle

### CPU Profile

- First run (build): `docker compose --profile cpu up -d --build`
- Next runs (no rebuild): `docker compose --profile cpu up -d`
- Next runs with local code sync (no rebuild): `docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile cpu up -d --no-build`
- Logs (standard): `docker compose logs -f app-yoi-cpu`
- Optional app logs with tail: `docker compose logs --tail 200 -f app-yoi-cpu`
- Optional MediaMTX logs: `docker compose logs -f mediamtx`
- Startup profile check: `Runtime     : profile=cpu ...`
- Stop profile stack: `docker compose --profile cpu down`

### GPU Profile

- First run (build): `docker compose --profile gpu up -d --build`
- Next runs (no rebuild): `docker compose --profile gpu up -d`
- Next runs with local code sync (no rebuild): `docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile gpu up -d --no-build`
- Logs (standard): `docker compose logs -f app-yoi-gpu`
- Optional app logs with tail: `docker compose logs --tail 200 -f app-yoi-gpu`
- Optional MediaMTX logs: `docker compose logs -f mediamtx`
- Startup profile check: `Runtime     : profile=gpu ...`
- Stop profile stack: `docker compose --profile gpu down`

### Builder Profile

- First run (build): `docker compose --profile builder up -d --build config-builder`
- Next runs: `docker compose --profile builder up -d config-builder`
- Logs: `docker compose logs -f config-builder`
- Stop service only: `docker compose stop config-builder`
- Stop profile stack: `docker compose --profile builder down`

## Stop and Down Scope

- Stop one service only:
  - `docker compose stop app-yoi-cpu`
  - `docker compose stop app-yoi-gpu`
  - `docker compose stop mediamtx`
  - `docker compose stop config-builder`
- Down one profile stack:
  - `docker compose --profile cpu down`
  - `docker compose --profile gpu down`
  - `docker compose --profile builder down`
- Down all active services in this project:
  - `docker compose down`

## Error-Only Logs

- Linux/macOS/Git Bash:
  - `docker compose logs -f app-yoi-cpu | grep -Ei "error|exception|traceback"`
- PowerShell:
  - `docker compose logs -f app-yoi-cpu | Select-String -Pattern "error|exception|traceback"`

## Quality and Validation

- Helper QA: `./yoi/devtools/dev.ps1 -Action qa`
- Helper QA with fix: `./yoi/devtools/dev.ps1 -Action qa-fix`
- Direct Ruff: `D:/AssistXenterprise/yoi/.venv/Scripts/python.exe -m ruff check yoi src config_builder tests --select I,F401,F841`
- Direct Pytest: `D:/AssistXenterprise/yoi/.venv/Scripts/python.exe -m pytest -q`

## Local Development Guidance

- Use no-build runs for iterative code changes.
- Rebuild only when dependency or image layers change.
- Keep log monitoring active during feature tuning.
- Perform clean compose down at the end of each session.

## Related Documents

- [Main README](../../README.md)
- [Before Run Checklist](before-run-checklist.md)
- [Architecture](architecture.md)
- [Runtime Flow](runtime-flow.md)
- [Configuration Builder](configuration-builder.md)
