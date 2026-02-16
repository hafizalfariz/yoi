# Before Run Checklist

Use this checklist before running any YOI profile (`cpu`, `gpu`, or `builder`).

## Execution Mode

- [ ] I will run `cpu`
- [ ] I will run `gpu`
- [ ] I will run `builder`

## 1) Input Data Ready

- [ ] Source video is placed in `input/`
- [ ] Example file exists (for quick test): `input/sample.mp4`
- [ ] Video file is readable and not locked by another app

## 2) Model Files Ready

- [ ] Model path uses structure `models/<model_name>/1/`
- [ ] CPU profile weight exists (recommended): `best.onnx`
- [ ] GPU profile weight exists (recommended): `best.pt`
- [ ] Warning acknowledged: for GPU profile, model folder must contain `best.pt` to avoid CPU fallback.
- [ ] Supported extension confirmed: `.onnx` / `.pt` / `.pth`
- [ ] Example valid paths: `models/person_general_detection/1/best.onnx` and `models/person_general_detection/1/best.pt`
- [ ] I understand runtime selection policy:
	- target `cpu` prefers `.onnx`
	- target `cuda/gpu` prefers `.pt/.pth`
	- strict GPU (`YOI_STRICT_DEVICE=1`) rejects ONNX
- [ ] `metadata.yaml` status is decided:
	- Optional (not required to run)
	- If present, place at `models/<model_name>/1/metadata.yaml`
	- If absent, runtime uses model class metadata fallback

## 3) Configuration Ready

- [ ] Active YAML exists in `configs/app/`
- [ ] Example config exists: `configs/app/kluis-line.yaml`
- [ ] Config model name matches folder in `models/`
- [ ] Config input source matches video/RTSP to be used

## 4) Runtime Environment Ready

- [ ] Docker Engine is running
- [ ] Required ports are available
- [ ] Optional `.env` values are adjusted (paths, limits, tuning)
- [ ] Optional max runtime is set when needed: `YOI_MAX_INFERENCE_SECONDS=<seconds>`

## 5) Startup Commands

- [ ] I will run only one app profile publisher (`cpu` or `gpu`) per RTSP path to avoid publisher overlap.

- [ ] First run with build (CPU): `docker compose --profile cpu up -d --build`
- [ ] First run with build (GPU): `docker compose --profile gpu up -d --build`
- [ ] Next run without build (CPU): `docker compose --profile cpu up -d`
- [ ] Next run without build (GPU): `docker compose --profile gpu up -d`
- [ ] If no-build must include latest source code changes: `docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile <cpu|gpu> up -d --no-build`
- [ ] Builder first run with build: `docker compose --profile builder up -d --build config-builder`

## 6) Log Checks

- [ ] Standard CPU app logs: `docker compose logs -f app-yoi-cpu`
- [ ] Standard GPU app logs: `docker compose logs -f app-yoi-gpu`
- [ ] Startup profile line matches target: `Runtime     : profile=cpu ...` or `Runtime     : profile=gpu ...`
- [ ] Optional app logs with tail: `docker compose logs --tail 200 -f app-yoi-cpu`
- [ ] Optional MediaMTX logs: `docker compose logs -f mediamtx`
- [ ] Error-only filter (PowerShell): `docker compose logs -f app-yoi-cpu | Select-String -Pattern "error|exception|traceback"`

## 7) Safe Stop Policy

- [ ] Stop CPU profile: `docker compose --profile cpu down`
- [ ] Stop GPU profile: `docker compose --profile gpu down`
- [ ] Stop Builder profile: `docker compose --profile builder down`
- [ ] Stop one service only (example): `docker compose stop config-builder`
- [ ] Stop all active services in project: `docker compose down`

## Final Go/No-Go

- [ ] I have completed all required checks above
- [ ] Runtime profile and command path are clear
- [ ] Ready to execute
