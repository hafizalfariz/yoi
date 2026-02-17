# Devtools Command Mapping

This page maps day-to-day dev helper commands between Windows PowerShell and Linux/macOS Bash.

## Scope

- Windows helper script: `yoi/devtools/dev.ps1`
- Linux/macOS helper script: `yoi/devtools/dev.sh`

Both scripts are wrappers for repetitive local operations (compose up/down, logs, QA, and local run).

## Quick Mapping

| Intent | Windows (PowerShell) | Linux/macOS (Bash) |
|---|---|---|
| Fast CPU up (no build) | `./yoi/devtools/dev.ps1 -Action up-cpu` | `./yoi/devtools/dev.sh --action up-cpu` |
| Fast GPU up (no build) | `./yoi/devtools/dev.ps1 -Action up-gpu` | `./yoi/devtools/dev.sh --action up-gpu` |
| Build + up CPU | `./yoi/devtools/dev.ps1 -Action build-cpu` | `./yoi/devtools/dev.sh --action build-cpu` |
| Build + up GPU | `./yoi/devtools/dev.ps1 -Action build-gpu` | `./yoi/devtools/dev.sh --action build-gpu` |
| Generic up by profile | `./yoi/devtools/dev.ps1 -Action up -Profile gpu` | `./yoi/devtools/dev.sh --action up --profile gpu` |
| Show logs | `./yoi/devtools/dev.ps1 -Action logs -Tail 200 -Follow` | `./yoi/devtools/dev.sh --action logs --tail 200 --follow 1` |
| Restart active app service | `./yoi/devtools/dev.ps1 -Action restart -Profile cpu` | `./yoi/devtools/dev.sh --action restart --profile cpu` |
| Run local runtime | `./yoi/devtools/dev.ps1 -Action run-local -Profile cpu -Config configs/app/dwelltime.yaml` | `./yoi/devtools/dev.sh --action run-local --profile cpu --config configs/app/dwelltime.yaml` |
| QA (ruff + pytest) | `./yoi/devtools/dev.ps1 -Action qa` | `./yoi/devtools/dev.sh --action qa` |
| QA fix then test | `./yoi/devtools/dev.ps1 -Action qa-fix` | `./yoi/devtools/dev.sh --action qa-fix` |
| Show percent-to-core table | `./yoi/devtools/dev.ps1 -Action limits` | `./yoi/devtools/dev.sh --action limits` |
| Cleanup old outputs/logs | `./yoi/devtools/dev.ps1 -Action cleanup` | `./yoi/devtools/dev.sh --action cleanup` |

## Memory Policy Behavior

- Source of truth is `.env` value: `MEMORY_LIMIT_PERCENT`.
- On helper run, script resolves it into `MEMORY_LIMIT` for compose runtime.
- If `MEMORY_LIMIT_PERCENT` is empty, compose falls back to service default `mem_limit` in `docker-compose.yml`.

## Linux/macOS Prerequisites

- Make script executable once:
  - `chmod +x yoi/devtools/dev.sh`
- Required commands on PATH:
  - `docker`
  - `bash`
  - `python3` (or project venv python)

## Notes

- Helper scripts are convenience wrappers, not runtime requirements.
- You can always run raw `docker compose ...` commands directly.
