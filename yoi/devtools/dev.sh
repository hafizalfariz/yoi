#!/usr/bin/env bash
set -euo pipefail

ACTION="up"
PROFILE="cpu"
NO_BUILD=1
TAIL=120
FOLLOW=0
PLAYER="vlc"
URL="rtsp://localhost:6554/kluis-line"
NETWORK_CACHING_MS=150
CONFIG="configs/app/dwelltime.yaml"
DRY_RUN=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVICE="app-yoi-cpu"

COMPOSE_FILES=(
  -f "${REPO_ROOT}/docker-compose.yml"
  -f "${REPO_ROOT}/docker-compose.dev.yml"
)

usage() {
  cat <<'USAGE'
Usage: ./yoi/devtools/dev.sh [options]

Options:
  --action <name>          up|up-cpu|up-gpu|build-cpu|build-gpu|run-local|up-builder|down-builder|restart|logs|watch|cleanup|qa|qa-fix|limits
  --profile <cpu|gpu>      profile for action up/run-local (default: cpu)
  --no-build <0|1>         default 1 (only used by action up)
  --tail <n>               log tail line count (default: 120)
  --follow <0|1>           follow logs (default: 0)
  --player <ffplay|vlc>    player for action watch (default: vlc)
  --url <rtsp-url>         stream URL for action watch
  --network-caching-ms <n> VLC network caching in ms (default: 150)
  --config <path>          config file for action run-local
  --dry-run <0|1>          print commands only (default: 0)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action) ACTION="${2:-}"; shift 2 ;;
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --no-build) NO_BUILD="${2:-1}"; shift 2 ;;
    --tail) TAIL="${2:-120}"; shift 2 ;;
    --follow) FOLLOW="${2:-1}"; shift 2 ;;
    --player) PLAYER="${2:-vlc}"; shift 2 ;;
    --url) URL="${2:-}"; shift 2 ;;
    --network-caching-ms) NETWORK_CACHING_MS="${2:-150}"; shift 2 ;;
    --config) CONFIG="${2:-configs/app/dwelltime.yaml}"; shift 2 ;;
    --dry-run) DRY_RUN="${2:-1}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ "${PROFILE}" == "gpu" ]]; then
  SERVICE="app-yoi-gpu"
fi

get_dotenv_value() {
  local key="$1"
  local default_value="${2:-}"
  local env_file="${REPO_ROOT}/.env"

  if [[ ! -f "${env_file}" ]]; then
    printf '%s' "${default_value}"
    return
  fi

  local line
  line="$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "${env_file}" | head -n1 || true)"
  if [[ -z "${line}" ]]; then
    printf '%s' "${default_value}"
    return
  fi

  local value="${line#*=}"
  value="${value#\"}"
  value="${value%\"}"
  printf '%s' "${value}"
}

get_config_value() {
  local key="$1"
  local default_value="${2:-}"
  local env_value="${!key:-}"
  if [[ -n "${env_value}" ]]; then
    printf '%s' "${env_value}"
    return
  fi
  get_dotenv_value "${key}" "${default_value}"
}

get_logical_core_count() {
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  elif command -v nproc >/dev/null 2>&1; then
    nproc
  else
    printf '1'
  fi
}

get_total_memory_mb() {
  if [[ -f /proc/meminfo ]]; then
    awk '/MemTotal:/ { printf "%d", $2/1024; exit }' /proc/meminfo
    return
  fi

  if command -v sysctl >/dev/null 2>&1; then
    local bytes
    bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
    if [[ -n "${bytes}" ]]; then
      awk -v b="${bytes}" 'BEGIN { printf "%d", b/1024/1024 }'
      return
    fi
  fi

  echo "Cannot detect total system memory for MEMORY_LIMIT_PERCENT conversion" >&2
  exit 1
}

convert_percent_to_cpu_limit() {
  local raw="$1"
  local core_count="$2"

  if [[ -z "${raw}" ]]; then
    printf ''
    return
  fi

  local normalized
  normalized="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
  normalized="${normalized%%%}"

  if [[ "${normalized}" == "max" ]]; then
    awk -v c="${core_count}" 'BEGIN { printf "%.1f", c }'
    return
  fi

  if ! [[ "${normalized}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid percent value: '${raw}'" >&2
    exit 1
  fi

  awk -v p="${normalized}" -v c="${core_count}" '
    BEGIN {
      if (p < 10 || p > 100) {
        exit 2
      }
      printf "%.1f", (c * p) / 100.0
    }
  '
  local rc=$?
  if [[ $rc -eq 2 ]]; then
    echo "Percent must be between 10 and 100 (or 'max'), got '${raw}'" >&2
    exit 1
  fi
}

convert_percent_to_memory_limit() {
  local raw="$1"
  local total_mb="$2"

  if [[ -z "${raw}" ]]; then
    printf ''
    return
  fi

  local normalized
  normalized="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
  normalized="${normalized%%%}"

  local percent
  if [[ "${normalized}" == "max" ]]; then
    percent="100"
  else
    if ! [[ "${normalized}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "Invalid MEMORY_LIMIT_PERCENT value: '${raw}'" >&2
      exit 1
    fi
    percent="${normalized}"
  fi

  awk -v p="${percent}" -v t="${total_mb}" '
    BEGIN {
      if (p < 10 || p > 100) {
        exit 2
      }
      m = int((t * p) / 100.0)
      if (m < 512) m = 512
      printf "%dm", m
    }
  '
  local rc=$?
  if [[ $rc -eq 2 ]]; then
    echo "MEMORY_LIMIT_PERCENT must be between 10 and 100 (or 'max'), got '${raw}'" >&2
    exit 1
  fi
}

show_cpu_limit_table() {
  local core_count
  core_count="$(get_logical_core_count)"
  echo "[dev] Logical cores detected: ${core_count}"
  echo "[dev] Percent -> CPU limit (cores)"
  local p
  for p in 10 20 30 40 50 60 70 80 90 100; do
    local limit
    limit="$(convert_percent_to_cpu_limit "${p}" "${core_count}")"
    printf '[dev] %3s%% => %s\n' "${p}" "${limit}"
  done
  local max_limit
  max_limit="$(convert_percent_to_cpu_limit "max" "${core_count}")"
  echo "[dev] max => ${max_limit}"
}

resolve_compose_memory_limit() {
  local percent_raw
  percent_raw="$(get_config_value "MEMORY_LIMIT_PERCENT" "")"
  if [[ -z "${percent_raw}" ]]; then
    unset MEMORY_LIMIT || true
    echo "[dev] MEMORY_LIMIT_PERCENT is empty; using compose default mem_limit"
    return
  fi

  local total_mb
  total_mb="$(get_total_memory_mb)"
  local resolved
  resolved="$(convert_percent_to_memory_limit "${percent_raw}" "${total_mb}")"
  export MEMORY_LIMIT="${resolved}"
  echo "[dev] Resolved MEMORY_LIMIT from MEMORY_LIMIT_PERCENT=${percent_raw} => ${resolved} (host=${total_mb}m)"
}

apply_local_thread_limits_from_percent() {
  local target_profile="$1"
  local core_count percent_key percent_raw
  core_count="$(get_logical_core_count)"
  percent_key="CPU_LIMIT_PERCENT"
  if [[ "${target_profile}" == "gpu" ]]; then
    percent_key="GPU_CPU_LIMIT_PERCENT"
  fi

  percent_raw="$(get_config_value "${percent_key}" "")"
  if [[ -z "${percent_raw}" ]]; then
    return
  fi

  local cpu_limit thread_budget opencv_threads
  cpu_limit="$(convert_percent_to_cpu_limit "${percent_raw}" "${core_count}")"
  thread_budget="$(awk -v n="${cpu_limit}" 'BEGIN { t=int(n); if (t < 1) t=1; print t }')"
  opencv_threads="$(awk -v t="${thread_budget}" 'BEGIN { o=int(t/2); if (o < 1) o=1; if (o > 4) o=4; print o }')"

  export OMP_NUM_THREADS="${thread_budget}"
  export OPENBLAS_NUM_THREADS="${thread_budget}"
  export MKL_NUM_THREADS="${thread_budget}"
  export NUMEXPR_NUM_THREADS="${thread_budget}"
  export OPENCV_FOR_THREADS_NUM="${opencv_threads}"

  echo "[dev] Local thread cap from ${percent_key}=${percent_raw} => OMP/BLAS/MKL/NUMEXPR=${thread_budget}, OpenCV=${opencv_threads}"
}

run_compose() {
  local args=("$@")
  echo "[dev] docker compose ${COMPOSE_FILES[*]} ${args[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  docker compose "${COMPOSE_FILES[@]}" "${args[@]}"
}

resolve_python_command() {
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    printf '%s' "${REPO_ROOT}/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "python3"
    return
  fi
  printf '%s' "python"
}

run_python() {
  local args=("$@")
  local py
  py="$(resolve_python_command)"
  echo "[dev] ${py} ${args[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "${py}" "${args[@]}"
}

start_profile() {
  local target_profile="$1"
  local use_no_build="$2"
  local up_args=(--profile "${target_profile}" up -d)
  if [[ "${use_no_build}" == "1" ]]; then
    up_args+=(--no-build)
  else
    up_args+=(--build)
  fi

  resolve_compose_memory_limit

  local attempt max_attempts=3
  for attempt in 1 2 3; do
    if run_compose "${up_args[@]}"; then
      break
    fi
    if [[ ${attempt} -ge ${max_attempts} ]]; then
      exit 1
    fi
    local delay_seconds=$((4 * attempt))
    echo "[dev] compose up attempt ${attempt}/${max_attempts} failed, retrying in ${delay_seconds}s ..."
    sleep "${delay_seconds}"
  done

  run_compose ps
  echo "[dev] Profile '${target_profile}' started (no-build=${use_no_build})"
}

start_local_runtime() {
  local config_path="$1"
  local target_profile="$2"
  local resolved_config="${config_path}"
  if [[ "${config_path}" != /* ]]; then
    resolved_config="${REPO_ROOT}/${config_path}"
  fi
  if [[ ! -f "${resolved_config}" ]]; then
    echo "Config file not found: ${resolved_config}" >&2
    exit 1
  fi

  export YOI_RUNTIME_PROFILE="${target_profile}"
  if [[ -z "${YOI_TARGET_DEVICE:-}" ]]; then
    if [[ "${target_profile}" == "gpu" ]]; then
      export YOI_TARGET_DEVICE="cuda"
    else
      export YOI_TARGET_DEVICE="cpu"
    fi
  fi

  apply_local_thread_limits_from_percent "${target_profile}"
  echo "[dev] Local runtime profile: ${target_profile} (target device: ${YOI_TARGET_DEVICE})"
  echo "[dev] Local runtime config: ${resolved_config}"
  run_python src/app/main.py --config "${resolved_config}"
}

cleanup_legacy_logs() {
  local p
  for p in "${REPO_ROOT}/logs"; do
    [[ -d "${p}" ]] || continue
    while IFS= read -r -d '' f; do
      echo "[dev] removing legacy log: ${f}"
      if [[ "${DRY_RUN}" != "1" ]]; then
        rm -f "${f}"
      fi
    done < <(find "${p}" -maxdepth 1 -type f -name 'ffmpeg_startup_*.log' -print0)
  done
}

cleanup_old_output_runs() {
  local retention_raw retention_days output_root_raw output_root
  retention_raw="$(get_dotenv_value "YOI_OUTPUT_RETENTION_DAYS" "30")"
  if ! [[ "${retention_raw}" =~ ^[0-9]+$ ]]; then
    retention_days=30
  else
    retention_days="${retention_raw}"
  fi
  if [[ "${retention_days}" -lt 1 ]]; then
    return
  fi

  output_root_raw="$(get_dotenv_value "OUTPUT_PATH" "./output")"
  if [[ "${output_root_raw}" == /* ]]; then
    output_root="${output_root_raw}"
  else
    output_root="${REPO_ROOT}/${output_root_raw#./}"
  fi
  [[ -d "${output_root}" ]] || return

  while IFS= read -r -d '' dir; do
    echo "[dev] removing old output run (> ${retention_days} days): ${dir}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      rm -rf "${dir}"
    fi
  done < <(find "${output_root}" -mindepth 1 -maxdepth 1 -type d -mtime +"${retention_days}" -print0)
}

cd "${REPO_ROOT}"
cleanup_legacy_logs
cleanup_old_output_runs

case "${ACTION}" in
  cleanup)
    echo "[dev] cleanup completed (legacy logs + old output runs)"
    ;;
  qa)
    run_python -m ruff check yoi src config_builder tests --select I,F401,F841
    run_python -m pytest -q
    echo "[dev] QA completed (ruff + pytest)"
    ;;
  qa-fix)
    run_python -m ruff check yoi src config_builder tests --select I --fix
    run_python -m ruff check yoi src config_builder tests --select I,F401,F841
    run_python -m pytest -q
    echo "[dev] QA-FIX completed (ruff --fix + ruff + pytest)"
    ;;
  limits)
    show_cpu_limit_table
    ;;
  up)
    start_profile "${PROFILE}" "${NO_BUILD}"
    echo "[dev] RTSP endpoint: rtsp://localhost:6554/kluis-line"
    ;;
  up-cpu)
    start_profile cpu 1
    echo "[dev] Fast CPU up complete (no build)"
    ;;
  up-gpu)
    start_profile gpu 1
    echo "[dev] Fast GPU up complete (no build)"
    ;;
  build-cpu)
    start_profile cpu 0
    echo "[dev] CPU build+up complete"
    ;;
  build-gpu)
    start_profile gpu 0
    echo "[dev] GPU build+up complete"
    ;;
  run-local)
    start_local_runtime "${CONFIG}" "${PROFILE}"
    ;;
  up-builder)
    if [[ "${NO_BUILD}" == "1" ]]; then
      run_compose --profile builder up -d config-builder --no-build
    else
      run_compose --profile builder up -d config-builder --build
    fi
    run_compose ps config-builder
    builder_port="$(get_dotenv_value CONFIG_BUILDER_PORT 8032)"
    echo "[dev] Config Builder endpoint: http://localhost:${builder_port}"
    ;;
  down-builder)
    run_compose --profile builder stop config-builder
    run_compose ps config-builder
    echo "[dev] Config Builder stopped"
    ;;
  restart)
    run_compose restart "${SERVICE}"
    run_compose logs --tail=80 "${SERVICE}" mediamtx
    ;;
  logs)
    log_args=(logs --tail "${TAIL}")
    if [[ "${FOLLOW}" == "1" ]]; then
      log_args+=(-f)
    fi
    log_args+=("${SERVICE}" mediamtx)
    run_compose "${log_args[@]}"
    ;;
  watch)
    echo "[dev] Watch RTSP -> ${URL} (${PLAYER})"
    if [[ "${PLAYER}" == "ffplay" ]]; then
      if ! command -v ffplay >/dev/null 2>&1; then
        echo "ffplay tidak ditemukan di PATH" >&2
        exit 1
      fi
      if [[ "${DRY_RUN}" != "1" ]]; then
        ffplay -rtsp_transport tcp -fflags nobuffer -flags low_delay -framedrop "${URL}"
      fi
    else
      if ! command -v vlc >/dev/null 2>&1; then
        echo "vlc tidak ditemukan di PATH" >&2
        exit 1
      fi
      if [[ "${DRY_RUN}" != "1" ]]; then
        vlc "--network-caching=${NETWORK_CACHING_MS}" "${URL}"
      fi
    fi
    ;;
  *)
    echo "Unsupported action: ${ACTION}" >&2
    usage
    exit 1
    ;;
esac
