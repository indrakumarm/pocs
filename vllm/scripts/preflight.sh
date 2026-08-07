#!/usr/bin/env bash
# =============================================================================
# preflight.sh — Host readiness checker for the LLM Serving Platform
#
# Run this on the GPU box BEFORE starting any containers.
# It validates every hard dependency so there are no surprises at runtime.
#
# Usage:
#   bash scripts/preflight.sh
#   bash scripts/preflight.sh --fix      # attempt automatic fixes (see below)
#
# What it checks:
#   [GPU]      nvidia-smi present, driver version, VRAM, GPU count
#   [CUDA]     CUDA version exposed by driver (informational)
#   [DOCKER]   docker daemon running, version, API version
#   [RUNTIME]  NVIDIA Container Runtime registered with Docker
#   [DISK]     Free space on HF cache path (warns if < 100GB)
#   [PORTS]    8000 (vLLM), 9400 (DCGM), 9100 (node-exp), 3000 (grafana),
#              9090 (prometheus), 80 (nginx), 443 (nginx TLS) are all free
#   [NETWORK]  Can reach huggingface.co (model downloads) and ghcr.io (images)
#   [COMPOSE]  docker compose v2 available
#
# Exit code:
#   0  — all checks passed (or only warnings)
#   1  — one or more REQUIRED checks failed
# =============================================================================

set -uo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

pass()  { echo -e "  ${GREEN}[PASS]${RESET} $*"; }
fail()  { echo -e "  ${RED}[FAIL]${RESET} $*"; FAILED=$((FAILED+1)); }
warn()  { echo -e "  ${YELLOW}[WARN]${RESET} $*"; }
info()  { echo -e "  ${CYAN}[INFO]${RESET} $*"; }
header(){ echo -e "\n${BOLD}━━━ $* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; }

FAILED=0
FIX_MODE=false
[[ "${1:-}" == "--fix" ]] && FIX_MODE=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Shared fixer: NVIDIA container-toolkit apt/yum repo ─────────────────────
# A stale or half-configured nvidia-container-toolkit repo (missing/mismatched
# GPG key) breaks `apt-get update` for EVERYTHING on the box, not just NVIDIA
# packages — including the plain `apt-get update` that get.docker.com runs
# internally. That was the root cause of the Docker install failure: this
# repo was left broken from an earlier run, so Docker's own install aborted
# before it ever got to install docker-ce. We fix/verify this repo up front,
# before anything else touches apt, so downstream installs (Docker included)
# never trip over it again.
NVIDIA_REPO_FIXED=false
fix_nvidia_apt_repo() {
    $NVIDIA_REPO_FIXED && return 0
    if command -v apt-get &>/dev/null; then
        info "  Ensuring nvidia-container-toolkit apt repo is correctly signed..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        if sudo apt-get update -qq 2>/tmp/apt_update_err; then
            NVIDIA_REPO_FIXED=true
        else
            warn "  apt-get update still failing after re-signing nvidia-container-toolkit repo:"
            sed 's/^/    /' /tmp/apt_update_err
        fi
    elif command -v dnf &>/dev/null; then
        info "  Ensuring nvidia-container-toolkit yum repo is present..."
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
            | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
        NVIDIA_REPO_FIXED=true
    fi
}

docker_service_exists() {
    systemctl list-unit-files --type=service --no-legend 2>/dev/null | grep -q '^docker\.service'
}

# ── Minimum version requirements ─────────────────────────────────────────────
MIN_DRIVER_MAJOR=525          # Supports CUDA 12.x; vLLM requires CUDA 12+
MIN_DOCKER_MAJOR=24           # Compose v2 stable; GPU device support solid
MIN_COMPOSE_MAJOR=2
MIN_FREE_DISK_GB=50           # Absolute floor; 100GB recommended for 32B+ models
WARN_FREE_DISK_GB=100
HF_CACHE_PATH="${HF_HOME:-${HOME}/.cache/huggingface}"

# ── Ports required by the platform ───────────────────────────────────────────
declare -A PLATFORM_PORTS=(
    [8000]="vLLM engine"
    [9400]="DCGM GPU exporter"
    [9100]="Node exporter"
    [9090]="Prometheus"
    [3000]="Grafana"
    [80]="Nginx HTTP"
    [443]="Nginx HTTPS"
)

echo -e "\n${BOLD}╔══════════════════════════════════════════╗"
echo -e "║   LLM Serving Platform — Preflight Check  ║"
echo -e "╚══════════════════════════════════════════╝${RESET}"
echo "  Host: $(hostname)  |  Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  OS:   $(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d'"' -f2 || uname -s)"
echo "  Arch: $(uname -m)"

if $FIX_MODE; then
    header "0. PRE-FLIGHT REPO SANITY"
    info "  --fix: repairing nvidia-container-toolkit apt/yum repo before touching Docker"
    info "  (a broken repo here breaks apt-get update for every other install below)"
    fix_nvidia_apt_repo
fi

# ═════════════════════════════════════════════════════════════════════════════
header "1. NVIDIA DRIVER & GPU"

if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found — NVIDIA driver not installed or not in PATH"
    if $FIX_MODE; then
        CUDA_INSTALL_SCRIPT="${SCRIPT_DIR}/cuda_installation.sh"
        if [[ -f "$CUDA_INSTALL_SCRIPT" ]]; then
            warn "  --fix: running ${CUDA_INSTALL_SCRIPT} to install the NVIDIA driver..."
            sudo bash "$CUDA_INSTALL_SCRIPT"
            echo ""
            fail "Driver install script ran — a REBOOT is required before the driver is usable."
            info "  Run 'sudo reboot', then re-run this preflight check afterward."
            info "  Stopping here; Docker/runtime checks below would be unreliable pre-reboot."
            echo -e "\n  ${RED}${BOLD}1 CHECK(S) FAILED${RESET} — reboot required, then re-run preflight."
            exit 1
        else
            warn "  --fix: ${CUDA_INSTALL_SCRIPT} not found — cannot auto-install driver."
            info "    Run scripts/cuda_installation.sh manually first, then re-run preflight."
        fi
    fi
else
    # nvidia-smi is present
    SMI_OUT=$(nvidia-smi --query-gpu=driver_version,name,memory.total,memory.free,temperature.gpu,power.draw,power.limit \
                          --format=csv,noheader,nounits 2>/dev/null || true)

    if [[ -z "$SMI_OUT" ]]; then
        fail "nvidia-smi found but returned no GPU data — driver may be broken"
    else
        GPU_COUNT=$(echo "$SMI_OUT" | wc -l)
        pass "nvidia-smi responsive  |  ${GPU_COUNT} GPU(s) detected"

        GPU_INDEX=0
        while IFS=',' read -r DRIVER GPU_NAME MEM_TOTAL MEM_FREE GPU_TEMP PWR_DRAW PWR_LIMIT; do
            DRIVER=$(echo "$DRIVER" | xargs)
            GPU_NAME=$(echo "$GPU_NAME" | xargs)
            MEM_TOTAL=$(echo "$MEM_TOTAL" | xargs)
            MEM_FREE=$(echo "$MEM_FREE" | xargs)
            GPU_TEMP=$(echo "$GPU_TEMP" | xargs)
            PWR_DRAW=$(echo "$PWR_DRAW" | xargs)
            PWR_LIMIT=$(echo "$PWR_LIMIT" | xargs)

            DRIVER_MAJOR=$(echo "$DRIVER" | cut -d'.' -f1)

            info "GPU ${GPU_INDEX}: ${GPU_NAME}"
            info "  Driver: ${DRIVER}  |  VRAM total: ${MEM_TOTAL} MiB  |  VRAM free: ${MEM_FREE} MiB"
            info "  Temp: ${GPU_TEMP}°C  |  Power: ${PWR_DRAW}/${PWR_LIMIT} W"

            if (( DRIVER_MAJOR >= MIN_DRIVER_MAJOR )); then
                pass "Driver ${DRIVER} >= minimum ${MIN_DRIVER_MAJOR}.x required for CUDA 12+"
            else
                fail "Driver ${DRIVER} < ${MIN_DRIVER_MAJOR}.x — upgrade needed for CUDA 12 / vLLM"
            fi

            # VRAM warnings for common models
            MEM_GB=$(( MEM_TOTAL / 1024 ))
            if (( MEM_TOTAL < 10240 )); then
                warn "Only ${MEM_GB}GB VRAM — limited to very small models (≤7B @ INT4)"
            elif (( MEM_TOTAL < 20480 )); then
                info "~${MEM_GB}GB VRAM — fits 7B-13B models comfortably"
            elif (( MEM_TOTAL < 40960 )); then
                info "~${MEM_GB}GB VRAM — fits up to 32B models (may need quantization)"
            else
                info "~${MEM_GB}GB VRAM — fits 32B+ models; 70B with quantization"
            fi

            GPU_INDEX=$((GPU_INDEX+1))
        done <<< "$SMI_OUT"
    fi

    # CUDA version (from driver)
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs)
    CUDA_FROM_SMI=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $NF}')
    [[ -n "$CUDA_FROM_SMI" ]] && info "CUDA (driver cap): ${CUDA_FROM_SMI}"
fi

# ═════════════════════════════════════════════════════════════════════════════
header "2. DOCKER"

if ! command -v docker &>/dev/null; then
    fail "docker not found — install Docker Engine first"
    if $FIX_MODE; then
        warn "--fix: installing Docker Engine via get.docker.com ..."
        # Belt-and-suspenders: make sure the nvidia-container-toolkit repo isn't
        # broken (it's fixed in step 0 above, but re-check in case this section
        # is ever run standalone) before letting get.docker.com run apt-get update.
        fix_nvidia_apt_repo
        if command -v apt-get &>/dev/null; then
            info "  Detected apt-based system"
            curl -fsSL https://get.docker.com | sudo sh
            sudo usermod -aG docker "$USER" 2>/dev/null || true
            sudo systemctl enable --now docker
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y docker
            sudo systemctl enable --now docker
            sudo usermod -aG docker "$USER" 2>/dev/null || true
        else
            warn "  Unsupported distro — install manually from https://docs.docker.com/engine/install/"
        fi
        # Check the actual systemd unit, not just the binary on PATH — a partial
        # install (e.g. apt-get update failed midway) can leave a docker binary
        # without docker.service ever being created, which is what happened here.
        if command -v docker &>/dev/null && docker_service_exists; then
            pass "Docker installed — re-run preflight (or 'newgrp docker') to verify daemon access"
        else
            fail "Docker installation failed — docker.service unit not found; install manually"
            info "  Likely cause: apt-get update failed partway through. Check for other broken"
            info "  apt repos with: sudo apt-get update"
        fi
    else
        info "  Run with --fix to attempt automatic install, or:"
        info "    curl -fsSL https://get.docker.com | sudo sh"
        info "    sudo usermod -aG docker \$USER && newgrp docker"
    fi
else
    DOCKER_VER=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
    DOCKER_API=$(docker version --format '{{.Server.APIVersion}}' 2>/dev/null || echo "unknown")
    DOCKER_MAJOR=$(echo "$DOCKER_VER" | cut -d'.' -f1)

    if ! docker info &>/dev/null; then
        fail "Docker daemon not running or current user lacks permission"
        info "  Fix: sudo systemctl start docker   and/or   sudo usermod -aG docker \$USER"
    else
        pass "Docker daemon running"

        if [[ "$DOCKER_MAJOR" != "unknown" ]] && (( DOCKER_MAJOR >= MIN_DOCKER_MAJOR )); then
            pass "Docker version ${DOCKER_VER} (API ${DOCKER_API}) >= ${MIN_DOCKER_MAJOR}.x"
        elif [[ "$DOCKER_MAJOR" == "unknown" ]]; then
            warn "Could not parse Docker version string: ${DOCKER_VER}"
        else
            warn "Docker ${DOCKER_VER} < ${MIN_DOCKER_MAJOR}.x — consider upgrading for latest GPU support"
        fi
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
header "3. NVIDIA CONTAINER RUNTIME (docker --gpus all)"

NVIDIA_RUNTIME_OK=false

if docker info 2>/dev/null | grep -q nvidia; then
    pass "NVIDIA runtime registered with Docker"
    NVIDIA_RUNTIME_OK=true

    # Quick smoke test: can we actually run a GPU container?
    echo -n "  Testing docker run --gpus all ..."
    if docker run --rm --gpus all --entrypoint nvidia-smi \
            nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 \
            -L &>/dev/null 2>&1; then
        pass "docker run --gpus all  succeeded — GPU passthrough works"
    else
        warn "NVIDIA runtime registered but test container failed — driver/toolkit mismatch?"
        info "  Try: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    fi
else
    fail "NVIDIA Container Runtime NOT registered with Docker"
    info "  This means 'docker run --gpus all ...' will fail."
    if $FIX_MODE; then
        warn "--fix: attempting to install and configure nvidia-container-toolkit..."
        fix_nvidia_apt_repo
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y nvidia-container-toolkit
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y nvidia-container-toolkit
        else
            warn "  Unsupported distro for auto-fix — install manually from https://docs.nvidia.com/datacenter/cloud-native/"
        fi
        sudo nvidia-ctk runtime configure --runtime=docker
        # Only try to restart docker if the unit actually exists — restarting a
        # nonexistent unit fails silently-ish ("Unit docker.service not found")
        # and masks the real problem, which is that Docker itself never installed.
        if docker_service_exists; then
            sudo systemctl restart docker
        else
            fail "docker.service does not exist — Docker was never installed; fix step 2 first"
        fi
        if docker info 2>/dev/null | grep -q nvidia; then
            pass "NVIDIA runtime registered after --fix"
            NVIDIA_RUNTIME_OK=true
        else
            fail "NVIDIA runtime still not registered after --fix — manual intervention required"
        fi
    else
        info "  Run with --fix to attempt automatic install, or:"
        info "    # Amazon Linux 2023:"
        info "    curl -sL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \\"
        info "      | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo"
        info "    sudo dnf install -y nvidia-container-toolkit"
        info "    sudo nvidia-ctk runtime configure --runtime=docker"
        info "    sudo systemctl restart docker"
        info ""
        info "    # Ubuntu / Debian:"
        info "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \\"
        info "      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        info "    curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \\"
        info "      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        info "    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        info "    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
header "4. DOCKER COMPOSE V2"

if docker compose version &>/dev/null 2>&1; then
    COMPOSE_VER=$(docker compose version --short 2>/dev/null | tr -d 'v')
    COMPOSE_MAJOR=$(echo "$COMPOSE_VER" | cut -d'.' -f1)
    if (( COMPOSE_MAJOR >= MIN_COMPOSE_MAJOR )); then
        pass "docker compose v${COMPOSE_VER} (plugin mode)"
    else
        warn "docker compose v${COMPOSE_VER} — upgrade to v2.x recommended"
    fi
elif command -v docker-compose &>/dev/null; then
    COMPOSE_VER=$(docker-compose version --short 2>/dev/null | tr -d 'v')
    warn "docker-compose (standalone) v${COMPOSE_VER} found — prefer 'docker compose' plugin (v2)"
else
    fail "docker compose not found"
    info "  Install: sudo apt-get install docker-compose-plugin  (Ubuntu)"
    info "           sudo dnf install docker-compose-plugin      (Amazon Linux 2023)"
fi

# ═════════════════════════════════════════════════════════════════════════════
header "5. DISK SPACE  (HF model cache)"

info "HF_CACHE_PATH: ${HF_CACHE_PATH}"
mkdir -p "${HF_CACHE_PATH}" 2>/dev/null || true

if command -v df &>/dev/null; then
    FREE_KB=$(df -k "${HF_CACHE_PATH}" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
    FREE_GB=$(( FREE_KB / 1024 / 1024 ))
    TOTAL_KB=$(df -k "${HF_CACHE_PATH}" 2>/dev/null | awk 'NR==2{print $2}' || echo 0)
    TOTAL_GB=$(( TOTAL_KB / 1024 / 1024 ))
    USED_PCT=$(df -k "${HF_CACHE_PATH}" 2>/dev/null | awk 'NR==2{print $5}' | tr -d '%' || echo "?")

    info "  Filesystem: total ${TOTAL_GB}GB  |  free ${FREE_GB}GB  |  used ${USED_PCT}%"

    if (( FREE_GB < MIN_FREE_DISK_GB )); then
        fail "Only ${FREE_GB}GB free on ${HF_CACHE_PATH} — minimum ${MIN_FREE_DISK_GB}GB required"
        info "  A 32B BF16 model weights alone need ~65GB + KV cache during runtime"
    elif (( FREE_GB < WARN_FREE_DISK_GB )); then
        warn "${FREE_GB}GB free — recommend ${WARN_FREE_DISK_GB}GB+ for 32B+ models"
    else
        pass "${FREE_GB}GB free at ${HF_CACHE_PATH}"
    fi

    # Check if NVMe instance store is mounted (AWS-specific recommendation)
    if mount | grep -q "/mnt/instance_store"; then
        pass "NVMe instance store mounted at /mnt/instance_store"
    else
        info "  /mnt/instance_store not mounted — consider running ubuntu_startup.sh or amazon_linux.sh"
        info "  to mount the ephemeral NVMe drive for large model caches"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
header "6. NETWORK CONNECTIVITY"

check_url() {
    local url="$1"
    local label="$2"
    local required="${3:-warn}"  # 'fail' or 'warn'
    local http_code
    # Use -w to capture status code without -f so we don't fail on 401/redirect.
    # Container registries (Docker Hub, ghcr.io) return HTTP 401 for unauthenticated
    # requests — that is a valid reachability signal, not an error.
    http_code=$(curl -sI --max-time 10 -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    if [[ "$http_code" == "000" ]]; then
        if [[ "$required" == "fail" ]]; then
            fail "${label} NOT reachable (${url}) — check outbound firewall / security group"
        else
            warn "${label} NOT reachable (${url}) — check outbound firewall / security group"
        fi
    else
        pass "${label} reachable (${url}) [HTTP ${http_code}]"
    fi
}

# HuggingFace is required for model downloads
check_url "https://huggingface.co"        "HuggingFace Hub       " fail
# Container registries: 401 = reachable (unauthenticated); 000 = blocked by firewall
# If blocked, 'docker pull' will fail — open outbound TCP 443 to these hosts in your security group
check_url "https://ghcr.io"               "GitHub Container Reg  " warn
check_url "https://registry-1.docker.io"  "Docker Hub            " warn

# ═════════════════════════════════════════════════════════════════════════════
header "7. PORT AVAILABILITY"

for PORT in "${!PLATFORM_PORTS[@]}"; do
    LABEL="${PLATFORM_PORTS[$PORT]}"
    # ss preferred; fall back to netstat
    if command -v ss &>/dev/null; then
        IN_USE=$(ss -tlnp 2>/dev/null | awk '{print $4}' | grep -E ":${PORT}$" || true)
    elif command -v netstat &>/dev/null; then
        IN_USE=$(netstat -tlnp 2>/dev/null | awk '{print $4}' | grep -E ":${PORT}$" || true)
    else
        IN_USE=""
        warn "Neither ss nor netstat available — skipping port ${PORT} check"
    fi

    if [[ -n "$IN_USE" ]]; then
        warn "Port ${PORT} (${LABEL}) already in use — check for conflicting processes"
        info "  ss -tlnp | grep :${PORT}"
    else
        pass "Port ${PORT} free  (${LABEL})"
    fi
done

# ═════════════════════════════════════════════════════════════════════════════
header "8. UTILITY TOOLS"

check_cmd() {
    local cmd="$1"
    local note="${2:-}"
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" --version 2>/dev/null | head -1 || echo "installed")
        pass "${cmd}: ${VER}"
    else
        warn "${cmd} not found${note:+  (${note})}"
    fi
}

check_cmd curl
check_cmd git    "needed for some model downloads"
check_cmd jq     "optional but useful for parsing API responses"

# huggingface-cli
if python3 -c "import huggingface_hub" &>/dev/null 2>&1; then
    HF_CLI_VER=$(python3 -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>/dev/null || echo "installed")
    pass "huggingface_hub (Python): v${HF_CLI_VER} — huggingface-cli available"
else
    warn "huggingface_hub not installed — 'huggingface-cli download' unavailable"
    info "  Install: pip install huggingface_hub"
fi

# ═════════════════════════════════════════════════════════════════════════════
header "9. RUNNING CONTAINERS (potential conflicts)"

if docker info &>/dev/null 2>&1; then
    RUNNING=$(docker ps --format "{{.Names}}\t{{.Image}}\t{{.Ports}}" 2>/dev/null || true)
    if [[ -z "$RUNNING" ]]; then
        info "No containers currently running"
    else
        info "Currently running containers:"
        echo "$RUNNING" | while IFS=$'\t' read -r NAME IMAGE PORTS; do
            echo "    ${NAME}  |  ${IMAGE}  |  ${PORTS}"
        done
        # Check for vllm-server specifically
        if docker ps --format "{{.Names}}" | grep -q "^vllm-server$"; then
            warn "vllm-server container already running — stop it with 'make stop' before re-deploying"
        fi
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
header "SUMMARY"

if (( FAILED == 0 )); then
    echo -e "\n  ${GREEN}${BOLD}ALL CHECKS PASSED${RESET} — box is ready for platform deployment."
    echo ""
    echo "  Next step:"
    echo "    make run VLLM_MODEL=<model> HF_TOKEN=<token>   # start vLLM"
    echo "    docker compose up -d                            # start support stack"
    exit 0
else
    echo -e "\n  ${RED}${BOLD}${FAILED} CHECK(S) FAILED${RESET} — resolve above issues before deploying."
    echo ""
    echo "  Re-run after fixes:"
    echo "    bash scripts/preflight.sh"
    echo ""
    echo "  For automatic fix attempts (NVIDIA runtime install):"
    echo "    bash scripts/preflight.sh --fix"
    exit 1
fi
