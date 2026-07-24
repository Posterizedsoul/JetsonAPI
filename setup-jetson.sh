#!/usr/bin/env bash
#
# Orin Nano 8GB: raw Jetson Linux -> running inference server, one command.
#
#   sudo ./setup-jetson.sh                  # everything, then deploy
#   sudo ./setup-jetson.sh --dry-run        # print what it would do, change nothing
#   sudo ./setup-jetson.sh base docker      # named steps only
#   sudo ./setup-jetson.sh deploy           # rebuild and restart the stack
#   sudo ./setup-jetson.sh nvme-format      # destructive, asks first
#
# Every step is idempotent: re-running is safe and is how you recover from a
# partial run. Everything is logged to /var/log/jetson-setup.log.
#
# Differences from the x86 guides you will find online are marked "x86:".

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-/var/log/jetson-setup.log}"
LOCK_FILE=/var/run/jetson-setup.lock

NVME_MOUNT="${NVME_MOUNT:-/mnt/nvme}"
DATA_DIR="${DATA_DIR:-$NVME_MOUNT/jetsonapi}"
SWAP_GB="${SWAP_GB:-8}"
NVME_DEV="${NVME_DEV:-/dev/nvme0n1}"
DRY_RUN="${DRY_RUN:-0}"

# Candidate CUDA-enabled PyTorch base images, best first. The first one that
# pulls wins — NGC tag availability lags L4T releases, so a fixed tag breaks
# on new JetPack versions.
L4T_TORCH_IMAGES=(
    "nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.2-py3"
    "dustynv/l4t-pytorch:r36.2.0"
    "dustynv/l4t-pytorch:r36.4.0"
)

readonly RED=$'\033[31m' GREEN=$'\033[32m' YELLOW=$'\033[33m' BLUE=$'\033[34m'
readonly BOLD=$'\033[1m' RESET=$'\033[0m'

NEEDS_REBOOT=0
declare -a WARNINGS=()

# ------------------------------------------------------------------ plumbing --

log()   { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >>"$LOG_FILE"; }
step()  { printf '\n%s==> %s%s\n' "$BOLD$BLUE" "$*" "$RESET"; log "STEP: $*"; }
info()  { printf '    %s\n' "$*"; log "  $*"; }
ok()    { printf '    %s✓%s %s\n' "$GREEN" "$RESET" "$*"; log "  OK: $*"; }
warn()  { printf '    %s!%s %s\n' "$YELLOW" "$RESET" "$*"; log "  WARN: $*"
          WARNINGS+=("$*"); }
die()   { printf '\n%sERROR:%s %s\n' "$RED$BOLD" "$RESET" "$*" >&2; log "FATAL: $*"
          exit 1; }

on_error() {
    local line=$1 cmd=$2
    printf '\n%sFAILED%s at line %s: %s\n' "$RED$BOLD" "$RESET" "$line" "$cmd" >&2
    printf 'Log: %s\n' "$LOG_FILE" >&2
    printf 'Fix the cause and re-run — every step is idempotent.\n' >&2
    log "FATAL line $line: $cmd"
}
trap 'on_error ${LINENO} "$BASH_COMMAND"' ERR

# run: the single execution point, so --dry-run is honest and everything lands
# in the log. Never use bare commands for anything with a side effect.
run() {
    log "RUN: $*"
    if [[ $DRY_RUN == 1 ]]; then
        printf '    %s[dry-run]%s %s\n' "$YELLOW" "$RESET" "$*"
        return 0
    fi
    "$@" >>"$LOG_FILE" 2>&1
}

# Same, but for shell snippets needing redirection or pipes.
run_sh() {
    log "RUN: $1"
    if [[ $DRY_RUN == 1 ]]; then
        printf '    %s[dry-run]%s %s\n' "$YELLOW" "$RESET" "$1"
        return 0
    fi
    bash -c "$1" >>"$LOG_FILE" 2>&1
}

# Transient apt/network failures are the single most common reason a bring-up
# script dies halfway. Retry them.
retry() {
    local tries=${RETRIES:-3} n=1
    until "$@" >>"$LOG_FILE" 2>&1; do
        if (( n >= tries )); then
            return 1
        fi
        warn "attempt $n failed, retrying in $((n * 5))s: $*"
        sleep $((n * 5))
        ((n++))
    done
    return 0
}

apt_install() {
    [[ $DRY_RUN == 1 ]] && { printf '    %s[dry-run]%s apt-get install %s\n' \
        "$YELLOW" "$RESET" "$*"; return 0; }
    DEBIAN_FRONTEND=noninteractive retry apt-get install -y -o Dpkg::Use-Pty=0 "$@"
}

need_root() { [[ ${EUID} -eq 0 ]] || die "run with sudo: sudo $0 $*"; }
real_user() { echo "${SUDO_USER:-root}"; }
have()      { command -v "$1" >/dev/null 2>&1; }

confirm() {
    local reply
    [[ ${ASSUME_YES:-0} == 1 ]] && return 0
    printf '\n%s%s%s\n' "$YELLOW$BOLD" "$*" "$RESET"
    read -r -p "    Type 'yes' to continue: " reply </dev/tty
    [[ $reply == "yes" ]] || die "aborted by user"
}

# ---------------------------------------------------------------- preflight --

step_preflight() {
    step "Preflight"

    [[ "$(uname -m)" == "aarch64" ]] || die "not arm64 — this is Jetson-only"
    if [[ ! -f /etc/nv_tegra_release ]]; then
        die "no /etc/nv_tegra_release — this is not a Jetson Linux install"
    fi

    local l4t_major l4t_minor model
    l4t_major=$(sed -n 's/^# R\([0-9]*\).*/\1/p' /etc/nv_tegra_release)
    l4t_minor=$(sed -n 's/.*REVISION: \([0-9]*\).*/\1/p' /etc/nv_tegra_release)
    model=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown)

    info "Model:  $model"
    info "L4T:    R${l4t_major}.${l4t_minor}  (JetPack $( [[ ${l4t_major:-0} == 36 ]] && echo 6.x || echo unknown ))"
    info "Kernel: $(uname -r)"
    info "RAM:    $(free -h | awk '/^Mem:/{print $2}')"
    info "Root fs free: $(df -h / | awk 'NR==2{print $4}')"

    [[ ${l4t_major:-0} == 36 ]] || warn "expected L4T R36 (JetPack 6.x); continuing"
    grep -qi orin <<<"$model" || warn "tuned for Orin Nano; '$model' may differ"

    # nvidia-jetpack alone is several GB. Failing here beats failing at 90%.
    local free_mb
    free_mb=$(df --output=avail -m / | tail -1)
    if (( free_mb < 12000 )); then
        warn "only ${free_mb}MB free on / — nvidia-jetpack needs ~12GB"
    else
        ok "disk space sufficient (${free_mb}MB free)"
    fi

    # Reachability, not a 200: the repo root returns 403 (it's a package host,
    # not a website), so -f would wrongly fail on a perfectly good network.
    # Any HTTP status back means DNS + TCP + TLS all worked; only 000 (no
    # connection) is a real failure.
    local code
    code=$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' \
           https://repo.download.nvidia.com 2>>"$LOG_FILE" || echo 000)
    if [[ $code != 000 ]]; then
        ok "network reachable (NVIDIA repo answered HTTP $code)"
    else
        die "cannot reach repo.download.nvidia.com — check networking first"
    fi
}

# --------------------------------------------------------------------- base --

step_base() {
    need_root
    step "Base packages"
    run_sh "apt-get update -o Dpkg::Use-Pty=0" || warn "apt-get update had errors"
    apt_install curl wget git jq htop nvme-cli tree unzip rsync parted \
                build-essential python3-pip python3-venv \
                ca-certificates gnupg lsb-release
    ok "base packages installed"

    if have jtop; then
        ok "jetson-stats already present"
    else
        # --break-system-packages: JetPack 6 ships PEP 668 marked Python.
        run_sh "pip3 install --quiet --break-system-packages jetson-stats" \
            || warn "jetson-stats install failed (non-fatal)"
        ok "jetson-stats installed — 'jtop' shows live GPU/RAM/thermals"
    fi
}

# ------------------------------------------------------------------ jetpack --

step_jetpack() {
    need_root
    step "JetPack runtime (CUDA, cuDNN, TensorRT)"
    # A raw Jetson Linux flash has the kernel and NVIDIA drivers but NOT CUDA.
    # x86: there is no .run installer and no separate driver package here. CUDA
    # arrives via apt from NVIDIA's L4T repo, version-matched to the kernel
    # already on the board. Never install a generic cuda-toolkit on a Jetson.
    if [[ ! -f /etc/apt/sources.list.d/nvidia-l4t-apt-source.list ]]; then
        local major minor rel
        major=$(sed -n 's/^# R\([0-9]*\).*/\1/p' /etc/nv_tegra_release)
        minor=$(sed -n 's/.*REVISION: \([0-9]*\).*/\1/p' /etc/nv_tegra_release)
        rel="r${major}.${minor}"
        info "adding NVIDIA L4T apt source ($rel)"
        run_sh "cat >/etc/apt/sources.list.d/nvidia-l4t-apt-source.list <<'EOF'
deb https://repo.download.nvidia.com/jetson/common $rel main
deb https://repo.download.nvidia.com/jetson/t234 $rel main
EOF"
        run_sh "apt-get update -o Dpkg::Use-Pty=0" \
            || warn "apt update failed after adding the NVIDIA source"
    fi

    if dpkg -l nvidia-jetpack 2>/dev/null | grep -q '^ii'; then
        ok "nvidia-jetpack already installed"
    else
        info "installing nvidia-jetpack — several GB, expect 15-40 minutes"
        apt_install nvidia-jetpack || die "nvidia-jetpack install failed (see $LOG_FILE)"
        ok "nvidia-jetpack installed"
    fi

    if [[ -d /usr/local/cuda/bin ]]; then
        run_sh "cat >/etc/profile.d/cuda.sh <<'EOF'
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
EOF"
        ok "CUDA on PATH for new shells"
    fi
}

# -------------------------------------------------------------------- power --

step_power() {
    need_root
    step "Power mode and clocks"
    # x86: no equivalent. Jetson boards ship power-capped and stay that way
    # across reboots until nvpmodel changes it.
    if ! have nvpmodel; then
        warn "nvpmodel missing — run the jetpack step first"
        return 0
    fi

    # Pick the best mode by PARSING the board's own config rather than
    # assuming mode 0. Orin Nano is 0=15W/1=7W, but Orin Nano Super exposes
    # MAXN at a different id, and hardcoding 0 silently leaves performance on
    # the table there.
    local best_id best_name
    best_id=$(awk -F'ID=' '/< POWER_MODEL/ {split($2,a," "); print a[1]}' \
              /etc/nvpmodel.conf 2>/dev/null | sort -n | tail -1)
    if grep -q 'NAME=MAXN' /etc/nvpmodel.conf 2>/dev/null; then
        best_id=$(awk -F'ID=' '/< POWER_MODEL.*NAME=MAXN/ {split($2,a," "); print a[1]}' \
                  /etc/nvpmodel.conf | head -1)
        best_name="MAXN"
    else
        best_name=$(awk -F'NAME=' -v id="$best_id" \
                    '$0 ~ "ID="id" " {split($2,a," "); print a[1]}' \
                    /etc/nvpmodel.conf 2>/dev/null | head -1)
    fi
    best_id=${best_id:-0}

    info "selecting power mode $best_id (${best_name:-unnamed})"
    run nvpmodel -m "$best_id" || warn "nvpmodel -m $best_id failed"
    ok "power mode set to $best_id"

    if [[ "${SKIP_JETSON_CLOCKS:-0}" != "1" ]]; then
        run jetson_clocks || warn "jetson_clocks failed (non-fatal)"
        # jetson_clocks does not persist across reboot; a oneshot unit re-applies
        # it. Ordered After nvpmodel so the mode is set before clocks are pinned.
        run_sh "cat >/etc/systemd/system/jetson-clocks.service <<'EOF'
[Unit]
Description=Pin Jetson clocks to maximum
After=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF"
        run systemctl daemon-reload
        run systemctl enable jetson-clocks.service
        ok "clocks pinned, and re-pinned on every boot"
    fi
}

step_headless() {
    need_root
    step "Headless mode"
    # ~800MB of an 8GB unified pool, back. Cheapest memory win on the board.
    if [[ "$(systemctl get-default 2>/dev/null)" == "multi-user.target" ]]; then
        ok "already headless"
    else
        run systemctl set-default multi-user.target
        ok "desktop disabled — frees roughly 800MB"
        NEEDS_REBOOT=1
    fi
}

# --------------------------------------------------------------------- nvme --

step_nvme() {
    need_root
    step "NVMe data volume"
    if [[ ! -b $NVME_DEV ]]; then
        warn "no NVMe at $NVME_DEV — data will live on the boot device"
        run mkdir -p "$DATA_DIR"
        return 0
    fi

    info "device: $(lsblk -dno SIZE,MODEL "$NVME_DEV" 2>/dev/null | xargs || echo "$NVME_DEV")"

    if mountpoint -q "$NVME_MOUNT"; then
        ok "$NVME_MOUNT already mounted"
    else
        local part="${NVME_DEV}p1"
        if [[ ! -b $part ]]; then
            warn "$part does not exist — run: $0 nvme-format"
            return 0
        fi
        run mkdir -p "$NVME_MOUNT"
        run mount "$part" "$NVME_MOUNT"
        local uuid
        uuid=$(blkid -s UUID -o value "$part" 2>/dev/null || echo "")
        if [[ -n $uuid ]] && ! grep -q "$uuid" /etc/fstab; then
            # nofail: a missing disk must not block boot into a shell.
            run_sh "echo 'UUID=$uuid $NVME_MOUNT ext4 defaults,noatime,nofail 0 2' >>/etc/fstab"
        fi
        ok "mounted $part at $NVME_MOUNT (persisted in fstab)"
    fi

    run mkdir -p "$DATA_DIR/postgres" "$DATA_DIR/minio" "$DATA_DIR/models"
    ok "data directories ready at $DATA_DIR"
}

step_swap() {
    need_root
    step "Swap"
    # Jetson enables zram by default. zram compresses into RAM, competing with
    # the model for the very resource it extends. On NVMe a real swapfile is
    # better: a cold model load pages from disk instead of evicting the
    # resident model.
    if ! mountpoint -q "$NVME_MOUNT"; then
        warn "$NVME_MOUNT not mounted — skipping swap"
        return 0
    fi
    if [[ -f "$NVME_MOUNT/swapfile" ]]; then
        ok "swapfile already present"
    else
        info "creating ${SWAP_GB}GB swapfile"
        run fallocate -l "${SWAP_GB}G" "$NVME_MOUNT/swapfile"
        run chmod 600 "$NVME_MOUNT/swapfile"
        run mkswap "$NVME_MOUNT/swapfile"
        run swapon "$NVME_MOUNT/swapfile"
        grep -q "$NVME_MOUNT/swapfile" /etc/fstab 2>/dev/null || \
            run_sh "echo '$NVME_MOUNT/swapfile none swap sw 0 0' >>/etc/fstab"
        ok "${SWAP_GB}GB swap active and persisted"
    fi

    if systemctl is-enabled nvzramconfig >/dev/null 2>&1; then
        run systemctl disable nvzramconfig
        ok "zram disabled in favour of NVMe swap"
        NEEDS_REBOOT=1
    fi
}

step_nvme_format() {
    need_root
    step "NVMe format (DESTRUCTIVE)"
    [[ -b $NVME_DEV ]] || die "no NVMe device at $NVME_DEV"
    lsblk "$NVME_DEV" || true
    confirm "This ERASES EVERYTHING on $NVME_DEV. The boot device is untouched."
    run wipefs -a "$NVME_DEV"
    run parted -s "$NVME_DEV" mklabel gpt mkpart primary ext4 0% 100%
    run udevadm settle
    sleep 2
    run mkfs.ext4 -F -L jetson-data "${NVME_DEV}p1"
    ok "formatted ${NVME_DEV}p1"
    step_nvme
}

step_nvme_boot() {
    need_root
    step "Move root filesystem to NVMe (DESTRUCTIVE)"
    # x86: nothing like this. There is no BIOS boot order to change — the Orin
    # bootloader lives in QSPI flash and takes the rootfs from the kernel
    # command line, so migration means copying the rootfs and editing
    # extlinux.conf.
    cat <<'EOF'
    Copies the running rootfs to NVMe and repoints the bootloader.

    Recovery path if it does not come back: the current boot device is
    untouched. Power off, remove the NVMe, boot as before, and restore
    /boot/extlinux/extlinux.conf from the .backup file this writes.
EOF
    confirm "Migrate the root filesystem to NVMe?"
    mountpoint -q "$NVME_MOUNT" || die "$NVME_MOUNT not mounted — run '$0 nvme' first"

    local target="$NVME_MOUNT/rootfs"
    run mkdir -p "$target"
    info "copying rootfs (several minutes)"
    run_sh "rsync -aAXH --delete \
        --exclude={'/dev/*','/proc/*','/sys/*','/tmp/*','/run/*','/mnt/*','/media/*','/lost+found','$NVME_MOUNT/*'} \
        / '$target/'"

    local uuid part="${NVME_DEV}p1"
    uuid=$(blkid -s UUID -o value "$part")
    [[ -n $uuid ]] || die "could not read UUID of $part"
    run cp /boot/extlinux/extlinux.conf /boot/extlinux/extlinux.conf.backup
    run_sh "sed -i 's|root=[^ ]*|root=UUID=$uuid|' /boot/extlinux/extlinux.conf"
    ok "bootloader points at UUID=$uuid (backup: extlinux.conf.backup)"
    NEEDS_REBOOT=1
}

# ------------------------------------------------------------------- docker --

step_docker() {
    need_root
    step "Docker and the NVIDIA container runtime"

    if have docker; then
        ok "docker present ($(docker --version 2>/dev/null | head -1))"
    else
        # docker.io from Ubuntu is the build NVIDIA tests against on L4T.
        apt_install docker.io
        ok "docker installed"
    fi

    # docker.io does NOT ship Compose v2, and this stack is compose-based.
    # Install the plugin binary directly — adding Docker's apt repo alongside
    # the L4T repos is a known source of conflicts on JetPack.
    if docker compose version >/dev/null 2>&1; then
        ok "compose v2 present ($(docker compose version --short 2>/dev/null))"
    else
        local plugin_dir=/usr/local/lib/docker/cli-plugins
        local url="https://github.com/docker/compose/releases/latest/download/docker-compose-linux-aarch64"
        info "installing Compose v2 plugin (arm64)"
        run mkdir -p "$plugin_dir"
        if retry curl -fsSL --max-time 120 -o "$plugin_dir/docker-compose" "$url"; then
            run chmod +x "$plugin_dir/docker-compose"
            ok "compose v2 installed"
        else
            die "could not download the Compose v2 plugin from $url"
        fi
    fi

    # x86: you install nvidia-container-toolkit from NVIDIA's repo and pass
    # --gpus all. On Jetson --gpus all does NOT work — the runtime must be the
    # DEFAULT runtime so it can bind-mount the L4T CUDA libraries into every
    # container, including during `docker build`.
    if ! dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q '^ii'; then
        apt_install nvidia-container-toolkit \
            || warn "nvidia-container-toolkit unavailable — run the jetpack step first"
    fi

    local daemon=/etc/docker/daemon.json
    if grep -q '"default-runtime": *"nvidia"' "$daemon" 2>/dev/null; then
        ok "nvidia already the default docker runtime"
    else
        [[ -f $daemon ]] && run cp "$daemon" "$daemon.backup"
        run mkdir -p /etc/docker
        run_sh "cat >$daemon <<'EOF'
{
  \"default-runtime\": \"nvidia\",
  \"runtimes\": {
    \"nvidia\": {
      \"path\": \"nvidia-container-runtime\",
      \"runtimeArgs\": []
    }
  },
  \"log-driver\": \"json-file\",
  \"log-opts\": { \"max-size\": \"10m\", \"max-file\": \"3\" }
}
EOF"
        run systemctl restart docker
        ok "nvidia set as the default docker runtime"
    fi

    # Keep images and volumes off the boot device.
    if mountpoint -q "$NVME_MOUNT" && ! grep -q '"data-root"' "$daemon" 2>/dev/null; then
        info "moving docker data-root to NVMe"
        run systemctl stop docker
        run mkdir -p "$NVME_MOUNT/docker"
        run_sh "python3 -c \"
import json
p='$daemon'
c=json.load(open(p))
c['data-root']='$NVME_MOUNT/docker'
json.dump(c, open(p,'w'), indent=2)
\""
        run systemctl start docker
        ok "docker data-root is $NVME_MOUNT/docker"
    fi

    run systemctl enable docker
    local u; u=$(real_user)
    if [[ $u != root ]] && ! id -nG "$u" 2>/dev/null | grep -qw docker; then
        run usermod -aG docker "$u"
        warn "added $u to the docker group — log out and back in to use docker unsudoed"
    fi
}

# ---------------------------------------------------------------- tailscale --

step_tailscale() {
    need_root
    step "Tailscale"
    if have tailscale; then
        ok "tailscale already installed"
    else
        retry curl -fsSL -o /tmp/tailscale-install.sh https://tailscale.com/install.sh \
            || die "could not download the Tailscale installer"
        run sh /tmp/tailscale-install.sh
        ok "tailscale installed"
    fi
    run systemctl enable tailscaled
    run systemctl start tailscaled

    if tailscale status >/dev/null 2>&1; then
        ok "connected: $(tailscale ip -4 2>/dev/null | head -1)"
    else
        warn "not logged in yet — run: sudo tailscale up"
        info "this is the only remote access path; do not forward ports"
    fi
}

# ------------------------------------------------------------------- deploy --

pick_base_image() {
    # First image that actually pulls. NGC tag availability lags L4T releases,
    # so a hardcoded tag is a time bomb.
    # Progress goes to stderr: stdout is the return value and must contain
    # nothing but the image name.
    local img
    for img in "${L4T_TORCH_IMAGES[@]}"; do
        info "trying base image $img" >&2
        if [[ $DRY_RUN == 1 ]] || docker pull "$img" >>"$LOG_FILE" 2>&1; then
            echo "$img"
            return 0
        fi
    done
    return 1
}

step_deploy() {
    need_root
    step "Deploy the stack"
    have docker || die "docker missing — run: $0 docker"
    [[ -f "$SCRIPT_DIR/docker-compose.yml" ]] \
        || die "docker-compose.yml not found next to this script"

    local env_file="$SCRIPT_DIR/.env"
    if [[ -f $env_file ]]; then
        ok ".env exists — leaving it alone"
    else
        info "generating .env with random credentials"
        local pg_pw minio_pw base_img
        pg_pw=$(openssl rand -hex 16 2>/dev/null || head -c16 /dev/urandom | xxd -p)
        minio_pw=$(openssl rand -hex 16 2>/dev/null || head -c16 /dev/urandom | xxd -p)

        base_img=$(pick_base_image) || {
            warn "no CUDA base image could be pulled — falling back to CPU torch"
            base_img="python:3.12-slim"
        }

        run_sh "cat >'$env_file' <<EOF
POSTGRES_USER=gibson
POSTGRES_PASSWORD=$pg_pw
POSTGRES_DB=gibson

MINIO_ROOT_USER=jetson
MINIO_ROOT_PASSWORD=$minio_pw
S3_BUCKET=boards

DATA_ROOT=$DATA_DIR
GATEWAY_BASE_IMAGE=$base_img
DOCKER_RUNTIME=nvidia
DEVICE=
EOF"
        run chmod 600 "$env_file"
        ok "wrote .env (base image: $base_img)"
    fi

    info "building and starting containers (first build takes a while)"
    run_sh "cd '$SCRIPT_DIR' && docker compose up -d --build"
    ok "containers started"

    # Health is the real success criterion, not "compose exited 0".
    info "waiting for /health"
    local i
    for i in $(seq 1 60); do
        if [[ $DRY_RUN == 1 ]]; then ok "[dry-run] skipping health wait"; break; fi
        if curl -fsS --max-time 5 http://localhost:8000/health 2>/dev/null | grep -q '"status":"ok"'; then
            ok "gateway healthy after ${i}0s"
            break
        fi
        if (( i == 60 )); then
            run_sh "cd '$SCRIPT_DIR' && docker compose logs --tail 50 gateway"
            die "gateway did not become healthy — see $LOG_FILE"
        fi
        sleep 10
    done

    # CUDA visibility inside the running container is what actually matters.
    if [[ $DRY_RUN != 1 ]]; then
        local cuda
        cuda=$(docker compose -f "$SCRIPT_DIR/docker-compose.yml" exec -T gateway \
               python3 -c "import torch;print(torch.cuda.is_available())" 2>/dev/null | tr -d '\r\n' || echo "?")
        if [[ $cuda == "True" ]]; then
            ok "CUDA available inside the gateway container"
        else
            warn "gateway is running on CPU (torch.cuda.is_available()=$cuda)"
            info "usually means the base image lacks CUDA torch or the nvidia runtime is not default"
        fi
    fi

    # First admin key, once. Re-runs must not mint duplicates.
    local marker="$DATA_DIR/.admin-key-created"
    if [[ -f $marker ]]; then
        ok "admin key already created (see $marker)"
    elif [[ $DRY_RUN == 1 ]]; then
        info "[dry-run] would create the first admin key"
    else
        local out
        out=$(docker compose -f "$SCRIPT_DIR/docker-compose.yml" exec -T gateway \
              python3 /app/scripts/create_key.py admin bootstrap 2>/dev/null || true)
        local key
        key=$(grep -oP 'key:\s*\K\S+' <<<"$out" || true)
        if [[ -n $key ]]; then
            printf '%s\n' "$key" >"$marker"
            chmod 600 "$marker"
            ok "admin API key created and saved to $marker"
        else
            warn "could not create the admin key automatically"
            info "run: docker compose exec gateway python3 /app/scripts/create_key.py admin bootstrap"
        fi
    fi

    [[ $DRY_RUN == 1 ]] || print_access
}

# -------------------------------------------------------------------- access --

# Everything needed to actually use the box, in one block. Printed at the end
# of deploy and re-runnable any time: `sudo ./setup-jetson.sh access`.
print_access() {
    # Every lookup ends in `|| true`: under `set -e` + pipefail a failing probe
    # (Tailscale down, gateway unreachable) would otherwise abort the whole
    # panel instead of just showing that one line as unavailable.
    local ts lan key health
    ts=$(tailscale ip -4 2>/dev/null | head -1 || true)
    lan=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
    health=$(curl -fsS --max-time 5 http://localhost:8000/health 2>/dev/null \
             | grep -o '"status":"[a-z]*"' | cut -d'"' -f4 || true)

    printf '\n%s┌─ Access ────────────────────────────────────────────%s\n' "$BOLD$BLUE" "$RESET"
    printf '%s│%s  Gateway health : %s\n' "$BOLD$BLUE" "$RESET" \
        "${health:-${RED}unreachable${RESET}}"

    if [[ -n $ts ]]; then
        printf '%s│%s  Admin UI       : %shttp://%s:8000/ui%s   (Tailscale, use this remotely)\n' \
            "$BOLD$BLUE" "$RESET" "$GREEN" "$ts" "$RESET"
        printf '%s│%s  API docs       : http://%s:8000/docs\n' "$BOLD$BLUE" "$RESET" "$ts"
    else
        printf '%s│%s  Tailscale      : %snot connected%s — run: sudo tailscale up\n' \
            "$BOLD$BLUE" "$RESET" "$YELLOW" "$RESET"
    fi
    if [[ -n $lan ]]; then
        printf '%s│%s  On the LAN     : http://%s:8000/ui\n' "$BOLD$BLUE" "$RESET" "$lan"
    fi
    printf '%s│%s  On the box     : http://localhost:8000/ui\n' "$BOLD$BLUE" "$RESET"

    local marker="$DATA_DIR/.admin-key-created"
    printf '%s│%s\n' "$BOLD$BLUE" "$RESET"
    if [[ -f $marker ]]; then
        key=$(cat "$marker" 2>/dev/null)
        printf '%s│%s  Admin key      : %s%s%s\n' "$BOLD$BLUE" "$RESET" "$BOLD" "$key" "$RESET"
        printf '%s│%s                   (also saved at %s)\n' "$BOLD$BLUE" "$RESET" "$marker"
    else
        printf '%s│%s  Admin key      : none yet — create one:\n' "$BOLD$BLUE" "$RESET"
        printf '%s│%s    docker compose exec gateway python3 /app/scripts/create_key.py admin you\n' \
            "$BOLD$BLUE" "$RESET"
    fi
    printf '%s│%s\n' "$BOLD$BLUE" "$RESET"
    printf '%s│%s  Logs           : cd %s && docker compose logs -f gateway\n' \
        "$BOLD$BLUE" "$RESET" "$SCRIPT_DIR"
    printf '%s│%s  Restart        : sudo %s deploy\n' "$BOLD$BLUE" "$RESET" "$0"
    printf '%s└─────────────────────────────────────────────────────%s\n' "$BOLD$BLUE" "$RESET"
}

# -------------------------------------------------------------------- verify --

check() {
    local label=$1 cmd=$2
    printf '    %-32s' "$label"
    if eval "$cmd" >/dev/null 2>&1; then
        printf '%s✓%s\n' "$GREEN" "$RESET"
        return 0
    fi
    printf '%s✗%s\n' "$RED" "$RESET"
    return 1
}

step_verify() {
    step "Verification"
    local failed=0

    check "CUDA toolkit"        "test -x /usr/local/cuda/bin/nvcc" || failed=1
    check "nvpmodel"            "command -v nvpmodel" || failed=1
    check "NVMe mounted"        "mountpoint -q $NVME_MOUNT" || failed=1
    check "swap active"         "test -n \"\$(swapon --show --noheadings)\"" || true
    check "docker running"      "docker info" || failed=1
    check "compose v2"          "docker compose version" || failed=1
    check "nvidia default rt"   "docker info --format '{{.DefaultRuntime}}' | grep -q nvidia" || failed=1
    check "tailscale up"        "tailscale status" || true
    check "gateway healthy"     "curl -fsS --max-time 5 http://localhost:8000/health | grep -q '\"status\":\"ok\"'" || failed=1

    echo
    info "Power mode: $(nvpmodel -q 2>/dev/null | tail -1 || echo unknown)"
    info "Memory:     $(free -h | awk '/^Mem:/{print $2" total, "$7" available"}')"
    if mountpoint -q "$NVME_MOUNT"; then
        info "NVMe free:  $(df -h "$NVME_MOUNT" | awk 'NR==2{print $4}')"
    fi
    if tailscale status >/dev/null 2>&1; then
        info "Reach it at: http://$(tailscale ip -4 2>/dev/null | head -1):8000"
    fi

    return $failed
}

# ----------------------------------------------------------------- dispatch --

summary() {
    echo
    if (( ${#WARNINGS[@]} > 0 )); then
        printf '%sWarnings (%d):%s\n' "$YELLOW$BOLD" "${#WARNINGS[@]}" "$RESET"
        printf '  - %s\n' "${WARNINGS[@]}"
    fi
    if (( NEEDS_REBOOT )); then
        printf '\n%sReboot required%s for headless / zram / boot changes to apply.\n' \
            "$YELLOW$BOLD" "$RESET"
        printf 'The stack runs now regardless; reboot when convenient: sudo reboot\n'
    fi
    printf '\n%sDone.%s Log: %s\n' "$BOLD$GREEN" "$RESET" "$LOG_FILE"
}

usage() {
    cat <<EOF
${BOLD}Orin Nano bring-up${RESET}

  sudo $0 [--dry-run] [--yes] [step ...]

Default runs: ${DEFAULT_STEPS[*]}

Steps:
  preflight    identify the board, check disk and network
  base         apt essentials + jetson-stats
  jetpack      CUDA / cuDNN / TensorRT from NVIDIA's L4T repo
  power        best nvpmodel mode + jetson_clocks, persisted
  headless     disable the desktop, frees ~800MB
  nvme         mount NVMe, create data dirs
  swap         ${SWAP_GB}GB swapfile on NVMe, disable zram
  docker       docker + Compose v2 + nvidia as DEFAULT runtime
  tailscale    install and enable
  deploy       generate .env, build, start, wait for health, mint admin key
  access       print IPs, UI/docs URLs, and the admin key
  verify       check everything

Destructive, run explicitly:
  nvme-format  wipe and repartition $NVME_DEV
  nvme-boot    copy rootfs to NVMe and repoint the bootloader

Options:
  --dry-run    print every command, change nothing
  --yes        skip confirmation prompts (destructive steps included)

Env: NVME_MOUNT=$NVME_MOUNT DATA_DIR=$DATA_DIR SWAP_GB=$SWAP_GB NVME_DEV=$NVME_DEV
EOF
}

DEFAULT_STEPS=(preflight base jetpack power headless nvme swap docker tailscale deploy verify)

main() {
    local steps=()
    while (( $# )); do
        case "$1" in
            --dry-run) DRY_RUN=1 ;;
            --yes|-y)  ASSUME_YES=1 ;;
            -h|--help|help) usage; exit 0 ;;
            -*) die "unknown option: $1" ;;
            *) steps+=("$1") ;;
        esac
        shift
    done
    (( ${#steps[@]} == 0 )) && steps=("${DEFAULT_STEPS[@]}")

    if [[ $DRY_RUN == 1 ]]; then
        LOG_FILE=$(mktemp)
        printf '%sDRY RUN — nothing will be changed%s\n' "$BOLD$YELLOW" "$RESET"
    else
        need_root "$@"
        touch "$LOG_FILE" 2>/dev/null || LOG_FILE=/tmp/jetson-setup.log
        # One run at a time: two concurrent apt runs will deadlock.
        exec 9>"$LOCK_FILE"
        flock -n 9 || die "another run is in progress (lock: $LOCK_FILE)"
    fi
    log "=== start: ${steps[*]} ==="

    local s
    for s in "${steps[@]}"; do
        case "$s" in
            preflight)   step_preflight ;;
            base)        step_base ;;
            jetpack)     step_jetpack ;;
            power)       step_power ;;
            headless)    step_headless ;;
            nvme)        step_nvme ;;
            swap)        step_swap ;;
            docker)      step_docker ;;
            tailscale)   step_tailscale ;;
            deploy)      step_deploy ;;
            access)      print_access ;;
            verify)      step_verify || warn "some verification checks failed" ;;
            nvme-format) step_nvme_format ;;
            nvme-boot)   step_nvme_boot ;;
            *) die "unknown step: $s (try --help)" ;;
        esac
    done
    summary
}

main "$@"
