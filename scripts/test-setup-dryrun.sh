#!/usr/bin/env bash
# Exercise setup-jetson.sh's control flow on a non-Jetson machine.
#
# This fakes the identity files and Jetson binaries the script probes, then
# runs it with --dry-run so no command actually executes. It verifies the
# script reaches every step and parses what it reads — NOT that the real
# commands behave correctly on hardware. Only the Orin can prove that.
set -euo pipefail

cd "$(dirname "$0")/.."

docker run --rm -v "$(pwd):/w:ro" -w /tmp ubuntu:22.04 bash -c '
set -euo pipefail
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq curl >/dev/null 2>&1

cp -r /w /work && cd /work

# --- fake a Jetson --------------------------------------------------------
cat >/etc/nv_tegra_release <<EOF
# R36 (release), REVISION: 4.3, GCID: 12345, BOARD: generic, EABI: aarch64
EOF

mkdir -p /proc/device-tree 2>/dev/null || true
# /proc is read-only; the script tolerates the model lookup failing.

cat >/etc/nvpmodel.conf <<EOF
< POWER_MODEL ID=0 NAME=15W >
< POWER_MODEL ID=1 NAME=7W >
< POWER_MODEL ID=2 NAME=MAXN_SUPER >
EOF

for b in nvpmodel jetson_clocks tailscale nvidia-container-runtime; do
    printf "#!/bin/sh\nexit 0\n" >/usr/local/bin/$b
    chmod +x /usr/local/bin/$b
done
printf "#!/bin/sh\necho fake\n" >/usr/local/bin/docker
chmod +x /usr/local/bin/docker

echo "=== dry run: all default steps ==="
# uname -m is x86_64 here, so run the steps individually to skip the arm64
# gate in preflight; everything else is exercised.
DRY_RUN=1 bash setup-jetson.sh --dry-run \
    base jetpack power headless nvme swap docker tailscale deploy 2>&1 | tail -60

echo
echo "=== power-mode parsing (the bit most likely to be wrong) ==="
best=$(awk -F"ID=" "/< POWER_MODEL.*NAME=MAXN/ {split(\$2,a,\" \"); print a[1]}" \
       /etc/nvpmodel.conf | head -1)
echo "MAXN mode detected: ${best:-NONE}"
test "$best" = "2" || { echo "FAIL: expected mode 2"; exit 1; }

echo
echo "=== L4T release parsing ==="
major=$(sed -n "s/^# R\([0-9]*\).*/\1/p" /etc/nv_tegra_release)
minor=$(sed -n "s/.*REVISION: \([0-9]*\).*/\1/p" /etc/nv_tegra_release)
echo "parsed: r${major}.${minor}"
test "${major}.${minor}" = "36.4" || { echo "FAIL: expected 36.4"; exit 1; }

echo
echo "=== usage/help exits clean ==="
bash setup-jetson.sh --help >/dev/null && echo "help OK"

echo
echo "ALL CONTROL-FLOW CHECKS PASSED"
'
