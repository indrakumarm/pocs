#!/bin/bash
# -------------------------------------------------------------------------
# cuda_installation.sh
# Native NVIDIA Driver Installer — Amazon Linux 2023 (AL2023) & Ubuntu
# -------------------------------------------------------------------------
set -e  # Exit immediately if any command fails

# ── Detect distro ────────────────────────────────────────────────────────
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
else
    echo "ERROR: /etc/os-release not found — cannot detect OS."
    exit 1
fi

OS_ID="${ID:-unknown}"
OS_ID_LIKE="${ID_LIKE:-}"

# -------------------------------------------------------------------------
# Amazon Linux 2023
# -------------------------------------------------------------------------
install_driver_amazon_linux() {
    echo "=== Detected Amazon Linux (${VERSION_ID:-unknown}) ==="

    echo "=== [1/4] Installing AWS NVIDIA Release Descriptor ==="
    sudo dnf install -y nvidia-release

    echo "=== [2/4] Querying Repository for Exact Kernel Packages ==="
    # Find the exact tracking packages matching your exact running kernel version
    DEVEL_PKG=$(dnf repoquery --whatprovides "kernel-devel-uname-r = $(uname -r)" | head -n 1)
    HEADERS_PKG=$(dnf repoquery --whatprovides "kernel-headers-uname-r = $(uname -r)" | head -n 1)

    if [ -z "$DEVEL_PKG" ] || [ -z "$HEADERS_PKG" ]; then
        echo "ERROR: Could not locate matching kernel packages in the repository."
        exit 1
    fi

    echo "Found matching package: $DEVEL_PKG"

    echo "=== [3/4] Installing Compilers and System Headers ==="
    sudo dnf install -y gcc make "$DEVEL_PKG" "$HEADERS_PKG"

    echo "=== [4/4] Installing AWS-Managed Native NVIDIA CUDA Driver ==="
    sudo dnf install -y nvidia-driver-cuda
}

# -------------------------------------------------------------------------
# Ubuntu
# -------------------------------------------------------------------------
install_driver_ubuntu() {
    echo "=== Detected Ubuntu (${VERSION_ID:-unknown} / ${VERSION_CODENAME:-unknown}) ==="

    echo "=== [1/4] Updating apt and installing kernel headers + build tools ==="
    # DKMS needs headers matching the *running* kernel to build the NVIDIA
    # kernel module, plus a compiler toolchain.
    sudo apt-get update -qq
    sudo apt-get install -y build-essential dkms "linux-headers-$(uname -r)"

    echo "=== [2/4] Installing ubuntu-drivers-common ==="
    sudo apt-get install -y ubuntu-drivers-common

    echo "=== [3/4] Detecting recommended NVIDIA driver for this GPU ==="
    DEVICES_OUT="$(ubuntu-drivers devices 2>/dev/null || true)"
    echo "$DEVICES_OUT"

    echo "=== [4/4] Installing NVIDIA driver ==="
    # Prefer the exact package ubuntu-drivers itself tags "recommended" for this
    # GPU. We do NOT rely on `ubuntu-drivers install --gpgpu` — on some Ubuntu
    # releases none of the listed driver candidates carry a "gpgpu" tag in their
    # metadata even when a perfectly good recommended driver exists, so --gpgpu
    # silently matches nothing ("No drivers found for installation") while the
    # correct package sits right there in `ubuntu-drivers devices` output.
    RECOMMENDED_PKG=$(echo "$DEVICES_OUT" | awk '/driver[[:space:]]*:/ && /recommended/ {print $3; exit}')

    if [[ -n "$RECOMMENDED_PKG" ]]; then
        echo "  Installing recommended driver package: $RECOMMENDED_PKG"
        sudo apt-get update -qq
        sudo apt-get install -y "$RECOMMENDED_PKG"
    else
        echo "  No package tagged 'recommended' in ubuntu-drivers output — falling back to autoinstall"
        sudo ubuntu-drivers autoinstall
    fi
}

# -------------------------------------------------------------------------
# Dispatch by distro
# -------------------------------------------------------------------------
case "$OS_ID" in
    amzn)
        install_driver_amazon_linux
        ;;
    ubuntu)
        install_driver_ubuntu
        ;;
    debian)
        echo "=== Detected Debian — using Ubuntu-style driver install path ==="
        install_driver_ubuntu
        ;;
    *)
        echo "ERROR: Unsupported OS '${PRETTY_NAME:-$OS_ID}' (ID=$OS_ID, ID_LIKE=$OS_ID_LIKE)."
        echo "This script supports Amazon Linux 2023 and Ubuntu/Debian only."
        exit 1
        ;;
esac

echo "=========================================================="
echo " INSTALLATION SUCCESSFUL!                                "
echo " System must be rebooted to activate the kernel modules. "
echo " Please run: sudo reboot                                  "
echo "=========================================================="
