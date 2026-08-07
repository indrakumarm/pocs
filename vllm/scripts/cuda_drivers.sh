# Update the system package repository
sudo dnf update -y

# Install kernel headers and development tools required by NVIDIA drivers
sudo dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r) gcc make dnf-plugins-core

# Add the official NVIDIA CUDA repository
sudo dnf config-manager --add-repo https://nvidia.com

# Install the NVIDIA driver and the CUDA Toolkit
sudo dnf clean all
sudo dnf install -y cuda-drivers cuda-toolkit

