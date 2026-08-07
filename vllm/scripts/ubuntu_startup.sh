#!/bin/bash
# -------------------------------------------------------------------------
# AWS EC2 Startup Script for vLLM on g4dn.xlarge (Ubuntu 22.04 / 24.04 DLAMI)
# -------------------------------------------------------------------------

# 1. Locate and Format the Ephemeral Instance Store
# On g4dn.xlarge, the local NVMe drive is typically /dev/nvme1n1
EPHEMERAL_DRIVE=$(lsblk -dno NAME,MODEL | grep -i "Amazon EC2 NVMe Instance Storage" | awk '{print "/dev/"$1}')

if [ -b "$EPHEMERAL_DRIVE" ]; then
    # Check if it already has a filesystem, if not, format it as ext4
    if ! blkid "$EPHEMERAL_DRIVE" | grep -q "type="; then
        mkfs.ext4 -F "$EPHEMERAL_DRIVE"
    fi

    # 2. Create Mount Point and Mount the Drive
    mkdir -p /mnt/instance_store
    mount "$EPHEMERAL_DRIVE" /mnt/instance_store
    
    # Give the default ubuntu user read/write permissions
    chown -R ubuntu:ubuntu /mnt/instance_store
    chmod 775 /mnt/instance_store
fi

# 3. Prepare Cache Directories
mkdir -p /mnt/instance_store/hf_cache
chown -R ubuntu:ubuntu /mnt/instance_store/hf_cache

# 4. Inject Environment Variables for the 'ubuntu' User
# This ensures that whenever you log in via SSH, HF_HOME is already configured
BASHRC="/home/ubuntu/.bashrc"
if ! grep -q "HF_HOME" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# Custom vLLM / Hugging Face configurations" >> "$BASHRC"
    echo "export HF_HOME=/mnt/instance_store/hf_cache" >> "$BASHRC"
    echo "export VLLM_CACHE=/mnt/instance_store/hf_cache/vllm" >> "$BASHRC"
fi

# 5. Fix permissions for the injected environment variables
chown ubuntu:ubuntu "$BASHRC"

