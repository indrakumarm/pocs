#!/bin/bash
# -------------------------------------------------------------------------
# AWS EC2 Startup Script for vLLM on g4dn.xlarge (Amazon Linux 2 / AL2023)
# -------------------------------------------------------------------------

# 1. Locate and Format the Ephemeral Instance Store
EPHEMERAL_DRIVE=$(lsblk -dno NAME,MODEL | grep -i "Amazon EC2 NVMe Instance Storage" | awk '{print "/dev/"$1}')

if [ -b "$EPHEMERAL_DRIVE" ]; then
    # Format as ext4 if no existing file system is found
    if ! blkid "$EPHEMERAL_DRIVE" | grep -q "type="; then
        mkfs.ext4 -F "$EPHEMERAL_DRIVE"
    fi

    # 2. Create Mount Point and Mount the Drive
    mkdir -p /mnt/instance_store
    mount "$EPHEMERAL_DRIVE" /mnt/instance_store
    
    # Pass permissions to the default 'ec2-user'
    chown -R ec2-user:ec2-user /mnt/instance_store
    chmod 775 /mnt/instance_store
fi

# 3. Prepare Cache Directories
mkdir -p /mnt/instance_store/hf_cache
chown -R ec2-user:ec2-user /mnt/instance_store/hf_cache

# 4. Inject Environment Variables for 'ec2-user'
PROFILE="/home/ec2-user/.bash_profile"
if ! grep -q "HF_HOME" "$PROFILE"; then
    echo "" >> "$PROFILE"
    echo "# Custom vLLM / Hugging Face configurations" >> "$PROFILE"
    echo "export HF_HOME=/mnt/instance_store/hf_cache" >> "$PROFILE"
    echo "export VLLM_CACHE=/mnt/instance_store/hf_cache/vllm" >> "$PROFILE"
fi

# 5. Fix ownership permissions for the profile file
chown ec2-user:ec2-user "$PROFILE"

