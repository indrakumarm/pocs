#!/bin/bash
# First-boot setup for the Deep Learning Base GPU AMI (driver, Docker, nvidia-container-toolkit preinstalled).
set -euxo pipefail

IDLE_MINUTES=__IDLE_MINUTES__

apt-get -o DPkg::Lock::Timeout=600 update
DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y python3-venv rsync
usermod -aG docker ubuntu
install -d -o ubuntu -g ubuntu /home/ubuntu/.cache/huggingface /home/ubuntu/inference-lab

if [ "$IDLE_MINUTES" -gt 0 ]; then
  echo "$IDLE_MINUTES" > /etc/lab-idle-minutes
  cat > /usr/local/bin/lab-idle-check <<'EOF'
#!/bin/bash
# Stops the instance after N minutes of idle GPU with no SSH connections (run every 5 min by cron).
limit=$(cat /etc/lab-idle-minutes)
state=/var/lib/lab-idle-count
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -nr | head -1)
ssh_conns=$(ss -Htn state established '( sport = :22 )' | wc -l)
if [ "${util:-0}" -lt 5 ] && [ "$ssh_conns" -eq 0 ]; then
  idle=$(( $(cat "$state" 2>/dev/null || echo 0) + 5 ))
else
  idle=0
fi
echo "$idle" > "$state"
if [ "$idle" -ge "$limit" ]; then
  echo 0 > "$state"
  logger "lab-idle-check: idle ${idle}m, shutting down"
  shutdown -h now
fi
EOF
  chmod +x /usr/local/bin/lab-idle-check
  echo "*/5 * * * * root /usr/local/bin/lab-idle-check" > /etc/cron.d/lab-idle
fi

touch /var/lib/lab-ready
