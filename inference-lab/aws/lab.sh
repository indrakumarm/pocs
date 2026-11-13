#!/usr/bin/env bash
# Manage named GPU instances. Usage: aws/lab.sh {up|sync|ssh|tunnel|pull|status|start|stop|down} [name]
set -euo pipefail

cd "$(dirname "$0")/.."
ENV_FILE=aws/lab.env
STATE_FILE=aws/.state

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE (cp aws/lab.env.example $ENV_FILE)" >&2; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
NAME=${2:-${NAME:-}}
if [[ ! "$NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "Instance name must use letters, digits, hyphens or underscores and start with a letter/digit" >&2
  exit 1
fi
for v in AWS_PROFILE AWS_REGION INSTANCE_TYPE MARKET ROOT_GB AMI_SSM KEY_NAME SSH_KEY SUBNET_ID SECURITY_GROUP_IDS NAME; do
  [ -n "${!v:-}" ] || { echo "$v is not set in $ENV_FILE" >&2; exit 1; }
done
INSTANCE_PROFILE=${INSTANCE_PROFILE:-}
USE_PRIVATE_IP=${USE_PRIVATE_IP:-0}
IDLE_MINUTES=${IDLE_MINUTES:-30}
SPOT_FALLBACK=${SPOT_FALLBACK:-1}
VLLM_PORT=${VLLM_PORT:-8000}
GRAFANA_PORT=${GRAFANA_PORT:-3000}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-9090}
KNOWN_HOSTS="aws/.known_hosts/$AWS_PROFILE/$AWS_REGION/$NAME"
RESULTS_DIR="results/aws/$NAME"
userdata=
err=
state_tmp=
locked=0

cleanup() {
  [ -z "$userdata" ] || rm -f "$userdata"
  [ -z "$err" ] || rm -f "$err"
  [ -z "$state_tmp" ] || rm -f "$state_tmp"
  [ "$locked" = 0 ] || rmdir aws/.state.lock
}
trap cleanup EXIT

ec2() { aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 "$@"; }
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN_HOSTS" -o ConnectTimeout=10)

# State is TSV: profile, region, name, instance ID. Preserve all other entries.
lock_state() {
  if ! mkdir aws/.state.lock 2>/dev/null; then
    echo "Another AWS lifecycle/state operation is running (aws/.state.lock). Retry when it finishes." >&2
    exit 1
  fi
  locked=1
}

unlock_state() {
  rmdir aws/.state.lock
  locked=0
}

write_state() {
  local id=${1:-}
  state_tmp=$(mktemp aws/.state.XXXXXX)
  if [ -s "$STATE_FILE" ]; then
    awk -F '\t' -v profile="$AWS_PROFILE" -v region="$AWS_REGION" -v name="$NAME" \
      '!($1 == profile && $2 == region && $3 == name)' "$STATE_FILE" > "$state_tmp"
  fi
  if [ -n "$id" ]; then
    printf '%s\t%s\t%s\t%s\n' "$AWS_PROFILE" "$AWS_REGION" "$NAME" "$id" >> "$state_tmp"
  fi
  mv "$state_tmp" "$STATE_FILE"
  state_tmp=
}

prepare_state() {
  local old_id old_name
  [ -s "$STATE_FILE" ] || return 0
  if grep -Eq '^i-[0-9a-f]+$' "$STATE_FILE" && [ "$(wc -l < "$STATE_FILE" | tr -d ' ')" = 1 ]; then
    old_id=$(cat "$STATE_FILE")
    # Associate the old ID with its actual AWS Name tag, never the newly requested name.
    old_name=$(ec2 describe-instances --instance-ids "$old_id" \
      --query 'Reservations[0].Instances[0].Tags[?Key==`Name`].Value | [0]' --output text)
    if [[ ! "$old_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || [ "$old_name" = None ]; then
      echo "Cannot migrate $STATE_FILE: existing instance has no valid Name tag" >&2
      exit 1
    fi
    state_tmp=$(mktemp aws/.state.XXXXXX)
    printf '%s\t%s\t%s\t%s\n' "$AWS_PROFILE" "$AWS_REGION" "$old_name" "$old_id" > "$state_tmp"
    mv "$state_tmp" "$STATE_FILE"
    state_tmp=
    echo "Migrated existing instance $old_id under name $old_name"
  fi
  if ! awk -F '\t' '
    NF != 4 || $1 == "" || $2 == "" || $3 !~ /^[a-zA-Z0-9][a-zA-Z0-9_-]*$/ ||
      $4 !~ /^i-[0-9a-f]+$/ { bad=1 }
    { key=$1 FS $2 FS $3; if (seen[key]++) bad=1 }
    END { exit bad ? 1 : 0 }' "$STATE_FILE"; then
    echo "Invalid or duplicate entries in $STATE_FILE; refusing to change AWS instances" >&2
    exit 1
  fi
}

instance_id() {
  local id
  id=$(awk -F '\t' -v profile="$AWS_PROFILE" -v region="$AWS_REGION" -v name="$NAME" \
    '$1 == profile && $2 == region && $3 == name { print $4 }' "$STATE_FILE")
  [ -n "$id" ] || { echo "No instance recorded for $NAME; run: make aws-up INSTANCE=$NAME" >&2; exit 1; }
  printf '%s\n' "$id"
}

discover() {
  local ids
  ids=$(ec2 describe-instances \
    --filters "Name=tag:Name,Values=$NAME" "Name=tag:Project,Values=inference-lab" \
      "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
    --query 'Reservations[].Instances[].InstanceId' --output text)
  if [ -z "$ids" ] || [ "$ids" = None ]; then
    write_state
    return
  fi
  # AWS text output can separate IDs with tabs or newlines.
  read -r -a found_ids <<< "$(printf '%s' "$ids" | tr '\t\n' '  ')"
  if [ "${#found_ids[@]}" -gt 1 ]; then
    echo "Multiple lab instances named $NAME in $AWS_PROFILE/$AWS_REGION; refusing ambiguous selection: $ids" >&2
    exit 1
  fi
  if [ "${#found_ids[@]}" = 1 ]; then
    write_state "${found_ids[0]}"
    echo "Selected $NAME (${found_ids[0]})"
  else
    write_state
  fi
}

# Sets HOST; uses the public IP when present, else the private IP (VPN).
resolve_host() {
  local ips pub priv
  ips=$(ec2 describe-instances --instance-ids "$(instance_id)" \
    --query 'Reservations[0].Instances[0].[PublicIpAddress,PrivateIpAddress]' --output text)
  read -r pub priv <<< "$ips"
  if [ "$USE_PRIVATE_IP" != 1 ] && [ "$pub" != None ] && [ -n "$pub" ]; then
    HOST="ubuntu@$pub"
  elif [ "$priv" != None ] && [ -n "$priv" ]; then
    HOST="ubuntu@$priv"
  else
    echo "Instance has no IP address (stopped?)" >&2; exit 1
  fi
  mkdir -p "$(dirname "$KNOWN_HOSTS")"
  if [ ! -f "$KNOWN_HOSTS" ] && [ -f aws/.known_hosts.legacy ]; then
    cp aws/.known_hosts.legacy "$KNOWN_HOSTS"
  fi
}

remote() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }

wait_ssh() {
  local out
  echo "Waiting for SSH on $HOST ..."
  for _ in $(seq 36); do
    out=$(ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$HOST" true 2>&1) && return 0
    sleep 5
  done
  echo "Cannot SSH to $HOST after 3 min: $out" >&2
  echo "Check VPN, security group allows port 22 from your network, and SSH_KEY matches $KEY_NAME" >&2
  exit 1
}

cmd_up() {
  discover
  if awk -F '\t' -v profile="$AWS_PROFILE" -v region="$AWS_REGION" -v name="$NAME" \
    '$1 == profile && $2 == region && $3 == name { found=1 } END { exit found ? 0 : 1 }' "$STATE_FILE"; then
    local st
    st=$(ec2 describe-instances --instance-ids "$(instance_id)" \
      --query 'Reservations[0].Instances[0].State.Name' --output text)
    case "$st" in
      pending|running) echo "Resuming setup of $NAME ($(instance_id), $st)"; finish_up; return ;;
      stopping|stopped) cmd_start; finish_up; return ;;
      shutting-down)
        ec2 wait instance-terminated --instance-ids "$(instance_id)"
        write_state
        ;;
      *) echo "Instance $(instance_id) is $st; retry aws-up after its transition" >&2; exit 1 ;;
    esac
  fi
  local ami root_dev market_opts=() profile_opts=()
  ami=$(aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ssm get-parameter \
    --name "$AMI_SSM" --query Parameter.Value --output text)
  root_dev=$(ec2 describe-images --image-ids "$ami" --query 'Images[0].RootDeviceName' --output text)
  echo "AMI $ami ($root_dev), $INSTANCE_TYPE, $MARKET"

  userdata=$(mktemp)
  sed "s/__IDLE_MINUTES__/$IDLE_MINUTES/" aws/user-data.sh > "$userdata"

  if [ "$MARKET" = spot ]; then
    # persistent + stop so the instance can be stopped/started like on-demand
    market_opts=(--instance-market-options
      '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}')
  fi
  [ -n "$INSTANCE_PROFILE" ] && profile_opts=(--iam-instance-profile "Name=$INSTANCE_PROFILE")

  local tags="Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=inference-lab}]" id
  err=$(mktemp)
  launch() {
    # shellcheck disable=SC2086
    ec2 run-instances \
      --image-id "$ami" \
      --instance-type "$INSTANCE_TYPE" \
      --key-name "$KEY_NAME" \
      --subnet-id "$SUBNET_ID" \
      --security-group-ids $SECURITY_GROUP_IDS \
      --block-device-mappings "[{\"DeviceName\":\"$root_dev\",\"Ebs\":{\"VolumeSize\":$ROOT_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
      --user-data "file://$userdata" \
      --metadata-options HttpTokens=required \
      --tag-specifications "ResourceType=instance,$tags" "ResourceType=volume,$tags" \
      ${profile_opts[@]+"${profile_opts[@]}"} "$@" \
      --query 'Instances[0].InstanceId' --output text 2>"$err"
  }

  if ! id=$(launch ${market_opts[@]+"${market_opts[@]}"}); then
    cat "$err" >&2
    if [ "$MARKET" = spot ] && [ "$SPOT_FALLBACK" = 1 ] && grep -q InsufficientInstanceCapacity "$err"; then
      echo "No spot capacity for $INSTANCE_TYPE; falling back to on-demand"
      id=$(launch) || { cat "$err" >&2; exit 1; }
    else
      exit 1
    fi
  fi
  if [[ ! "$id" =~ ^i-[0-9a-f]+$ ]]; then
    echo "AWS returned an invalid instance ID: $id" >&2; exit 1
  fi
  write_state "$id"
  echo "Launched $NAME ($id)"
  finish_up
}

finish_up() {
  echo "Waiting for running state ..."
  ec2 wait instance-running --instance-ids "$(instance_id)"
  resolve_host
  wait_ssh
  echo "Waiting for first-boot setup ..."
  for _ in $(seq 120); do
    remote test -f /var/lib/lab-ready && break
    sleep 5
  done
  remote test -f /var/lib/lab-ready || { echo "Setup not finished; see /var/log/cloud-init-output.log" >&2; exit 1; }
  remote 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader &&
          docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L'
  cmd_sync
  echo "Ready: make aws-ssh INSTANCE=$NAME, then: make up"
}

cmd_sync() {
  # '.*' skips all dotfiles/dirs (.git, .venv, .pvenv, aws/.state, ...); excluded paths are never deleted remotely
  rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
    --exclude '.*' --exclude results --exclude '__pycache__' --exclude aws/lab.env \
    ./ "$HOST:inference-lab/"
  if [ -n "${HF_TOKEN:-}" ]; then
    # via stdin so the token never appears in a remote command line
    printf %s "$HF_TOKEN" | remote 'umask 077 && cat > ~/.cache/huggingface/token'
  fi
  remote 'cd inference-lab && make setup'
}

cmd_ssh() { ssh "${SSH_OPTS[@]}" -t "$HOST" 'cd inference-lab && exec bash -l'; }

cmd_tunnel() {
  echo "$NAME: forwarding localhost:$VLLM_PORT (vLLM), $GRAFANA_PORT (Grafana), $PROMETHEUS_PORT (Prometheus); Ctrl-C to stop"
  ssh "${SSH_OPTS[@]}" -o ExitOnForwardFailure=yes -N \
    -L "$VLLM_PORT:localhost:8000" -L "$GRAFANA_PORT:localhost:3000" \
    -L "$PROMETHEUS_PORT:localhost:9090" "$HOST"
}

cmd_pull() {
  mkdir -p "$RESULTS_DIR"
  rsync -az -e "ssh ${SSH_OPTS[*]}" "$HOST:inference-lab/results/" "$RESULTS_DIR/"
}

cmd_status() {
  ec2 describe-instances --instance-ids "$(instance_id)" --output table \
    --query 'Reservations[0].Instances[0].{Id:InstanceId,State:State.Name,Type:InstanceType,Market:InstanceLifecycle,PublicIp:PublicIpAddress,PrivateIp:PrivateIpAddress,Launched:LaunchTime}'
}

cmd_start() {
  local st
  st=$(ec2 describe-instances --instance-ids "$(instance_id)" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)
  case "$st" in
    stopping)
      ec2 wait instance-stopped --instance-ids "$(instance_id)"
      ec2 start-instances --instance-ids "$(instance_id)" >/dev/null ;;
    stopped) ec2 start-instances --instance-ids "$(instance_id)" >/dev/null ;;
    pending|running) echo "$NAME is already $st; waiting for SSH" ;;
    *) echo "$NAME is $st; use make aws-up INSTANCE=$NAME" >&2; exit 1 ;;
  esac
  ec2 wait instance-running --instance-ids "$(instance_id)"
  resolve_host
  wait_ssh
  echo "Started $NAME: $HOST (start vLLM separately with make up on the host)"
}

cmd_stop() {
  ec2 stop-instances --instance-ids "$(instance_id)" >/dev/null
  echo "Stopping $(instance_id) (EBS + model cache kept, still billed for storage)"
}

cmd_down() {
  local id spot
  id=$(instance_id)
  read -r -p "Terminate $NAME ($id) and delete its volume (model cache)? [y/N] " ans
  [ "$ans" = y ] || [ "$ans" = Y ] || { echo "Aborted"; exit 1; }
  spot=$(ec2 describe-instances --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].SpotInstanceRequestId' --output text)
  # cancel first, otherwise a persistent spot request relaunches the instance
  [ "$spot" != None ] && ec2 cancel-spot-instance-requests --spot-instance-request-ids "$spot" >/dev/null
  ec2 terminate-instances --instance-ids "$id" >/dev/null
  ec2 wait instance-terminated --instance-ids "$id"
  write_state
  rm -f "$KNOWN_HOSTS"
  echo "Terminated $NAME ($id); other named instances are unchanged"
}

# Serialize short state updates and lifecycle changes, not interactive SSH/tunnels.
case "${1:-}" in
  up|sync|ssh|tunnel|pull|status|start|stop|down) ;;
  *) echo "Usage: $0 {up|sync|ssh|tunnel|pull|status|start|stop|down} [name]" >&2; exit 1 ;;
esac
lock_state
prepare_state
touch "$STATE_FILE"
# Keep the old SSH data while moving to per-name files, under the state lock.
if [ -f aws/.known_hosts ]; then
  if [ -e aws/.known_hosts.legacy ]; then
    echo "Both legacy known-hosts files exist; resolve aws/.known_hosts and aws/.known_hosts.legacy first" >&2
    exit 1
  fi
  mv aws/.known_hosts aws/.known_hosts.legacy
fi
if [ "$1" != up ] && ! awk -F '\t' -v profile="$AWS_PROFILE" -v region="$AWS_REGION" -v name="$NAME" \
  '$1 == profile && $2 == region && $3 == name { found=1 } END { exit found ? 0 : 1 }' "$STATE_FILE"; then
  discover
fi
case "$1" in
  up|start|stop|down) ;;
  *) unlock_state ;;
esac

case "${1:-}" in
  up) cmd_up ;;
  sync|ssh|tunnel|pull)
    instance_id >/dev/null
    resolve_host
    "cmd_$1" ;;
  status|start|stop|down)
    instance_id >/dev/null
    "cmd_$1" ;;
esac
