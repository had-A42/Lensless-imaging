#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/wandb_tools/pull_run.sh RUN_ID [REMOTE_WANDB_DIR] [LOCAL_GROUP]

Copies one offline run from A800. Checkpoints are skipped by default.
Set INCLUDE_CHECKPOINTS=1 only when a checkpoint backup is actually needed.

Examples:
  scripts/wandb_tools/pull_run.sh p9bqdl6x
  scripts/wandb_tools/pull_run.sh pl8ljnw9 \
    /home/hadhad/project/Lensless-imaging-research-508a878/wandb operator-prompt
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 2
fi

run_id=$1
remote_wandb=${2:-/home/hadhad/project/Lensless-imaging/wandb}
local_group=${3:-downloaded}
remote_host=${REMOTE_HOST:-a800.sas.yp-c.yandex.net}
local_root=${LOCAL_WANDB_ROOT:-wandb/remote-a800}

if [[ ! $run_id =~ ^[[:alnum:]]{8}$ ]]; then
  echo "RUN_ID must contain exactly eight letters or digits: $run_id" >&2
  exit 2
fi

remote_run=$(ssh -o StrictHostKeyChecking=no "$remote_host" \
  "find '$remote_wandb' -maxdepth 1 -type d -name 'offline-run-*-$run_id' -print -quit")
if [[ -z $remote_run ]]; then
  echo "Run $run_id was not found under $remote_host:$remote_wandb" >&2
  exit 1
fi

run_dir=${remote_run##*/}
destination="$local_root/$local_group/$run_dir"
mkdir -p "$destination"
remote_size=$(ssh -o StrictHostKeyChecking=no "$remote_host" "du -sh '$remote_run'" | awk '{print $1}')

echo "Run:         $run_id"
echo "Remote:      $remote_host:$remote_run"
echo "Remote size: $remote_size"
echo "Local:       $destination"

rsync_args=(-a --partial --progress)
if [[ ${INCLUDE_CHECKPOINTS:-0} == 1 ]]; then
  rsync_args+=(-L)
  echo "Checkpoints: included"
else
  rsync_args+=(--exclude='*.pth' --exclude='*.ckpt')
  echo "Checkpoints: skipped"
fi

rsync "${rsync_args[@]}" \
  "$remote_host:$remote_run/" \
  "$destination/"

stream="$destination/run-$run_id.wandb"
if [[ ! -f $stream ]]; then
  echo "Copied directory does not contain $stream" >&2
  exit 1
fi
echo "Done: $(du -sh "$destination" | awk '{print $1}') locally"
