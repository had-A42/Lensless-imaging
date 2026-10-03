#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  cat <<'EOF' >&2
Usage: scripts/wandb_tools/pull_queue.sh RUN_ID [RUN_ID ...]
   or: scripts/wandb_tools/pull_queue.sh --file scripts/wandb_tools/passed_runs.txt

Runs are copied from A800 one by one without checkpoints. The queue stops at
the first error. REMOTE_WANDB_DIR and LOCAL_GROUP can override the defaults.
EOF
  exit 2
fi

if [[ $1 == --file ]]; then
  if [[ $# -ne 2 ]]; then
    echo "--file needs exactly one path" >&2
    exit 2
  fi
  run_ids=()
  while IFS= read -r run_id; do
    run_ids+=("$run_id")
  done < <(awk 'NF && $1 !~ /^#/ {print $1}' "$2")
else
  run_ids=("$@")
fi

remote_wandb=${REMOTE_WANDB_DIR:-/home/hadhad/project/Lensless-imaging/wandb}
local_group=${LOCAL_GROUP:-downloaded}
count=${#run_ids[@]}
for ((index = 0; index < count; index++)); do
  run_id=${run_ids[$index]}
  echo "[$((index + 1))/$count] Pulling $run_id"
  scripts/wandb_tools/pull_run.sh "$run_id" "$remote_wandb" "$local_group"
done

echo "Queue completed: $count run(s)"
