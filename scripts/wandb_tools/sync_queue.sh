#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  cat <<'EOF' >&2
Usage: scripts/wandb_tools/sync_queue.sh RUN_ID [RUN_ID ...]
   or: scripts/wandb_tools/sync_queue.sh --file scripts/wandb_tools/passed_runs.txt

Runs are uploaded one by one. The queue stops at the first error.
SYNC_MODE defaults to auto (metrics-only).
Set DRY_RUN=1 to prepare and inspect every metrics stream without uploading.
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

mode=${SYNC_MODE:-auto}
count=${#run_ids[@]}
for ((index = 0; index < count; index++)); do
  run_id=${run_ids[$index]}
  echo "[$((index + 1))/$count] Syncing $run_id in $mode mode"
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    scripts/wandb_tools/sync_run.sh "$run_id" "$mode" --dry-run
  else
    scripts/wandb_tools/sync_run.sh "$run_id" "$mode"
  fi
done

echo "Queue completed: $count run(s)"
