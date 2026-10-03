#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/wandb_tools/sync_run.sh RUN_ID|RUN_DIR [auto|metrics|original] [--dry-run]

auto prepares and uploads a metrics-only stream. It does not upload model
weights, media files or W&B artifacts. Use original only after checking that
the complete stream and its artifact files are really needed.
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 2
fi

target=$1
mode=${2:-auto}
dry_run=${3:-}
archive=${LOCAL_WANDB_ROOT:-wandb/remote-a800}
wandb_cli=${WANDB_CLI:-.venv/bin/wandb}
python_bin=${PYTHON_BIN:-.venv/bin/python}
entity=${WANDB_ENTITY:-had-2005-hse-university}
project=${WANDB_PROJECT:-lensless-imaging}

if [[ $dry_run != "" && $dry_run != "--dry-run" ]]; then
  usage
  exit 2
fi
if [[ $mode != auto && $mode != metrics && $mode != original ]]; then
  usage
  exit 2
fi

if [[ -d $target ]]; then
  run_dir=${target%/}
  run_id=${run_dir##*-}
else
  run_id=$target
  matches=()
  while IFS= read -r match; do
    matches+=("$match")
  done < <(
    find "$archive" -type d -name "offline-run-*-$run_id" \
      ! -path '*/all-checkouts/*' | sort
  )
  unique_matches=()
  for match in "${matches[@]}"; do
    segment=${match##*/}
    duplicate=0
    if [[ ${#unique_matches[@]} -gt 0 ]]; then
      for existing in "${unique_matches[@]}"; do
        if [[ ${existing##*/} == "$segment" ]]; then
          duplicate=1
          break
        fi
      done
    fi
    if [[ $duplicate -eq 0 ]]; then
      unique_matches+=("$match")
    fi
  done
  matches=("${unique_matches[@]}")
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "Run $run_id is not present under $archive" >&2
    exit 1
  fi
  if [[ ${#matches[@]} -gt 1 ]]; then
    echo "Found ${#matches[@]} segments for resumed run $run_id."
    for ((index = 0; index < ${#matches[@]}; index++)); do
      echo "[$((index + 1))/${#matches[@]}] ${matches[$index]}"
      "$0" "${matches[$index]}" "$mode" "$dry_run"
    done
    exit 0
  fi
  run_dir=${matches[0]}
fi

original="$run_dir/run-$run_id.wandb"
if [[ ! -f $original ]]; then
  echo "Original stream not found: $original" >&2
  exit 1
fi

if [[ $mode == original ]]; then
  stream=$original
else
  stream="$run_dir/run-$run_id.metrics.wandb"
  if [[ ! -f $stream || $original -nt $stream ]]; then
    "$python_bin" scripts/wandb_tools/prepare_metrics_stream.py "$original" --force
  fi
fi

echo "Run:     $run_id"
echo "Mode:    $mode"
echo "Stream:  $stream"
echo "Size:    $(du -h "$stream" | awk '{print $1}')"
echo "Target:  $entity/$project"

if [[ ( -f $stream.verified || -f $stream.uploaded ) && ${FORCE_SYNC:-0} != 1 ]]; then
  echo "Already uploaded according to local marker; skipping."
  echo "Set FORCE_SYNC=1 to upload it again."
  exit 0
fi

if [[ $dry_run == "--dry-run" ]]; then
  "$wandb_cli" beta sync --dry-run \
    --entity "$entity" \
    --project "$project" \
    "$stream"
  exit 0
fi

http_code=$(curl -sS --netrc-file "$HOME/.netrc" \
  -H 'Content-Type: application/json' \
  --data '{"query":"query { viewer { id } }"}' \
  -o /dev/null -w '%{http_code}' \
  https://api.wandb.ai/graphql)
if [[ $http_code != 200 ]]; then
  echo "W&B API is unavailable: HTTP $http_code" >&2
  exit 1
fi

mkdir -p wandb/sync-logs
segment=${run_dir##*/}
log="wandb/sync-logs/$run_id-$segment-$mode.log"
WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT:-120} \
  "$wandb_cli" beta sync --yes -n 1 --skip-synced \
    --entity "$entity" \
    --project "$project" \
    "$stream" 2>&1 | tee "$log"
touch "$stream.uploaded"

echo "Upload finished. Check the run in W&B, then mark it manually:"
echo "  touch '$stream.verified'"
