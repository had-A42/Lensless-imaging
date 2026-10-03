#!/usr/bin/env bash
set -euo pipefail

# Run explicitly from the isolated project root. Does not stop existing sessions.
package="${1:-outputs/scene_scaling_20260907}"
python_bin="${SCENE_SCALING_PYTHON:-.venv/bin/python}"
session_prefix="${SCENE_SCALING_SESSION_PREFIX:-scene_scaling_20260907}"
gpus=()
while IFS= read -r gpu; do
  gpus+=("$gpu")
done < <("$python_bin" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["queue"]))' "$package/plan.json")
test "${#gpus[@]}" -gt 0

for gpu in "${gpus[@]}"; do
  session="${session_prefix}_gpu${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Session already exists: $session" >&2
    exit 1
  fi
done

# Refuse occupied GPUs and incomplete baseline inventory before creating any session.
"$python_bin" - "$package" <<'PY'
import sys
from scripts.scene_scaling.common import verify_package
from scripts.scene_scaling.run import assert_gpu_idle
plan = verify_package(sys.argv[1])
for run in plan['runs']:
    if run['control']['archive']['status'] != 'available':
        raise SystemExit('Missing final control: ' + run['control_id'])
from pathlib import Path
if not Path(plan['gopro']['path']).is_file():
    raise SystemExit('Missing GoPro initialisation weights')
for gpu in map(int, plan['queue']):
    assert_gpu_idle(gpu)
PY

mkdir -p "$package/logs"
for gpu in "${gpus[@]}"; do
  printf -v command '%q ' "$python_bin" -m scripts.scene_scaling.run worker --package "$package" --gpu "$gpu"
  printf -v logfile '%q' "$package/logs/gpu${gpu}.log"
  tmux new-session -d -s "${session_prefix}_gpu${gpu}" -c "$PWD" "$command > $logfile 2>&1"
done
echo "Started SS01 workers on GPU ${gpus[*]}. Logs: $package/logs"
