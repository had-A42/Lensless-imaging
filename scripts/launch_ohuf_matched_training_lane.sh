#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?Usage: launch_ohuf_matched_training_lane.sh GPU CONFIG_NAME}
config_name=${2:?Usage: launch_ohuf_matched_training_lane.sh GPU CONFIG_NAME}
seeds=${SEEDS:-"42 52 62"}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-"$repo_root/.venv/bin/python"}
hf_home=${HF_HOME:-"$repo_root/data/huggingface"}
lane_name="${config_name}-gpu${gpu}"
output_root="$repo_root/outputs/ohuf_matched_large_v1_training/$lane_name"

case "$config_name" in
  ohuf_matched_psf_free_large_v1) arm=psf-free ;;
  ohuf_matched_psf_aware_large_v1) arm=psf-aware ;;
  *) echo "Unsupported config: $config_name" >&2; exit 2 ;;
esac
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment is unavailable: $python_bin" >&2
  exit 2
fi
used_mib=$(
  nvidia-smi --id="$gpu" --query-compute-apps=used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk '{total += $1} END {print total + 0}'
)
if (( used_mib >= 256 )); then
  echo "Physical GPU $gpu is occupied (${used_mib} MiB); lane not started" >&2
  exit 3
fi
if [[ -e "$output_root" ]]; then
  echo "Refusing to reuse output directory: $output_root" >&2
  exit 4
fi

mkdir -p "$output_root"
exec > >(tee -a "$output_root/lane.log") 2>&1
export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HOME="$hf_home"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_root/.cache/matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$MPLCONFIGDIR"
cd "$repo_root"

for seed in $seeds; do
  run_name="ohuf-matched-large-v1-${arm}-seed${seed}"
  checkpoint="$repo_root/saved/$run_name/checkpoint-epoch10.pth"
  safe_checkpoint="$repo_root/saved/$run_name/model-state-epoch10.pth"
  if [[ -e "$repo_root/saved/$run_name" ]]; then
    echo "Refusing to reuse checkpoint directory: $repo_root/saved/$run_name" >&2
    exit 4
  fi
  echo "$(date --iso-8601=seconds) starting $config_name seed=$seed on GPU $gpu"
  "$python_bin" train.py \
    -cn="$config_name" \
    trainer.seed="$seed" \
    writer.mode=offline \
    writer.run_name="$run_name" \
    "hydra.run.dir=$output_root/seed$seed"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Expected checkpoint is missing: $checkpoint" >&2
    exit 5
  fi
  "$python_bin" scripts/export_safe_model_checkpoint.py \
    --input "$checkpoint" \
    --output "$safe_checkpoint"
  sha256sum "$checkpoint" > "$output_root/seed${seed}_checkpoint.sha256"
  sha256sum "$safe_checkpoint" > "$output_root/seed${seed}_model_state.sha256"
  echo "$(date --iso-8601=seconds) completed $config_name seed=$seed"
done

"$python_bin" - "$output_root" "$gpu" "$config_name" "$arm" $seeds <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output, gpu, config_name, arm, *seeds = sys.argv[1:]
output = Path(output)
manifest = {
    "status": "complete",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "physical_gpu": int(gpu),
    "config": config_name,
    "arm": arm,
    "seeds": [int(seed) for seed in seeds],
    "checkpoint_hash_files": sorted(
        str(path.resolve()) for path in output.glob("seed*_checkpoint.sha256")
    ),
}
(output / "lane_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

echo "$(date --iso-8601=seconds) training lane complete: $output_root"
