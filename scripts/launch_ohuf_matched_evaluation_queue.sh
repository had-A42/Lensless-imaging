#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-2}
seeds=${SEEDS:-"42 52 62"}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-"$repo_root/.venv/bin/python"}
hf_home=${HF_HOME:-"$repo_root/data/huggingface"}
output_root="$repo_root/outputs/ohuf_matched_large_v1_evaluation"
training_root="$repo_root/outputs/ohuf_matched_large_v1_training"

if [[ -e "$output_root" ]]; then
  echo "Refusing to reuse evaluation output: $output_root" >&2
  exit 4
fi
mkdir -p "$output_root"
exec > >(tee -a "$output_root/queue.log") 2>&1
export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HOME="$hf_home"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_root/.cache/matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$MPLCONFIGDIR"
cd "$repo_root"

wait_for_training_seed() {
  local seed=$1
  local free_marker="$training_root/ohuf_matched_psf_free_large_v1-gpu0/seed${seed}_checkpoint.sha256"
  local aware_marker="$training_root/ohuf_matched_psf_aware_large_v1-gpu1/seed${seed}_checkpoint.sha256"
  while [[ ! -f "$free_marker" || ! -f "$aware_marker" ]]; do
    if ! tmux has-session -t ohuf_matched_free_gpu0 2>/dev/null && [[ ! -f "$free_marker" ]]; then
      echo "PSF-free training lane stopped before seed $seed" >&2
      exit 5
    fi
    if ! tmux has-session -t ohuf_matched_aware_gpu1 2>/dev/null && [[ ! -f "$aware_marker" ]]; then
      echo "PSF-aware training lane stopped before seed $seed" >&2
      exit 5
    fi
    sleep 30
  done
}

wait_for_gpu() {
  while true; do
    local used_mib
    used_mib=$(
      nvidia-smi --id="$gpu" --query-compute-apps=used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk '{total += $1} END {print total + 0}'
    )
    if (( used_mib < 256 )); then
      return
    fi
    echo "$(date --iso-8601=seconds) GPU $gpu occupied (${used_mib} MiB); waiting"
    sleep 30
  done
}

for seed in $seeds; do
  wait_for_training_seed "$seed"
  wait_for_gpu
  free_dir="$repo_root/saved/ohuf-matched-large-v1-psf-free-seed${seed}"
  aware_dir="$repo_root/saved/ohuf-matched-large-v1-psf-aware-seed${seed}"
  free_full="$free_dir/checkpoint-epoch10.pth"
  aware_full="$aware_dir/checkpoint-epoch10.pth"
  free_safe="$free_dir/model-state-epoch10.pth"
  aware_safe="$aware_dir/model-state-epoch10.pth"
  if [[ ! -f "$free_safe" ]]; then
    "$python_bin" scripts/export_safe_model_checkpoint.py \
      --input "$free_full" --output "$free_safe"
  fi
  if [[ ! -f "$aware_safe" ]]; then
    "$python_bin" scripts/export_safe_model_checkpoint.py \
      --input "$aware_full" --output "$aware_safe"
  fi

  seed_root="$output_root/seed${seed}"
  echo "$(date --iso-8601=seconds) calibration seed=$seed on GPU $gpu"
  "$python_bin" scripts/ohuf_matched_large_evaluate.py \
    --phase calibration \
    --output "$seed_root/calibration" \
    --seed "$seed" \
    --proposal-checkpoint "$free_safe" \
    --aware-checkpoint "$aware_safe"
  echo "$(date --iso-8601=seconds) confirmation seed=$seed on GPU $gpu"
  "$python_bin" scripts/ohuf_matched_large_evaluate.py \
    --phase confirmation \
    --output "$seed_root/confirmation" \
    --seed "$seed" \
    --proposal-checkpoint "$free_safe" \
    --aware-checkpoint "$aware_safe" \
    --selection "$seed_root/calibration/selection.json"
  echo "$(date --iso-8601=seconds) evaluation complete seed=$seed"
done

"$python_bin" scripts/summarize_ohuf_matched_large.py \
  --root "$output_root" \
  --output "$output_root/aggregate"
echo "$(date --iso-8601=seconds) matched OHUF evaluation queue complete"
