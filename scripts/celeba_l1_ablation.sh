#!/usr/bin/env bash
set -euo pipefail

# SEED=52 bash scripts/celeba_l1_ablation.sh 0 0.1
gpu=${1:?Usage: bash scripts/celeba_l1_ablation.sh GPU WEIGHT [--dry-run]}
weight=${2:?Provide 0.1 for the control or 1.0 for stronger full-resolution L1}
dry_run=${3:-}
seed=${SEED:-42}
if [[ $weight != 0.1 && $weight != 1.0 ]]; then
  echo "Expected L1 weight 0.1 or 1.0" >&2
  exit 2
fi
if [[ -n $dry_run && $dry_run != --dry-run ]]; then
  echo "Expected --dry-run or no third argument" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
python_bin=${PYTHON_BIN:-.venv/bin/python}
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline PYTHONUNBUFFERED=1 MPLCONFIGDIR=.cache/matplotlib
export HF_HOME="${HF_HOME:-$PWD/data/huggingface}"
export XRESTORMER_CHECKPOINT="${XRESTORMER_CHECKPOINT:-$PWD/model_weights/xrestormer/net_g_latest.pth}"
parent_dir="$PWD/saved/celeba32-sp8-xrestormer-gopro-pretrained-finite-100-10000step-seed${seed}"
parent_checkpoint="$parent_dir/checkpoint-epoch4.pth"
if [[ ! -f $parent_checkpoint ]]; then parent_checkpoint="$parent_dir/model_best.pth"; fi
export CELEBA_XREST_PARENT="${CELEBA_XREST_PARENT:-$parent_checkpoint}"

run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ $dry_run != --dry-run ]]; then "$@"; fi
}

name="celeba32-xrest-l1-${weight}-2000step-seed${seed}"
run "$python_bin" train.py -cn=celeba_xrestormer_l1_ablation \
  trainer.device=cuda "trainer.seed=$seed" "celeba_loss.full_l1_weight=$weight" "writer.run_name=$name"
checkpoint="$PWD/saved/$name/checkpoint-epoch2.pth"
if [[ ! -f $checkpoint ]]; then checkpoint="$PWD/saved/$name/model_best.pth"; fi
run "$python_bin" inference.py -cn=celeba_xrestormer_grid_eval inferencer.device=cuda \
  "inferencer.seed=$seed" "inferencer.from_pretrained=$checkpoint" "writer.run_name=${name}-eval-unseen"
echo "$(date '+%F %T') Completed $name"
