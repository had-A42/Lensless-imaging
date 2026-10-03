#!/usr/bin/env bash
set -euo pipefail

# Foreground worker, one experiment at a time on the selected GPU.
# bash scripts/celeba_queue.sh 0 --dry-run
# SEEDS='42 52 62' bash scripts/celeba_queue.sh 1
# CELEBA_LANE=3 RUN_PREFLIGHT=0 bash scripts/celeba_queue.sh 0
gpu=${1:?Usage: bash scripts/celeba_queue.sh GPU_INDEX [--dry-run]}
dry_run=${2:-}
if [[ -n $dry_run && $dry_run != --dry-run ]]; then
  echo "Expected --dry-run or no second argument" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
python_bin=${PYTHON_BIN:-.venv/bin/python}
read -r -a seeds <<< "${SEEDS:-42}"
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=offline
export HF_HOME="${HF_HOME:-$PWD/data/huggingface}"
lane=${CELEBA_LANE:-$gpu}

case "$lane" in
  0) config=celeba_single_mask; condition=finite_1; label=single-seen; size=32; superpixel=8; masks=1 ;;
  1) config=celeba_psf_free; condition=finite_100; label=finite-100; size=32; superpixel=8; masks=100 ;;
  2) config=celeba_psf_free; condition=infinite; label=infinite; size=32; superpixel=8; masks=0 ;;
  3) config=celeba_64; condition=finite_100; label=finite-100; size=64; superpixel=4; masks=100 ;;
  *) echo "Prepared lanes use GPU 0, 1, 2; GPU 3 is optional 64x64 sensitivity." >&2; exit 2 ;;
esac

run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ $dry_run != --dry-run ]]; then
    "$@"
  fi
}

if [[ $lane -eq 0 && ${RUN_PREFLIGHT:-1} == 1 && ${EVALUATE_ONLY:-0} != 1 ]]; then
  run "$python_bin" inference.py -cn=celeba_mean_eval inferencer.device=cuda
  run "$python_bin" train.py -cn=celeba_overfit trainer.device=cuda
fi

for seed in "${seeds[@]}"; do
  name="celeba${size}-sp${superpixel}-drunet-${label}-10000step-seed${seed}"
  if [[ ${EVALUATE_ONLY:-0} != 1 ]]; then
    run "$python_bin" train.py -cn="$config" "scale_condition=$condition" \
      "trainer.seed=$seed" trainer.device=cuda "writer.run_name=$name"
  fi
  checkpoint="$PWD/saved/$name/checkpoint-epoch4.pth"
  # The trainer avoids duplicating the periodic file when this epoch is best.
  # After a completed 4-epoch run, its final state is then model_best.pth.
  if [[ ! -f $checkpoint ]]; then
    checkpoint="$PWD/saved/$name/model_best.pth"
  fi
  run "$python_bin" inference.py -cn=celeba_eval inferencer.device=cuda \
    "inferencer.seed=$seed" "inferencer.from_pretrained=$checkpoint" \
    "scene_size=$size" "superpixel_size=$superpixel" \
    "writer.run_name=${name}-eval-unseen${EVAL_SUFFIX:-}"
  if [[ $masks -gt 0 ]]; then
    seen_count=32
    if [[ $masks -eq 1 ]]; then seen_count=1; fi
    run "$python_bin" inference.py -cn=celeba_eval inferencer.device=cuda \
      "inferencer.seed=$seed" "inferencer.from_pretrained=$checkpoint" \
      "scene_size=$size" "superpixel_size=$superpixel" evaluation_name=seen-masks \
      +dataloader_builder.train_mode=finite "+dataloader_builder.finite_mask_count=$masks" \
      +dataloader_builder.validation_mask_split=train \
      "dataloader_builder.validation_mask_count=$seen_count" \
      "writer.run_name=${name}-eval-seen${EVAL_SUFFIX:-}"
  fi
done
