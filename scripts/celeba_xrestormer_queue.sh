#!/usr/bin/env bash
set -euo pipefail

# Each worker uses one GPU. An optional tmux predecessor is awaited first:
# bash scripts/celeba_xrestormer_queue.sh 0 --dry-run
# WAIT_FOR=celebaf1 bash scripts/celeba_xrestormer_queue.sh 1
gpu=${1:?Usage: bash scripts/celeba_xrestormer_queue.sh GPU_INDEX [--dry-run]}
dry_run=${2:-}
if [[ -n $dry_run && $dry_run != --dry-run ]]; then
  echo "Expected --dry-run or no second argument" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
python_bin=${PYTHON_BIN:-.venv/bin/python}
read -r -a seeds <<< "${SEEDS:-42 52 62}"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline PYTHONUNBUFFERED=1 MPLCONFIGDIR=.cache/matplotlib
export HF_HOME="${HF_HOME:-$PWD/data/huggingface}"
export XRESTORMER_CHECKPOINT="${XRESTORMER_CHECKPOINT:-$PWD/model_weights/xrestormer/net_g_latest.pth}"

case "$gpu" in
  0) config=celeba_xrestormer_single_mask; condition=finite_1; label=single-seen; masks=1 ;;
  1) config=celeba_xrestormer; condition=finite_100; label=finite-100; masks=100 ;;
  2) config=celeba_xrestormer; condition=infinite; label=infinite; masks=0 ;;
  *) echo "Prepared workers use GPUs 0, 1, 2" >&2; exit 2 ;;
esac

run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ $dry_run != --dry-run ]]; then "$@"; fi
}

if [[ -n ${WAIT_FOR:-} && $dry_run != --dry-run ]]; then
  echo "Waiting for tmux session $WAIT_FOR"
  while tmux has-session -t "=$WAIT_FOR" 2>/dev/null; do sleep 20; done
fi

train_and_evaluate() {
  seed=$1
  size=$2
  superpixel=$3
  name="celeba${size}-sp${superpixel}-xrestormer-gopro-pretrained-${label}-10000step-seed${seed}"
  echo "$(date '+%F %T') Starting $name"
  run "$python_bin" train.py -cn="$config" "scale_condition=$condition" \
    "scene_size=$size" "superpixel_size=$superpixel" \
    "trainer.seed=$seed" trainer.device=cuda "writer.run_name=$name"
  checkpoint="$PWD/saved/$name/checkpoint-epoch4.pth"
  # The final state is saved only as model_best when epoch 4 improves validation.
  if [[ ! -f $checkpoint ]]; then checkpoint="$PWD/saved/$name/model_best.pth"; fi
  run "$python_bin" inference.py -cn=celeba_xrestormer_eval \
    inferencer.device=cuda "inferencer.seed=$seed" \
    "inferencer.from_pretrained=$checkpoint" \
    "scene_size=$size" "superpixel_size=$superpixel" \
    "writer.run_name=${name}-eval-unseen"
  if [[ $masks -gt 0 ]]; then
    seen_count=32
    if [[ $masks -eq 1 ]]; then seen_count=1; fi
    run "$python_bin" inference.py -cn=celeba_xrestormer_eval \
      inferencer.device=cuda "inferencer.seed=$seed" \
      "inferencer.from_pretrained=$checkpoint" \
      "scene_size=$size" "superpixel_size=$superpixel" evaluation_name=seen-masks \
      +dataloader_builder.train_mode=finite "+dataloader_builder.finite_mask_count=$masks" \
      +dataloader_builder.validation_mask_split=train \
      "dataloader_builder.validation_mask_count=$seen_count" \
      "writer.run_name=${name}-eval-seen"
  fi
  echo "$(date '+%F %T') Completed $name"
}

for seed in "${seeds[@]}"; do train_and_evaluate "$seed" 32 8; done
if [[ $gpu -eq 1 ]]; then train_and_evaluate 42 64 4; fi
echo "$(date '+%F %T') GPU${gpu} X-Restormer queue complete"
