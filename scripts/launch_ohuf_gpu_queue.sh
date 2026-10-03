#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-3}
run_id=${OHUF_RUN_ID:-ohuf-published-dev-20260911}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-"$repo_root/.venv/bin/python"}
hf_home=${HF_HOME:-"$repo_root/data/huggingface"}
output_root="$repo_root/outputs/$run_id"
saved_root="$repo_root/data/saved"

if [[ ! -x "$python_bin" ]]; then
  echo "Python environment is unavailable: $python_bin" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 2
fi

used_mib=$(
  nvidia-smi --id="$gpu" \
    --query-compute-apps=used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk '{total += $1} END {print total + 0}'
)
if (( used_mib >= 256 )); then
  echo "Physical GPU $gpu is occupied (${used_mib} MiB); queue not started" >&2
  exit 3
fi

baseline_name="${run_id}-baseline"
k4_name="${run_id}-k4"
k8_name="${run_id}-k8"
for name in "$baseline_name" "$k4_name" "$k8_name"; do
  if [[ -e "$saved_root/$name" ]]; then
    echo "Refusing to overwrite existing results: $saved_root/$name" >&2
    exit 4
  fi
done
if [[ -e "$output_root" ]]; then
  echo "Refusing to reuse output directory: $output_root" >&2
  exit 4
fi

mkdir -p "$output_root"
exec > >(tee -a "$output_root/queue.log") 2>&1

export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HOME="$hf_home"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_root/.cache/matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$MPLCONFIGDIR"

run_inference() {
  local config_name=$1
  local save_name=$2
  shift 2
  echo "$(date --iso-8601=seconds) starting $config_name as $save_name"
  "$python_bin" inference.py \
    -cn="$config_name" \
    inferencer.device=cuda \
    inferencer.save_path="$save_name" \
    inferencer.example_indices='[]' \
    dataloader.batch_size=4 \
    "hydra.run.dir=$output_root/hydra/$save_name" \
    "$@"
  echo "$(date --iso-8601=seconds) completed $config_name as $save_name"
}

cd "$repo_root"
run_inference ohuf_published_baseline_confirmation_eval "$baseline_name"
run_inference ohuf_published_confirmation_eval "$k4_name" \
  model=ohuf_published_k4
run_inference ohuf_published_confirmation_eval "$k8_name"

for candidate in "$k4_name" "$k8_name"; do
  "$python_bin" scripts/summarize_ohuf_comparison.py \
    --baseline "$saved_root/$baseline_name/confirmation/per_image.csv" \
    --candidate "$saved_root/$candidate/confirmation/per_image.csv" \
    --output "$output_root/comparison-$candidate" \
    --expected-masks 16 \
    --expected-scenes-per-mask 4
done

"$python_bin" - "$output_root" "$saved_root" \
  "$baseline_name" "$k4_name" "$k8_name" "$gpu" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output_root, saved_root = map(Path, sys.argv[1:3])
baseline, k4, k8, gpu = sys.argv[3:]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

repo_root = Path.cwd()
manifest = {
    "status": "complete",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "physical_gpu": int(gpu),
    "cuda_visible_devices": gpu,
    "hostname": platform.node(),
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "protocol": "published-backbone development smoke; official test untouched",
    "runs": {},
}
for name in (baseline, k4, k8):
    summary = saved_root / name / "confirmation" / "summary.json"
    manifest["runs"][name] = {
        "summary": str(summary.resolve()),
        "summary_sha256": sha256(summary),
        "metrics": json.loads(summary.read_text()),
    }
(output_root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, allow_nan=False) + "\n"
)
PY

echo "$(date --iso-8601=seconds) OHUF queue complete: $output_root"
