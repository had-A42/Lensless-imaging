#!/usr/bin/env bash
# Finite follow-up queue for the already running REFINE-01 pilot on GPU3.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=offline PYTHONUNBUFFERED=1
python_bin=.venv/bin/python

while tmux has-session -t '=psffrefine01' 2>/dev/null; do sleep 20; done
"$python_bin" - <<'PY'
import json,subprocess
from pathlib import Path
p=Path('outputs/psff_refine_01/pair-seed42-r1/summary.json')
s=json.loads(p.read_text())
assert s['status']=='complete' and s['baseline_parity_pass'], 'Predecessor needs audit'
used=int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<256, f'GPU3 is occupied ({used} MiB); queue stopped'
print('REFINE-01 completed; begin bounded oracle diagnostics',flush=True)
PY

timeout 600 "$python_bin" run_tools/oracle_gradient_pilot.py --val-masks 4 --val-scenes 4 \
  --output outputs/psff_oracle_grad_01/smoke-r1
timeout 1800 "$python_bin" run_tools/oracle_gradient_pilot.py \
  --output outputs/psff_oracle_grad_01/pilot-r1

if "$python_bin" - <<'PY'
import json,sys
from pathlib import Path
s=json.loads(Path('outputs/psff_refine_01/pair-seed42-r1/summary.json').read_text())
signal=s['h1_signal'] or s['h2_signal']
print('Continuation control needed:',signal,flush=True)
sys.exit(0 if signal else 10)
PY
then
  timeout 600 "$python_bin" run_tools/residual_refinement_pilot.py --mode continue \
    --steps 2 --batch-size 4 --val-masks 4 --val-scenes 4 --workers 2 \
    --output outputs/psff_refine_01/continue-smoke-r1
  timeout 7200 "$python_bin" run_tools/residual_refinement_pilot.py --mode continue \
    --output outputs/psff_refine_01/continue-seed42-r1
fi
echo 'Bounded hypothesis queue complete.'
