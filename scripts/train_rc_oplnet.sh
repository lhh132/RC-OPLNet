#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
: "${DATA_ROOT:?Set DATA_ROOT to the external GDRBench directory}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
exec "${PYTHON:-python}" main.py --algorithm RC-OPLNet --root "$DATA_ROOT" \
  --dg_mode DG --split-profile strict \
  --source-domains APTOS DEEPDR FGADR IDRID --target-domains RLDR \
  --epochs 100 --batch-size 32 --num-workers 4 --val_ep 1 \
  --output "${OUTPUT:-rc_oplnet_rldr_seed42}" "$@"
