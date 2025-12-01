#!/bin/bash
# Train TGSFN MVP

set -e

cd "$(dirname "$0")/.."

echo "Training TGSFN MVP..."
python experiments/train_tgsfn_mvp.py \
    --n_neurons 256 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --lambda_homeo 0.1 \
    --lambda_diss 0.01 \
    "$@"
