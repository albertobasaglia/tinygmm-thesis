#!/usr/bin/env bash
#
# Train the HAR (WISDM-2019 watch) feature extractor (Stage 1).
#
# The extractor must never see any personalization subject, so the held-out set
# is the UNION of the on-device (validation) and test subjects. We read both
# straight from the class-split files so this script and the sweep stay in sync:
#
#   src/compare/classes/har/ondevice.txt   -> validation subjects (--split val)
#   src/compare/classes/har/test.txt       -> test subjects       (--split test)
#
# Hyperparameters mirror checkpoints/har/misc/hparams.yaml (emb=16, ch=32-64-128,
# dropout=0.1, adamw, lr=1e-3, wd=1e-4, ReduceLROnPlateau factor=0.5/patience=3).
#
# After training, the best checkpoint lands in logs/har_extractor/version_N/.
# Promote the chosen one into the committed location, e.g.:
#   cp logs/har_extractor/version_N/checkpoints/<best>.ckpt checkpoints/har/best.ckpt
#   cp logs/har_extractor/version_N/hparams.yaml            checkpoints/har/misc/
#   cp logs/har_extractor/version_N/metrics.csv             checkpoints/har/misc/
#
set -euo pipefail
cd "$(dirname "$0")/.."

CLASSES="src/compare/classes/har"
# shellcheck disable=SC2046  # intentional word-splitting: one subject id per arg
HELD_OUT=$(tr '\n' ' ' < "$CLASSES/ondevice.txt"; tr '\n' ' ' < "$CLASSES/test.txt")

echo "[*] Held-out subjects (ondevice + test): $HELD_OUT"

# shellcheck disable=SC2086
uv run python -m src.train.har_extractor \
    --embedding_dim 16 \
    --channels 32 64 128 \
    --dropout 0.1 \
    --optimizer adamw \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --lr_factor 0.5 \
    --lr_patience 3 \
    --lr_min 1e-6 \
    --epochs 50 \
    --patience 7 \
    --seed 42 \
    --held_out_subjects $HELD_OUT
