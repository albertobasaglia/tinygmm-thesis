#!/usr/bin/env bash
#
# Train the speech (Google Speech Commands v0.02) feature extractor (Stage 1).
#
# The extractor must never see any personalization word, so the held-out set is
# the UNION of the on-device (validation) and test words. We read both straight
# from the class-split files so this script and the sweep stay in sync:
#
#   src/compare/classes/speech/ondevice.txt  -> validation words (--split val)
#   src/compare/classes/speech/test.txt      -> test words       (--split test)
#
# Hyperparameters mirror checkpoints/speech/misc/hparams.yaml (emb=16,
# ch=32-64-128, dropout=0.1, Adam, lr=1e-3).
#
# After training, the best checkpoint lands in logs/speech_extractor/version_N/.
# Promote the chosen one into the committed location, e.g.:
#   cp logs/speech_extractor/version_N/checkpoints/<best>.ckpt checkpoints/speech/best.ckpt
#   cp logs/speech_extractor/version_N/hparams.yaml            checkpoints/speech/misc/
#   cp logs/speech_extractor/version_N/metrics.csv             checkpoints/speech/misc/
#
set -euo pipefail
cd "$(dirname "$0")/.."

CLASSES="src/compare/classes/speech"
HELD_OUT=$(tr '\n' ' ' < "$CLASSES/ondevice.txt"; tr '\n' ' ' < "$CLASSES/test.txt")

echo "[*] Held-out words (ondevice + test): $HELD_OUT"

# shellcheck disable=SC2086  # intentional word-splitting: one word per arg
uv run python -m src.train.speech_extractor \
    --embedding_dim 16 \
    --channels 32 64 128 \
    --dropout 0.1 \
    --lr 1e-3 \
    --epochs 50 \
    --patience 7 \
    --seed 42 \
    --held_out_words $HELD_OUT
