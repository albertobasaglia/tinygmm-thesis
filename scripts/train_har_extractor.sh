set -euo pipefail
cd "$(dirname "$0")/.."

CLASSES="src/compare/classes/har"
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
