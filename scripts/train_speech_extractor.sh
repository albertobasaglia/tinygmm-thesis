set -euo pipefail
cd "$(dirname "$0")/.."

CLASSES="src/compare/classes/speech"
HELD_OUT=$(tr '\n' ' ' < "$CLASSES/ondevice.txt"; tr '\n' ' ' < "$CLASSES/test.txt")

echo "[*] Held-out words (ondevice + test): $HELD_OUT"

uv run python -m src.train.speech_extractor \
    --embedding_dim 16 \
    --channels 32 64 128 \
    --dropout 0.1 \
    --lr 1e-3 \
    --epochs 50 \
    --patience 7 \
    --seed 42 \
    --held_out_words $HELD_OUT
