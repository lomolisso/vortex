#!/usr/bin/env bash
# Open an Innovus GUI session at a saved checkpoint.
# Usage: ./open_checkpoint.sh <config> <stage>     (stage = N or full name)
#        ./open_checkpoint.sh <config> --list
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYN_DIR="$(cd "$SCRIPT_DIR/../../syn/synopsys" && pwd)"
LIBS_DIR="$SYN_DIR/libs"

usage() {
    echo "Usage: $0 <config> <stage|--list>"
    exit 1
}

CONFIG="${1:-}"
STAGE="${2:-}"

if [[ -z "$CONFIG" || -z "$STAGE" ]]; then
    usage
fi

RUN_DIR="$SCRIPT_DIR/runs/$CONFIG"
CKPT_DIR="$RUN_DIR/checkpoints"

list_checkpoints() {
    [[ -d "$CKPT_DIR" ]] || { echo "No checkpoints at $CKPT_DIR"; exit 1; }
    for f in "$CKPT_DIR/after_"*; do
        [[ "$f" == *.dat ]] && continue
        [[ -f "$f" ]] && echo "  $(basename "$f" | sed 's/^after_//')"
    done
}

if [[ "$STAGE" == "--list" ]]; then
    echo "Available checkpoints for $CONFIG:"
    list_checkpoints
    exit 0
fi

# Normalize STAGE: strip "after_", accept bare N or NN.
STAGE="${STAGE#after_}"
if [[ "$STAGE" =~ ^[0-9]+$ ]]; then
    NUM=$(printf "%02d" "$((10#$STAGE))")
    MATCH=$(ls "$CKPT_DIR/" 2>/dev/null | grep "^after_${NUM}_" | grep -v '\.dat$' | head -1 | sed 's/^after_//' || true)
    [[ -z "$MATCH" ]] && { echo "No checkpoint for stage $STAGE in $CONFIG."; exit 1; }
    STAGE="$MATCH"
fi

[[ -d "$RUN_DIR" ]]                 || { echo "Run dir not found: $RUN_DIR"; exit 1; }
[[ -f "$CKPT_DIR/after_${STAGE}" ]] || { echo "Checkpoint not found: $CKPT_DIR/after_${STAGE}"; echo "Available:"; list_checkpoints; exit 1; }

if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is unset — Innovus needs an X11 session (FastX desktop or 'ssh -Y')."
    exit 1
fi

mkdir -p "$RUN_DIR/logs"
echo "=== Opening checkpoint: $CONFIG / after_${STAGE} ==="

cd "$RUN_DIR"
exec env \
    CONFIG="$CONFIG" \
    LIBS_DIR="$LIBS_DIR" \
    SYN_DIR="$SYN_DIR" \
    CHECKPOINT="$STAGE" \
    innovus \
        -files "$SCRIPT_DIR/scripts/utils/open_checkpoint.tcl" \
        -log   "logs/open_checkpoint_${STAGE}.log" \
        -overwrite
