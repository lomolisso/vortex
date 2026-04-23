#!/usr/bin/env bash
#=======================================================================
# open_checkpoint.sh — Open an Innovus GUI session at a saved checkpoint
#
# Usage:
#   ./open_checkpoint.sh <config> <stage>
#   ./open_checkpoint.sh <config> --list
#
# Arguments:
#   config  — design configuration, e.g. 1c1n4w4t
#   stage   — checkpoint stage name, e.g. 03_floorplan
#             (prefix "after_" is optional and will be stripped automatically)
#
# Examples:
#   # View the SRAM macro pre-placement (what the floorplan stage produced)
#   ./open_checkpoint.sh 1c1n4w4t 03_floorplan
#
#   # Inspect the design right after standard-cell placement
#   ./open_checkpoint.sh 1c1n4w4t 05_placement
#
#   # List all available checkpoints for a config
#   ./open_checkpoint.sh 1c1n4w4t --list
#
# Available configs:
#   1c1n4w4t  1c2n4w4t  2c2n4w4t_l2  4c2n4w4t_l2l3  4c4n8w4t_l2l3
#
# Checkpoint stages (one saved after each completed stage):
#   01_init_design       02_power_connect    03_floorplan
#   04_power_plan        05_placement        06_pre_cts_opt
#   07_cts               08_post_cts_opt     09_routing
#   10_post_route_opt    11_reports          12_outputs
#=======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYN_DIR="$(cd "$SCRIPT_DIR/../../syn/synopsys" && pwd)"
LIBS_DIR="$SYN_DIR/libs"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF

Usage: $0 <config> <stage>
       $0 <config> --list

  config  — e.g. 1c1n4w4t
  stage   — stage number (3) or full name (03_floorplan)

Available configs:
  1c1n4w4t  1c2n4w4t  2c2n4w4t_l2  4c2n4w4t_l2l3  4c4n8w4t_l2l3

Checkpoint stages:
  01_init_design       02_power_connect    03_floorplan
  04_power_plan        05_placement        06_pre_cts_opt
  07_cts               08_post_cts_opt     09_routing
  10_post_route_opt    11_reports          12_outputs

Examples:
  $0 1c1n4w4t 3       view floorplan (SRAM macro pre-placement)
  $0 1c1n4w4t 5       view placement
  $0 1c1n4w4t 03_floorplan   full name also works

EOF
    exit 1
}

CONFIG="${1:-}"
STAGE="${2:-}"

if [[ -z "$CONFIG" || -z "$STAGE" ]]; then
    usage
fi

RUN_DIR="$SCRIPT_DIR/runs/$CONFIG"
CKPT_DIR="$RUN_DIR/checkpoints"

# ---------------------------------------------------------------------------
# --list: print available checkpoints and exit
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "--list" ]]; then
    if [[ ! -d "$CKPT_DIR" ]]; then
        echo ""
        echo "No checkpoints found for config '$CONFIG'."
        echo "Expected directory: $CKPT_DIR"
        echo ""
        exit 1
    fi
    echo ""
    echo "Available checkpoints for config '$CONFIG':"
    for f in "$CKPT_DIR/after_"*; do
        # Skip the .dat binary directories; only list the header files
        [[ "$f" == *.dat ]] && continue
        [[ -f "$f" ]] && echo "  $(basename "$f" | sed 's/^after_//')"
    done
    echo ""
    exit 0
fi

# ---------------------------------------------------------------------------
# Normalize: accept bare number (3 or 03), full name, or "after_" prefix
# ---------------------------------------------------------------------------
STAGE="${STAGE#after_}"
if [[ "$STAGE" =~ ^[0-9]+$ ]]; then
    NUM=$(printf "%02d" "$((10#$STAGE))")
    MATCH=$(ls "$CKPT_DIR/" 2>/dev/null \
            | grep "^after_${NUM}_" | grep -v '\.dat$' \
            | head -1 | sed 's/^after_//' || true)
    if [[ -z "$MATCH" ]]; then
        echo ""
        echo "ERROR: No checkpoint found for stage number '$STAGE' in config '$CONFIG'."
        echo "       Run: $0 $CONFIG --list"
        echo ""
        exit 1
    fi
    STAGE="$MATCH"
fi

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ ! -d "$RUN_DIR" ]]; then
    echo ""
    echo "ERROR: Run directory not found: $RUN_DIR"
    echo "       Run 'make $CONFIG' in hw/pnr/cadence first."
    echo ""
    exit 1
fi

CKPT_FILE="$CKPT_DIR/after_${STAGE}"
if [[ ! -f "$CKPT_FILE" ]]; then
    echo ""
    echo "ERROR: Checkpoint not found: $CKPT_FILE"
    echo ""
    echo "Available checkpoints for config '$CONFIG':"
    for f in "$CKPT_DIR/after_"*; do
        [[ "$f" == *.dat ]] && continue
        [[ -f "$f" ]] && echo "  $(basename "$f" | sed 's/^after_//')"
    done
    echo ""
    exit 1
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if [[ -z "${DISPLAY:-}" ]]; then
    echo ""
    echo "ERROR: DISPLAY is not set — Innovus cannot open its GUI."
    echo ""
    echo "You must run this script from a graphical session with X11 available."
    echo "On the EWS cluster, the two options are:"
    echo ""
    echo "  Option A — FastX desktop (recommended):"
    echo "    1. Open https://fastx.ews.illinois.edu in your browser"
    echo "    2. Start a new GNOME/XFCE session"
    echo "    3. Open a terminal inside that desktop"
    echo "    4. Run this script from there (DISPLAY will be :1 or similar)"
    echo ""
    echo "  Option B — SSH with X11 forwarding:"
    echo "    ssh -Y <netid>@$(hostname)"
    echo "    Then run this script in that SSH session."
    echo ""
    exit 1
fi

echo ""
echo "================================================================"
echo " Opening Innovus at checkpoint: after_${STAGE}"
echo " Config  : $CONFIG"
echo " Run dir : $RUN_DIR"
echo " DISPLAY : $DISPLAY"
echo "================================================================"
echo ""

mkdir -p "$RUN_DIR/logs"

cd "$RUN_DIR"
exec env \
    CONFIG="$CONFIG" \
    LIBS_DIR="$LIBS_DIR" \
    SYN_DIR="$SYN_DIR" \
    CHECKPOINT="$STAGE" \
    innovus \
        -files "$SCRIPT_DIR/scripts/open_checkpoint.tcl" \
        -log   "logs/open_checkpoint_${STAGE}.log" \
        -overwrite
