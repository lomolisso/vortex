#!/usr/bin/env bash
# Split runs/<cfg>/logs/run.log into per-stage files under logs/stages/.
# Slices on the "Stage: NN_name" banners pnr.tcl emits.
#
# Usage:
#   ./split_run_log.sh <config>
#   ./split_run_log.sh runs/tcu             # equivalent
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config|run_dir>" >&2
    exit 1
fi

target="$1"
[ -d "$target/logs" ] || target="runs/$1"
[ -d "$target/logs" ] || { echo "No logs/ under $target" >&2; exit 1; }

log="$target/logs/run.log"
[ -f "$log" ] || { echo "Missing $log" >&2; exit 1; }

out="$target/logs/stages"
mkdir -p "$out"
rm -f "$out"/*.log

awk -v outdir="$out" '
    match($0, /Stage:[[:space:]]+([0-9]+_[a-zA-Z_]+)/, m) {
        if (fh) close(fh)
        fh = outdir "/" m[1] ".log"
        print > fh
        next
    }
    /Setup:[[:space:]]+00_config/ {
        if (fh) close(fh)
        fh = outdir "/00_config.log"
    }
    /Loading checkpoint:[[:space:]]+after_/ {
        if (fh) close(fh)
        fh = outdir "/00_resume.log"
    }
    fh { print > fh }
' "$log"

echo "Per-stage logs in $out/:"
ls -1 "$out" 2>/dev/null | sed 's/^/  /'
