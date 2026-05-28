#!/bin/bash
# Copyright © 2019-2023. Apache 2.0.
# Emit a VCS-style filelist for Synopsys DC.
#
# Usage: gen_dc_sources.sh [-D<define>]... [-T <top>] [-S <sram_dir>]... -O <out>
#   -T <top>             synthesis top (default: Vortex)
#   -S <sram_macro_dir>  one libs/sram/<m>/ dir per macro; appends the
#                        <m>.bb.v blackbox stub (DC links the matching .db)
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RTL_DIR="$SCRIPT_DIR/../../rtl"
VORTEX_ROOT="$SCRIPT_DIR/../../.."
CVFPU_DIR="$VORTEX_ROOT/third_party/cvfpu/src"
HARDFLOAT_DIR="$VORTEX_ROOT/third_party/hardfloat/source"

extra_defines=()
sram_macro_dirs=()
output_file="$SCRIPT_DIR/flists/dc_flist.f"
top_module="Vortex"

while getopts "D:T:O:S:" flag; do
    case "${flag}" in
        D) extra_defines+=("-D${OPTARG}") ;;
        T) top_module="${OPTARG}" ;;
        O) output_file="${OPTARG}" ;;
        S) sram_macro_dirs+=("${OPTARG}") ;;
        \?) echo "Usage: $0 [-D<define>]... [-T <top>] [-S <sram_dir>]... -O <out>" 1>&2; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$output_file")"

"$SCRIPT_DIR/../../scripts/gen_sources.sh" \
    -DASIC_SYNTHESIS -DSYNTHESIS -DNDEBUG \
    -DFPU_FPNEW -DEXT_F_ENABLE -DEXT_M_ENABLE -DEXT_TCU_ENABLE \
    -DICACHE_ENABLE -DDCACHE_ENABLE -DLMEM_ENABLE \
    "${extra_defines[@]}" \
    -T "$top_module" \
    -J "$CVFPU_DIR/common_cells/src" \
    -J "$CVFPU_DIR/fpu_div_sqrt_mvp/hdl" \
    -J "$CVFPU_DIR" \
    -J "$CVFPU_DIR/common_cells/include" \
    -J "$HARDFLOAT_DIR" \
    -J "$HARDFLOAT_DIR/RISCV" \
    -I "$RTL_DIR" \
    -I "$RTL_DIR/interfaces" \
    -I "$RTL_DIR/core" \
    -I "$RTL_DIR/libs" \
    -I "$RTL_DIR/fpu" \
    -I "$RTL_DIR/mem" \
    -I "$RTL_DIR/cache" \
    -I "$RTL_DIR/tcu" \
    -I "$RTL_DIR/tcu/bhf" \
    -O "$output_file"

# Each SRAM macro dir has both <m>.v (behavioural sim; DC chokes on its
# `=== 1'bx` checks) and <m>.bb.v (empty module, DC-friendly). Append the
# blackbox stub directly — gen_sources.sh's `-J` filter would re-include the .v.
for d in "${sram_macro_dirs[@]:+${sram_macro_dirs[@]}}"; do
    macro=$(basename "$d")
    bb_path="$d/$macro.bb.v"
    if [ ! -f "$bb_path" ]; then
        echo "ERROR: blackbox stub not found: $bb_path" 1>&2
        exit 1
    fi
    {
        echo "+incdir+$(realpath "$d")"
        echo "$(realpath "$bb_path")"
    } >> "$output_file"
done

echo "Generated: $output_file"
