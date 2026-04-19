#!/bin/bash

# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generates a VCS-style filelist for Synopsys DC targeting NanGate45 / stdcells.
#
# Usage: gen_dc_sources.sh [-D<define>]... -O <output_flist>
#
# Pass config-specific defines (-DNUM_CLUSTERS=1, -DL2_ENABLE, etc.) via -D.
# All mandatory ASIC defines and source/include paths are hardcoded below.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

RTL_DIR="$SCRIPT_DIR/../../rtl"
VORTEX_ROOT="$SCRIPT_DIR/../../.."
GPGPU_ROOT="$SCRIPT_DIR/../../../.."

CVFPU_DIR="$VORTEX_ROOT/third_party/cvfpu/src"
HARDFLOAT_DIR="$VORTEX_ROOT/third_party/hardfloat/source"
LIBS_DIR="$SCRIPT_DIR/libs"

extra_defines=()
output_file="$SCRIPT_DIR/dc_flist.f"

while getopts "D:O:" flag; do
    case "${flag}" in
        D) extra_defines+=("-D${OPTARG}") ;;
        O) output_file="${OPTARG}" ;;
        \?) echo "Usage: $0 [-D<define>]... -O <output_flist>" 1>&2; exit 1 ;;
    esac
done

"$SCRIPT_DIR/../../scripts/gen_sources.sh" \
    -DASIC_SYNTHESIS \
    -DSYNTHESIS \
    -DNDEBUG \
    -DFPU_FPNEW \
    -DEXT_F_ENABLE \
    -DEXT_M_ENABLE \
    -DICACHE_ENABLE \
    -DDCACHE_ENABLE \
    -DLMEM_ENABLE \
    "${extra_defines[@]}" \
    -T Vortex \
    -J "$CVFPU_DIR" \
    -J "$HARDFLOAT_DIR" \
    -J "$HARDFLOAT_DIR/RISCV" \
    -J "$LIBS_DIR" \
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

echo "Generated: $output_file"
