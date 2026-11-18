#!/usr/bin/env python3
"""Invariants of the unified L1 / shared-memory array over configurations.

For each --configs string the knobs are expanded through ci/gen_config.py and
checked: the array's geometry, the carve-out list against the shared-memory
maximum, the smallest L1 against the lines in flight, the tags against the
largest L1, the tag-side macro shapes and the valid bits per line. With
--capacity, measured all-hit capacities (carveout-capacity runs) are compared
with the L1 left over at their configuration.

  e12b_sweep_check.py --configs "-DVX_CFG_H100_SM_ENABLE" --configs "... -DVX_CFG_SM_DATAPATH_BYTES=64" \
      [--capacity "<configs index>:<c KiB>:<bytes>" ...]
"""
import argparse
import os
import re
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def knobs(configs, xlen):
    out = subprocess.check_output(
        [sys.executable, os.path.join(ROOT, "ci", "gen_config.py"),
         "--config=" + os.path.join(ROOT, "VX_config.toml"),
         "--cflags=%s -DVX_CFG_XLEN=%d" % (configs, xlen)],
        universal_newlines=True)
    k = {}
    for tok in shlex.split(out):
        m = re.match(r"-D(VX_CFG_\w+)(?:=(.*))?$", tok)
        if not m:
            continue
        v = m.group(2)
        if v is None:
            k[m.group(1)] = 1
            continue
        try:
            k[m.group(1)] = int(v, 0)
        except ValueError:
            k[m.group(1)] = v
    return k


def clog2(x):
    return max(0, (x - 1).bit_length())


def flog2(x):
    return x.bit_length() - 1


def check(k):
    """Returns a list of (name, ok, detail)."""
    g = lambda n: k["VX_CFG_" + n]
    r = []
    sram, row = g("SM_SRAM_BYTES"), g("SM_DATAPATH_BYTES")
    banks, bank_bytes = g("LMEM_NUM_BANKS"), g("LMEM_BANK_BYTES")
    line, resv = g("DCACHE_LINE_SIZE"), g("L1_RESERVE_BYTES")
    rows = sram // row
    r.append(("array = banks x bank bytes x rows",
              sram % row == 0 and banks * bank_bytes == row and banks * bank_bytes * rows == sram,
              "%d = %d x %d x %d" % (sram, banks, bank_bytes, rows)))
    cfgs = [g("SMEM_CONFIG_KB_%d" % i) for i in range(9)]
    r.append(("KB_8 x 1024 = SMEM_PER_SM_MAX", cfgs[8] * 1024 == g("SMEM_PER_SM_MAX"),
              "%d vs %d" % (cfgs[8] * 1024, g("SMEM_PER_SM_MAX"))))
    small = (sram - resv - cfgs[8] * 1024) // line
    r.append(("smallest L1 > SM_MAX_READ_LINES", small > g("SM_MAX_READ_LINES"),
              "%d lines vs %d" % (small, g("SM_MAX_READ_LINES"))))
    max_lines = g("L1_MAX_LINES")
    r.append(("L1_MAX_LINES = L1 at the smallest config", max_lines == (sram - resv - cfgs[0] * 1024) // line,
              "%d" % max_lines))
    sets, ways, tbanks = g("L1_TAG_SETS"), g("L1_TAG_WAYS"), g("L1_TAG_BANKS")
    r.append(("tags cover the largest L1", sets * ways >= max_lines, "%d x %d vs %d" % (sets, ways, max_lines)))
    r.append(("tag banks divide the sets", sets % tbanks == 0, "%d / %d" % (sets, tbanks)))
    valid_bits = line // g("L1_VALID_BYTES")
    r.append(("valid bits per line = line / L1_VALID_BYTES", line % g("L1_VALID_BYTES") == 0,
              "%d bits" % valid_bits))
    maxb = g("SRAM_MACRO_MAX_BITS")
    tag_w = g("MEM_ADDR_WIDTH") - clog2(line) - clog2(sets)
    way_bits = 1 + tag_w + clog2(max_lines)
    tag_wpm = 1 << flog2(max(1, min(ways, maxb // way_bits)))
    val_wpm = 1 << flog2(max(1, min(ways, maxb // valid_bits)))
    depth = sets // tbanks
    r.append(("tag macro <= SRAM_MACRO_MAX_BITS wide", tag_wpm * way_bits <= maxb,
              "%d x %d (%d ways of %d bits)" % (tag_wpm * way_bits, depth, tag_wpm, way_bits)))
    r.append(("valid macro <= SRAM_MACRO_MAX_BITS wide", val_wpm * valid_bits <= maxb,
              "%d x %d (%d ways of %d bits)" % (val_wpm * valid_bits, depth, val_wpm, valid_bits)))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", action="append", required=True)
    ap.add_argument("--xlen", type=int, default=64)
    ap.add_argument("--capacity", action="append", default=[],
                    help="<configs index>:<carve-out KiB>:<measured all-hit bytes>")
    a = ap.parse_args()
    fails = 0
    ks = []
    for i, c in enumerate(a.configs):
        k = knobs(c, a.xlen)
        ks.append(k)
        print("[%d] %s" % (i, c))
        for name, ok, detail in check(k):
            print("  %s %s: %s" % ("ok  " if ok else "FAIL", name, detail))
            fails += not ok
    for cap in a.capacity:
        i, c_kb, got = (int(x) for x in cap.split(":"))
        k = ks[i]
        want = k["VX_CFG_SM_SRAM_BYTES"] - k["VX_CFG_L1_RESERVE_BYTES"] - c_kb * 1024
        ok = got == want
        print("  %s [%d] capacity at %d KiB: %d bytes (L1 lines %d), want %d (H100: %d)" % (
            "ok  " if ok else "FAIL", i, c_kb, got, got // k["VX_CFG_DCACHE_LINE_SIZE"], want,
            (248 - c_kb) * 1024))
        fails += not ok
    print("e12b_sweep_check: %s (%d failures)" % ("FAILED" if fails else "PASSED", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
