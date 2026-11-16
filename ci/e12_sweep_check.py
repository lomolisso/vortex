#!/usr/bin/env python3
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

"""Invariants of the H100 SM path at one configuration point of a sweep.

  e12_sweep_check.py --configs "<-D flags>" [--xlen 64] [--log <app log> ...]

Resolves the configuration with ci/gen_config.py and checks the structural
invariants every shape and knob point must keep; then every app log given
must report PASSED! and no MISMATCH (the apps compare against the goldens at
the same point, so a sweep never reuses one shape's numbers for another).
Exits 1 on any failure.
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def resolve(configs, xlen):
    out = subprocess.run([sys.executable, os.path.join(HERE, "gen_config.py"),
                          "--config", os.path.join(ROOT, "VX_config.toml"),
                          "--cflags", "{} -DVX_CFG_XLEN={}".format(configs, xlen)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if out.returncode != 0:
        sys.exit("gen_config failed:\n" + out.stdout)
    knobs = {}
    for flag in out.stdout.split():
        m = re.match(r"-DVX_CFG_([A-Z0-9_]+)(?:=(.*))?$", flag)
        if m:
            v = m.group(2)
            try:
                knobs[m.group(1)] = int(v, 0) if v is not None else 1
            except ValueError:
                knobs[m.group(1)] = v
    return knobs


def invariants(k):
    """(name, holds, detail) of each structural invariant."""
    banks, bank_bytes, dp = k["LMEM_NUM_BANKS"], k["LMEM_BANK_BYTES"], k["SM_DATAPATH_BYTES"]
    line, sector = k["DCACHE_LINE_SIZE"], k["DCACHE_SECTOR_SIZE"]
    lmem = 1 << k["LMEM_LOG_SIZE"]
    return [
        ("banks x bank bytes = datapath", banks * bank_bytes == dp,
         "{} x {} = {} vs {}".format(banks, bank_bytes, banks * bank_bytes, dp)),
        ("one L1 per SM", k["NUM_DCACHES"] == k["SOCKET_SIZE"],
         "{} D-caches, {} SMs per socket".format(k["NUM_DCACHES"], k["SOCKET_SIZE"])),
        ("one SM port per SM", k["L1_MEM_PORTS"] == k["SOCKET_SIZE"],
         "{} L1 memory ports, {} SMs per socket".format(k["L1_MEM_PORTS"], k["SOCKET_SIZE"])),
        ("whole sectors per line", line % sector == 0 and line // sector >= 1,
         "line {} / sector {}".format(line, sector)),
        ("egress beat = sector", k["SM_EGRESS_BEAT"] == sector,
         "beat {} sector {}".format(k["SM_EGRESS_BEAT"], sector)),
        ("L1 line = datapath", line == dp, "line {} datapath {}".format(line, dp)),
        ("smem per CTA <= LMEM <= descriptor reach",
         k["SMEM_PER_CTA_MAX"] <= lmem <= k["LMEM_DESC_REACH_BYTES"],
         "{} <= {} <= {}".format(k["SMEM_PER_CTA_MAX"], lmem, k["LMEM_DESC_REACH_BYTES"])),
        ("write-through L1", k["DCACHE_WRITEBACK"] == 0, "DCACHE_WRITEBACK {}".format(k["DCACHE_WRITEBACK"])),
        ("store window holds whole lines", k["DLMEM_STORE_WINDOW_BYTES"] % line == 0,
         "window {} line {}".format(k["DLMEM_STORE_WINDOW_BYTES"], line)),
    ]


def check_log(path):
    text = open(path, errors="replace").read()
    mism = [l.strip() for l in text.splitlines() if "MISMATCH" in l]
    passed = "PASSED!" in text
    return passed and not mism, mism


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--xlen", type=int, default=64)
    ap.add_argument("--log", action="append", default=[])
    a = ap.parse_args()
    k = resolve(a.configs, a.xlen)
    if not k.get("H100_SM_ENABLED"):
        sys.exit("the point does not enable the H100 SM path: " + a.configs)
    ok = True
    print("point: {}".format(a.configs))
    for name, holds, detail in invariants(k):
        print("  {:<44} {:<5} {}".format(name, "ok" if holds else "FAIL", detail))
        ok &= holds
    for log in a.log:
        good, mism = check_log(log)
        print("  {:<44} {:<5} {}".format(os.path.basename(log), "ok" if good else "FAIL",
                                          "; ".join(mism[:3])))
        ok &= good
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
