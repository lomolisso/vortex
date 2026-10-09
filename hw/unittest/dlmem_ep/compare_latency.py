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

"""Compare the RTL and SimX DLMEM endpoint latency CSVs of one cluster shape.

Both files come from the fixed latency schedule of the shared bench
(tests/unittest/dlmem_ep/dlmem_ep_bench.h), run by the RTL testbench
(hw/unittest/dlmem_ep, --mode=latency) and by the SimX unit test. A row is one
remote lane: issue_cycle is the first cycle its LSU request is offered,
done_cycle its LSU response (LD, AMO, LR, SC) or the store ack of its warp (ST).
Rows are matched on (src, seq); the models agree when every lane exists on both
sides with the same dst, op, issue and done cycle.

usage: compare_latency.py RTL.csv SIMX.csv [--max-report=N]
exit status: 0 identical, 1 any difference, 2 unreadable input.
"""

import argparse
import collections
import csv
import sys

FIELDS = ["src", "dst", "op", "seq", "issue_cycle", "done_cycle"]


def load(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError("%s: header %s, expected %s" % (path, reader.fieldnames, FIELDS))
        rows = {}
        for line, row in enumerate(reader, 2):
            key = (int(row["src"]), int(row["seq"]))
            if key in rows:
                raise ValueError("%s:%d: duplicate lane src=%d seq=%d" % ((path, line) + key))
            rows[key] = (int(row["dst"]), row["op"], int(row["issue_cycle"]), int(row["done_cycle"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rtl")
    ap.add_argument("simx")
    ap.add_argument("--max-report", type=int, default=20)
    args = ap.parse_args()

    try:
        rtl = load(args.rtl)
        simx = load(args.simx)
    except (OSError, ValueError) as e:
        print("compare_latency: %s" % e)
        return 2

    only_rtl = sorted(set(rtl) - set(simx))
    only_simx = sorted(set(simx) - set(rtl))
    mismatched = []  # same lane, different dst or op: the two runs saw different schedules
    diffs = []
    # SimX-minus-RTL latency per op: a constant offset points at a fixed stage
    # difference, a spread at arbitration, credits or buffering.
    deltas = collections.defaultdict(collections.Counter)
    for key in sorted(set(rtl) & set(simx)):
        (rdst, rop, ri, rd), (sdst, sop, si, sd) = rtl[key], simx[key]
        if (rdst, rop) != (sdst, sop):
            mismatched.append((key, rtl[key], simx[key]))
            continue
        if (ri, rd) != (si, sd):
            diffs.append((key, rtl[key], simx[key]))
        deltas[rop][(sd - si) - (rd - ri)] += 1

    common = len(set(rtl) & set(simx))
    print("compare_latency: %d RTL lanes, %d SimX lanes, %d matched exactly"
          % (len(rtl), len(simx), common - len(mismatched) - len(diffs)))
    for key in only_rtl[:args.max_report]:
        print("  only in RTL : src=%d seq=%d" % key)
    for key in only_simx[:args.max_report]:
        print("  only in SimX: src=%d seq=%d" % key)
    for key, r, s in mismatched[:args.max_report]:
        print("  src=%d seq=%d: RTL dst %d op %s, SimX dst %d op %s" % (key + r[:2] + s[:2]))
    for key, (dst, op, ri, rd), (_, _, si, sd) in diffs[:args.max_report]:
        print("  src=%d seq=%d dst=%d %s: RTL issue %d done %d (latency %d), SimX issue %d done %d (latency %d)"
              % (key + (dst, op, ri, rd, rd - ri, si, sd, sd - si)))
    if len(diffs) > args.max_report:
        print("  ... %d more differing lanes" % (len(diffs) - args.max_report))
    for op in sorted(deltas):
        hist = ", ".join("%+d: %d" % (d, n) for d, n in sorted(deltas[op].items()))
        print("  latency delta SimX-RTL (%s): %s" % (op, hist))

    if only_rtl or only_simx or mismatched or diffs:
        print("DIFFERENT")
        return 1
    print("IDENTICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
