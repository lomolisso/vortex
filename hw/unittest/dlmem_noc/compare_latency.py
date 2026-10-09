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

"""Compare the RTL and SimX DLMEM network latency.csv of one cluster shape.

Both files come from the fixed latency schedule described in
hw/unittest/dlmem_noc/main.cpp, run by the RTL testbench (--mode=latency) and
by the SimX unit test. Packets are matched on (src, dst, seq); the models agree
when every packet exists on both sides with the same inject and deliver cycle.

usage: compare_latency.py RTL.csv SIMX.csv [--socket-size=S] [--max-report=N]
exit status: 0 identical, 1 any difference, 2 unreadable input.
"""

import argparse
import collections
import csv
import sys

FIELDS = ["src", "dst", "seq", "inject_cycle", "deliver_cycle"]


def load(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError("%s: header %s, expected %s" % (path, reader.fieldnames, FIELDS))
        rows = {}
        for line, row in enumerate(reader, 2):
            key = (int(row["src"]), int(row["dst"]), int(row["seq"]))
            if key in rows:
                raise ValueError("%s:%d: duplicate packet src=%d dst=%d seq=%d" % ((path, line) + key))
            rows[key] = (int(row["inject_cycle"]), int(row["deliver_cycle"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rtl")
    ap.add_argument("simx")
    ap.add_argument("--socket-size", type=int, default=0,
                    help="classify latency deltas as local/remote (same socket or not)")
    ap.add_argument("--max-report", type=int, default=20)
    args = ap.parse_args()

    try:
        rtl = load(args.rtl)
        simx = load(args.simx)
    except (OSError, ValueError) as e:
        print("compare_latency: %s" % e)
        return 2

    def kind(key):
        if args.socket_size <= 0:
            return "all"
        src, dst, _ = key
        return "local" if src // args.socket_size == dst // args.socket_size else "remote"

    only_rtl = sorted(set(rtl) - set(simx))
    only_simx = sorted(set(simx) - set(rtl))
    diffs = []
    # Histogram of SimX-minus-RTL latency per class: a constant offset points at
    # a per-hop timing difference, a spread at arbitration or buffering.
    deltas = collections.defaultdict(collections.Counter)
    for key in sorted(set(rtl) & set(simx)):
        (ri, rd), (si, sd) = rtl[key], simx[key]
        if (ri, rd) != (si, sd):
            diffs.append((key, rtl[key], simx[key]))
        deltas[kind(key)][(sd - si) - (rd - ri)] += 1

    print("compare_latency: %d RTL packets, %d SimX packets, %d matched exactly"
          % (len(rtl), len(simx), len(set(rtl) & set(simx)) - len(diffs)))
    for key in only_rtl[:args.max_report]:
        print("  only in RTL : src=%d dst=%d seq=%d" % key)
    for key in only_simx[:args.max_report]:
        print("  only in SimX: src=%d dst=%d seq=%d" % key)
    for key, (ri, rd), (si, sd) in diffs[:args.max_report]:
        print("  src=%d dst=%d seq=%d: RTL inject %d deliver %d (latency %d), SimX inject %d deliver %d (latency %d)"
              % (key + (ri, rd, rd - ri, si, sd, sd - si)))
    if len(diffs) > args.max_report:
        print("  ... %d more differing packets" % (len(diffs) - args.max_report))
    for k in sorted(deltas):
        hist = ", ".join("%+d: %d" % (d, n) for d, n in sorted(deltas[k].items()))
        print("  latency delta SimX-RTL (%s): %s" % (k, hist))

    if only_rtl or only_simx or diffs:
        print("DIFFERENT")
        return 1
    print("IDENTICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
