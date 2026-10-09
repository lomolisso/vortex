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

"""Gang placement parity: the RTL bench's placement.csv against the SimX
model's run of the same deterministic schedule. The models are cycle exact, so
every row must match, commit cycle included."""

import argparse
import csv
import sys

COLUMNS = ["gang_seq", "rank", "core", "slot", "gang_id", "epoch", "commit_cycle"]


def load(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            sys.exit(f"{path}: columns {reader.fieldnames}, expected {COLUMNS}")
        rows = [tuple(int(r[c]) for c in COLUMNS) for r in reader]
    if not rows:
        sys.exit(f"{path}: no placements")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rtl", help="placement CSV of the RTL bench")
    parser.add_argument("simx", help="placement CSV of the SimX unit test")
    args = parser.parse_args()

    rtl = load(args.rtl)
    simx = load(args.simx)
    bad = 0
    for i, (a, b) in enumerate(zip(rtl, simx)):
        if a != b:
            bad += 1
            if bad <= 10:
                diff = ", ".join(f"{c} {x} vs {y}" for c, x, y in zip(COLUMNS, a, b) if x != y)
                print(f"row {i + 1} (gang {a[0]} rank {a[1]}): {diff}")
    if len(rtl) != len(simx):
        bad += 1
        print(f"{len(rtl)} RTL placements, {len(simx)} SimX placements")
    gangs = len({r[0] for r in rtl})
    if bad:
        print(f"FAILED: {bad} differences over {len(rtl)} placements")
        return 1
    print(f"PASSED: {len(rtl)} placements of {gangs} gangs identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
