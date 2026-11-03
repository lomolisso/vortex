#!/usr/bin/env python3
# Builds h100_atomics.csv.gz, the H100 atomics measurements the amo_ref unit
# test and atomic_fp read by default, from the full semantics CSV of a probe
# run (one measured operation per row: form, space, op, variant, test,
# pairset, pair_idx, tag, half, old_hex, rhs_hex, rhs2_hex, ret_hex, mem_hex,
# word_hex, guard_ok). Kept, in every form x space x variant group:
#   - every constructed edge row (tag edge);
#   - every special-value row (tag special) of the scalar and packed forms, and
#     of the vN.* vector forms those with test % 3 == 0 (lane k of test t holds
#     pair t + 682k, so the lanes still cover every residue);
#   - the random rows with test % 64 == 0 (vector forms: % 192).
# Rows are sorted by operation and test, then space and variant, so the same
# operands measured in different spaces sit together and compress well.
#
#   gen_h100_golden.py <semantics.csv[.gz]> [-o h100_atomics.csv.gz]

import argparse
import csv
import gzip
import io
import sys

COLS = ["form", "space", "op", "variant", "test", "tag", "half",
        "old_hex", "rhs_hex", "rhs2_hex", "ret_hex", "mem_hex", "guard_ok"]
SPACES = ["global", "generic_global", "shared_cta", "generic_smem",
          "shared_cluster", "generic_dsmem"]


def keep(r):
    vec = r["form"].startswith("v")
    t = int(r["test"])
    if r["tag"] == "edge":
        return True
    if r["tag"] == "special":
        return not vec or t % 3 == 0
    return t % (192 if vec else 64) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="h100_atomics.csv.gz")
    a = ap.parse_args()

    opener = gzip.open if a.csv.endswith(".gz") else open
    rows, total = [], 0
    with opener(a.csv, "rt", newline="") as f:
        for r in csv.DictReader(f):
            total += 1
            if keep(r):
                rows.append([r[c] for c in COLS])

    rank = {s: i for i, s in enumerate(SPACES)}
    rows.sort(key=lambda r: (r[0], r[2], int(r[4]), rank.get(r[1], len(SPACES)), r[1], r[3]))

    text = io.StringIO()
    w = csv.writer(text, lineterminator="\n")
    w.writerow(COLS)
    w.writerows(rows)
    with open(a.out, "wb") as out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0, compresslevel=9) as gz:
            gz.write(text.getvalue().encode())
        size = out.tell()
    print(f"{a.out}: {len(rows)} of {total} rows, {size} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
