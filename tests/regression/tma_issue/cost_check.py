#!/usr/bin/env python3
# Per-warp DXA issue cost, run from a configured build directory. The issue cost
# is the header's own instructions plus the DXA unit's accept, so the check is
# structural: with 4 issuing warps no warp issues faster than alone and the SM
# issues no fewer copies per cycle than with one warp. The H100's per-form issue costs are printed as
# H100REF lines for task G, not judged.
# Env: DRIVER (simx, rtlsim), CONFIGS.
import os
import re
import subprocess
import sys

# 14/spec.md §3, §5, §7 and M2: H100 cycles between one warp's successive issues of each form.
H100 = {"bulk": 47, "tensor": 47, "bulk_store": 69, "bulk_reduce": 69, "tensor_store": 131,
        "tensor_reduce": 158, "prefetch_bulk": 9, "prefetch_tensor": 60, "tmap_inval": 7,
        "tmap_prefetch": 3}
LINE = re.compile(r"cost form=(\S+) warp=(\d+) warps=(\d+) t1=\d+ t2=\d+ percopy_x100=(\d+)")


def run(configs, args):
    env = dict(os.environ, CONFIGS=configs)
    subprocess.run(["make", "-s", "-C", "tests/regression/tma_issue", "clean"], env=env,
                   stdout=subprocess.DEVNULL, check=True)
    p = subprocess.run(["./ci/blackbox.sh", "--driver=" + os.environ.get("DRIVER", "simx"),
                        "--app=tma_issue", "--l2cache", "--args=" + args], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    out = {}
    for m in LINE.finditer(p.stdout):
        out.setdefault(m.group(1), []).append(int(m.group(4)))
    if p.returncode != 0 or not out:
        print(p.stdout[-3000:])
        sys.exit("FAIL: tma_issue %s exited %d" % (args, p.returncode))
    return out


def main():
    base = os.environ.get("CONFIGS", "")
    fails = []
    one = run(base, "-k1 -l3")
    four = run(base, "-k1 -l3 -w4")
    for f in sorted(one):
        per1 = one[f][0]
        pers = four.get(f, [])
        if len(pers) != 4:
            fails.append("%s: %d warps reported with -w4" % (f, len(pers)))
            continue
        # 1 % for the rounding of the per-copy quotient
        fast = [p for p in pers if p * 100 < per1 * 99]
        rate1, rate4 = 1.0 / per1, sum(1.0 / p for p in pers)
        ok = not fast and rate4 * 100 >= rate1 * 99
        print("%-16s per_copy=%7.2f 4w=[%s] copies/100cyc 1w=%.3f 4w=%.3f %s" % (
            f, per1 / 100, " ".join("%.2f" % (p / 100) for p in pers), rate1 * 1e4, rate4 * 1e4,
            "ok" if ok else "FAIL"))
        print("H100REF metric=tma_issue_%s_cycles h100=%d vortex=%.2f" % (f, H100[f], per1 / 100))
        if fast:
            fails.append("%s: a warp of 4 issues faster than one warp alone (%.2f < %.2f)"
                         % (f, min(fast) / 100, per1 / 100))
        if rate4 * 100 < rate1 * 99:
            fails.append("%s: 4 warps issue fewer copies per cycle than one warp" % f)
    for s in fails:
        print("FAIL: " + s)
    print("PASSED!" if not fails else "FAILED! (%d)" % len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
