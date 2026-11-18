#!/usr/bin/env python3
# mbarrier try_wait suspension (12/spec.md §4.2/§4.4), run from a configured build directory.
#   limit:  a pending try_wait of a full warp returns false after MBAR_SUSPEND_CYCLES plus the op and the follow-up
#           test_wait (2 x the op + 48); a partial warp (1 lane, half the lanes) is not suspended: it returns
#           false once its ops answer (at most 2 x the op + 48).
#   hint:   suspendTimeHint t ns adds a sleep of t to 1.3 t (+ the op + 64), full warp and lane 0 alike.
#   wake:   a phase flip releases the suspended waiters the next cycle: the arrive-to-waiter time of a try_wait
#           loop is at most the arrive, the waiter's try_wait and the loop (2 x the op + 48); a hinted try_wait whose
#           phase completes during its sleep (the arrive lands after the suspension limit) returns by the next
#           test_wait poll (one more op: 3 x the op + 48), not at the end of the hint.
# H100 numbers are printed as H100REF lines, not judged: partial warp 40, wake-up 110 end to end, test_wait
# spin 137, hinted wake 117, hint 0 7877 (full) and 190 (lane 0).
# Usage: mbar_suspend_check.py <check>; env DRIVER (simx, rtlsim), CONFIGS.
import os
import re
import subprocess
import sys

KNOBS = re.compile(r"knobs suspend=(\d+) fe=(\d+) threads=(\d+)")
RESULT = re.compile(r"result mode=\d+ lanes=\d+ hint=-?\d+ rounds=\d+ median=(\d+)")


def run(args, extra=""):
    env = dict(os.environ)
    if extra:
        env["CONFIGS"] = (env.get("CONFIGS", "") + " " + extra).strip()
    p = subprocess.run(["./ci/blackbox.sh", "--driver=" + os.environ.get("DRIVER", "simx"), "--app=mbar_suspend",
                        "--args=" + args],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
    k, m = KNOBS.search(p.stdout), RESULT.search(p.stdout)
    if p.returncode != 0 or "PASSED!" not in p.stdout or not k or not m:
        print(p.stdout[-3000:])
        sys.exit("FAIL: mbar_suspend %s exited %d" % (args, p.returncode))
    k = dict(zip(["suspend", "fe", "threads"], (int(x) for x in k.groups())))
    if k["fe"] == 0:
        sys.exit("FAIL: the build has the front-end switch off")
    t = int(m.group(1))
    print("mbar_suspend %s%s: median %d" % (args, (" [" + extra + "]") if extra else "", t))
    return t, k


def h100ref(metric, h100, vortex):
    print("H100REF metric=%s h100=%d vortex=%d" % (metric, h100, vortex))


def within(fails, what, v, lo, hi):
    ok = lo <= v <= hi
    print("%s %s: %d in [%d, %d]" % ("ok  " if ok else "FAIL", what, v, lo, hi))
    if not ok:
        fails.append(what)


def check_limit(fails):
    d, k = run("-m1 -r3")
    full, _ = run("-m0 -r3")
    within(fails, "full warp limit", full, k["suspend"], k["suspend"] + 2 * d + 48)
    for lanes in sorted({1, k["threads"] // 2}):
        part, _ = run("-m0 -r3 -l%d" % lanes)
        within(fails, "partial warp (%d lanes) not suspended" % lanes, part, 0, 2 * d + 48)
        h100ref("mbar_partial_return_%dlanes" % lanes, 40, part)
    h100ref("mbar_full_limit", 7762, full)
    print("record: op %d" % d)


def check_hint(fails):
    d, _ = run("-m1 -r3")
    for lanes, label in ((0, "full warp"), (1, "lane 0")):
        base, _ = run("-m0 -r3 -l%d" % lanes)
        zero, _ = run("-m0 -r3 -l%d -t0" % lanes)
        hinted, _ = run("-m0 -r3 -l%d -t1000" % lanes)
        within(fails, "%s hint 1000 ns sleep" % label, hinted - base, 1000, 1300 + d + 64)
        tag = "full" if lanes == 0 else "lane0"
        h100ref("mbar_hint0_" + tag, 7877 if lanes == 0 else 190, zero)
        h100ref("mbar_hint1000_" + tag, 9221 if lanes == 0 else 948, hinted)


def check_wake(fails):
    d, k = run("-m1 -r3")
    t, _ = run("-m2 -r5")
    spin, _ = run("-m3 -r5")
    within(fails, "arrive-to-waiter time of a suspended try_wait", t, 1, 2 * d + 48)
    hinted, _ = run("-m2 -r5 -t10000 -d%d" % (k["suspend"] + 2 * d + 48 + 1000))
    within(fails, "arrive-to-waiter time of a hinted try_wait woken in its sleep", hinted, 1, 3 * d + 48)
    h100ref("mbar_wake_e2e", 110, t)
    h100ref("mbar_wake_hint10000_e2e", 117, hinted)
    h100ref("mbar_test_wait_spin_wake", 137, spin)


def main():
    checks = {"limit": check_limit, "hint": check_hint, "wake": check_wake}
    if len(sys.argv) < 2 or any(a not in checks for a in sys.argv[1:]):
        sys.exit("usage: mbar_suspend_check.py {%s} ..." % ",".join(checks))
    fails = []
    for a in sys.argv[1:]:
        checks[a](fails)
    if fails:
        sys.exit("FAIL: " + "; ".join(fails))
    print("PASS")


if __name__ == "__main__":
    main()
