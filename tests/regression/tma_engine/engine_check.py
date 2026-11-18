#!/usr/bin/env python3
# TMA engine structure checks, run from a configured build directory. The issue side has no
# throttle, credit or hold: the DXA unit takes at most one op a cycle, and the TMA queue, the
# bulk groups, the workers' store fences and the SM's egress port bind. H100 measurements are
# printed as H100REF lines for task G, never judged.
#   intake: 32 issuing warps, TMA_QUEUE copies of 16 B: at most one copy a cycle SM-wide, and the
#           SM issues at least one warp's rate; 4 warps: no warp issues faster than alone.
#   drain:  32 warps, four queues' worth of 16-B copies: the backlog after the last issue is at
#           most one queue, so the last copy is done within TMA_QUEUE copies at the run's own rate
#           of the one-copy-per-warp reference.
#   queue:  TMA_QUEUE copies of 2 KiB (none completes during the issue) issue as fast as 16-B
#           copies: the queue holds them without an issue stall.
#   store:  31 groups of one STORE_BYTES bulk store: never more than SM_EGRESS_BYTES a cycle leave
#           the SM, and when the issue loop outpaces the egress the stores drain at the port's
#           width; wait_group K >= pending returns as fast as with nothing pending.
#   reduce: 4 KiB more of bulk reduce takes longer, at no more than the egress width.
#   read:   wait_group.read 0 returns before wait_group 0 alone does for the same store
#           (the source read is released before the write's round trip ends).
#   complete: a 16-B bulk store's issue -> wait_group 0 (printed against the H100's 108 cycles).
#   cold:   the same to lines nothing touched before: no faster than the L2-resident case.
# Usage: engine_check.py <check> ...; env DRIVER (simx, rtlsim), CONFIGS.
import os
import re
import subprocess
import sys

# 14/spec.md §5, M2, M6, M7 and 07 M6 (task G reference data).
H100_INTAKE = 6
H100_STORE_PENDING = {4096: 8, 16384: 4}
H100_REDUCE_BPC = 25.6
H100_DONE_AFTER_READ = 6
H100_STORE16_DONE = 108
H100_COLD_SLACK = 5

KNOBS = re.compile(r"knobs queue=(\d+) st_fences=(\d+) groups=(\d+) egress=(\d+)")


def run(args, extra=""):
    env = dict(os.environ)
    if extra:
        env["CONFIGS"] = (env.get("CONFIGS", "") + " " + extra).strip()
    p = subprocess.run(["./ci/blackbox.sh", "--driver=" + os.environ.get("DRIVER", "simx"), "--app=tma_engine",
                        "--warps=32", "--l2cache", "--args=" + args],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
    m = KNOBS.search(p.stdout)
    if p.returncode != 0 or "PASSED!" not in p.stdout or not m:
        print(p.stdout[-3000:])
        sys.exit("FAIL: tma_engine %s exited %d" % (args, p.returncode))
    k = dict(zip(["queue", "st_fences", "groups", "egress"], (int(x) for x in m.groups())))
    if k["queue"] == 0:
        sys.exit("FAIL: the build has no TMA engine knobs (the front-end switch is off)")
    return p.stdout, k


def ref(metric, h100, vortex):
    print("H100REF metric=%s h100=%s vortex=%s" % (metric, h100, vortex))


def k_queue_per_warp():
    return int(os.environ.get("QUEUE_PER_WARP", "4"))


def check_intake(fails):
    out, k = run("-m0 -w32 -p%d -b16" % max(1, k_queue_per_warp()))
    per = int(re.search(r"intake warps=32 .*percopy_x100=(\d+)", out).group(1)) / 100
    out, k = run("-m0 -w1 -p8 -b16")
    alone = int(re.search(r"intake warp=0 .*pergap_x100=(\d+)", out).group(1)) / 100
    print("intake: 32 warps %.2f cycles per copy SM-wide, one warp alone %.2f" % (per, alone))
    ref("tma_intake_cycles_per_copy", H100_INTAKE, "%.2f" % per)
    if per < 0.99:
        fails.append("intake: %.2f cycles per copy, the DXA unit takes one op a cycle" % per)
    if per > alone * 1.01:
        fails.append("intake: 32 warps issue a copy per %.2f cycles, slower than one warp's %.2f" % (per, alone))
    out, k = run("-m0 -w4 -p8 -b16")
    for w, gap in re.findall(r"intake warp=(\d+) .*pergap_x100=(\d+)", out):
        g = int(gap) / 100
        if g < alone * 0.99:
            fails.append("4 warps: warp %s issues every %.2f, faster than %.2f alone" % (w, g, alone))


def check_drain(fails):
    out, k = run("-m0 -w32 -p1 -b16")
    one = int(re.search(r"intake warps=32 .*lag=(\d+)", out).group(1))
    out, k = run("-m0 -w32 -p%d -b16" % (4 * k["queue"] // 32))
    m = re.search(r"intake warps=32 .*percopy_x100=(\d+) lag=(\d+)", out)
    per, lag = int(m.group(1)) / 100, int(m.group(2))
    bound = one + k["queue"] * per
    print("drain: %.2f cycles per copy, last done %d after the last issue (bound %.0f, one copy per warp %d)"
          % (per, lag, bound, one))
    ref("tma_drain_cycles_per_copy", H100_INTAKE, "%.2f" % per)
    if lag > bound:
        fails.append("drain: the last copy done %d cycles after the last issue, more than a queue's worth (%.0f)"
                     % (lag, bound))


def issue_span(size):
    out, k = run("-m0 -w32 -p%d -b%d" % (max(1, k_queue_per_warp()), size))
    m = re.search(r"intake warps=32 copies=(\d+) .*span=(\d+)", out)
    return 32 * int(m.group(1)), int(m.group(2)), k


def check_queue(fails):
    copies, span, k = issue_span(2048)
    _, span16, _ = issue_span(16)
    print("queue: %d copies of 2 KiB issued in %d cycles, of 16 B in %d" % (copies, span, span16))
    if copies < k["queue"]:
        fails.append("queue: only %d copies issued, TMA_QUEUE is %d" % (copies, k["queue"]))
    if span > span16 * 1.15 + 16:
        fails.append("queue: 2-KiB copies took %d cycles to issue, 16-B copies %d: the issue stalled" % (span, span16))


def store_run(groups, size):
    out, k = run("-m1 -g%d -p1 -b%d -k31" % (groups, size))
    commits = [int(x) for x in re.findall(r"store group=\d+ commit=(\d+)", out)]
    t = {n: int(re.search(r"%s=(\d+)" % n, out).group(1)) for n in ("t_waitk", "t_wait0", "t_waitb")}
    return k, commits, t


def check_store(fails, size):
    k, ref_commits, _ = store_run(16, 128)
    gaps = [b - a for a, b in zip(ref_commits, ref_commits[1:])]
    period = sorted(gaps)[len(gaps) // 2]
    groups = min(31, k["groups"] - 1)
    k, commits, t = store_run(groups, size)
    drain = size / k["egress"]
    # the window: the first issue to the return of the wait_group 0 after the last commit
    end = commits[-1] + t["t_waitk"] + t["t_wait0"]
    rate = groups * size / end
    gaps = [b - a for a, b in zip(commits, commits[1:])]
    j = next((i for i, g in enumerate(gaps) if g > period * 1.15), None)
    backlog = "none" if j is None else "%.1f" % ((j + 1) - (commits[j] - commits[0]) / drain)
    print("store %d B: loop period %d, egress drain %.0f per store, %d groups in %d cycles = %.2f B/cycle "
          "(port %d), stores queued at the first stall %s; wait_group 31 %d (none pending %d)"
          % (size, period, drain, groups, end, rate, k["egress"], backlog, t["t_waitk"], t["t_waitb"]))
    if size in H100_STORE_PENDING:
        ref("tma_store_pending_%dB" % size, H100_STORE_PENDING[size], backlog)
    if rate > k["egress"]:
        fails.append("store %d B: %.2f B/cycle left the SM, above its %d-B egress port" % (size, rate, k["egress"]))
    if period * 1.15 < drain and rate < 0.85 * k["egress"]:
        fails.append("store %d B: the egress binds (period %d < %.0f) but drained %.2f B/cycle of %d"
                     % (size, period, drain, rate, k["egress"]))
    if t["t_waitk"] > t["t_waitb"] + 16:
        fails.append("store: wait_group 31 with %d groups pending took %d cycles, %d with none"
                     % (groups, t["t_waitk"], t["t_waitb"]))


def check_reduce(fails):
    t, k = {}, None
    for b in (4096, 8192):
        out, k = run("-m2 -b%d" % b)
        t[b] = int(re.search(r"reduce bytes=%d t=(\d+)" % b, out).group(1))
    delta = t[8192] - t[4096]
    print("reduce: t(4 KiB) %d, t(8 KiB) %d, delta %d" % (t[4096], t[8192], delta))
    ref("tma_reduce_bytes_per_cycle", H100_REDUCE_BPC, "%.2f" % (4096.0 / delta if delta > 0 else 0))
    if delta <= 0:
        fails.append("reduce: 4 KiB more took no longer (%d)" % delta)
    elif 4096.0 / delta > k["egress"]:
        fails.append("reduce: %.2f B/cycle, above the %d-B egress port" % (4096.0 / delta, k["egress"]))


def store_times(args):
    out, _ = run(args)
    m = re.search(r"read bytes=\d+ t_read=(\d+) t_done=(\d+) t_full=(\d+)", out)
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def store_done(args):
    return store_times(args)[:2]


def check_read(fails):
    t_read, after, full = store_times("-m3 -b4096")
    print("read: 4-KiB store issue -> wait_group.read 0 %d, then wait_group 0 %d later; issue -> wait_group 0 "
          "alone %d" % (t_read, after, full))
    ref("bulk_done_after_read_cycles", H100_DONE_AFTER_READ, after)
    if t_read <= 0:
        fails.append("read: wait_group.read 0 returned in %d cycles" % t_read)
    if full <= t_read:
        fails.append("read: wait_group.read 0 took %d cycles, wait_group 0 alone %d: the read release waited "
                     "for the write" % (t_read, full))


def check_complete(fails):
    t_read, after = store_done("-m3 -b16")
    print("complete: 16-B store issue -> wait_group 0 %d cycles, %d after wait_group.read 0" % (t_read + after, after))
    ref("bulk_store16_issue_to_wait0_cycles", H100_STORE16_DONE, t_read + after)


def check_cold(fails):
    warm = sum(store_done("-m3 -b16"))
    cold = sum(store_done("-m3 -b16 -c"))
    print("cold: 16-B store issue -> wait_group 0, L2-resident %d, cold destination %d" % (warm, cold))
    ref("bulk_store16_cold_minus_warm_cycles", "<=%d" % H100_COLD_SLACK, cold - warm)
    if cold < warm:
        fails.append("cold: a cold destination (%d) completed before an L2-resident one (%d)" % (cold, warm))


def main():
    checks = sys.argv[1:] or ["intake", "drain", "queue", "store", "reduce", "read"]
    fails = []
    for c in checks:
        if c == "intake":
            check_intake(fails)
        elif c == "drain":
            check_drain(fails)
        elif c == "queue":
            check_queue(fails)
        elif c == "store":
            check_store(fails, int(os.environ.get("STORE_BYTES", "4096")))
        elif c == "reduce":
            check_reduce(fails)
        elif c == "read":
            check_read(fails)
        elif c == "complete":
            check_complete(fails)
        elif c == "cold":
            check_cold(fails)
        else:
            sys.exit("unknown check " + c)
    for f in fails:
        print("FAIL: " + f)
    if fails:
        sys.exit(1)
    print("PASSED: " + " ".join(checks))


if __name__ == "__main__":
    main()
