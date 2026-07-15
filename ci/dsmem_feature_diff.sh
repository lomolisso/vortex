#!/bin/bash
# dsmem_feature_diff.sh <build_dir>
#
# Proves the cycle-accurate DSMEM features (F1–F3) are actually MODELLED, not
# dead code: run a targeted test at the default config and again with the
# feature's knob disabled, and compare cycles + the feature-specific counter.
# There is no dedicated per-feature counter, so a differential is the only
# rigorous proof.
#   F1 (residency lookup latency): CTA_TABLE_LOOKUP_LATENCY 2 -> 0  (MCYCLE, LOAD_LT fall)
#   F2 (barrier latency):          CLUSTER_BARRIER_LATENCY  4 -> 0  (MCYCLE, STALLED_WARPS fall)
#   F3 (store credit):             DSMEM_STORE_CREDITS 64 -> huge   (expected: NO change — see note)
set -u
BUILD=${1:?usage: dsmem_feature_diff.sh <build_dir>}
TMP=$(mktemp -d)
rc=0

# run <outfile> <app> <cores> <extra-configs> <args>
run() {
  local out=$1 app=$2 cores=$3 cfg=$4 args=$5
  ( cd "$BUILD" && CONFIGS="-DVX_CFG_DSMEM_ENABLE $cfg" timeout 120 \
      ./ci/blackbox.sh --driver=simx --app="$app" --cores="$cores" --l2cache --perf=1 \
      --args="$args" > "$TMP/$out" 2>&1 )
}
pass()   { grep -q '^PASSED' "$TMP/$1" && echo 1 || echo 0; }
field()  { grep -m1 '^COUNTERS' "$TMP/$2" | sed -E "s/.*$1=([0-9]+).*/\1/"; }

# diff_feature <label> <app> <cores> <off-cfg> <args> <counter> <mode:assert|report>
diff_feature() {
  local label=$1 app=$2 cores=$3 offcfg=$4 args=$5 ctr=$6 mode=$7
  run on_$label  "$app" "$cores" ""       "$args"
  run off_$label "$app" "$cores" "$offcfg" "$args"
  local op=$(pass on_$label)  of=$(pass off_$label)
  local oc=$(field mcycle on_$label)  fc=$(field mcycle off_$label)
  local ok=$(field "$ctr" on_$label)  fk=$(field "$ctr" off_$label)
  printf "[%s] on:{pass=%s mcycle=%s %s=%s}  off:{pass=%s mcycle=%s %s=%s}\n" \
         "$label" "$op" "$oc" "$ctr" "$ok" "$of" "$fc" "$ctr" "$fk"
  if [ "$op" != 1 ] || [ "$of" != 1 ]; then
    echo "  FAIL: a config did not PASS (correctness must be invariant to the knob)"; rc=1; return; fi
  if [ "$mode" = assert ]; then
    if [ "${oc:-0}" -gt "${fc:-0}" ] && [ "${ok:-0}" -gt "${fk:-0}" ]; then
      echo "  OK: feature active — mcycle and $ctr both fall when disabled"
    else
      echo "  FAIL: disabling the knob did not reduce mcycle and $ctr (feature not active)"; rc=1; fi
  else # report
    if [ "${oc:-0}" -gt "${fc:-0}" ]; then
      echo "  NOTE: feature changes cycles (${fc} -> ${oc})"
    else
      echo "  NOTE: NO cycle change (${oc} == ${fc}) — this knob does not bind in this scenario"; fi
  fi
}

echo "=== F1: residency-replica lookup latency (dsmem_multicluster) ==="
diff_feature f1 dsmem_multicluster 8 "-DVX_CFG_CTA_TABLE_LOOKUP_LATENCY=0" "-n4 -g4" load_lt assert

echo "=== F2: cluster-barrier latency (dsmem_barrier_stress) ==="
diff_feature f2 dsmem_barrier_stress 8 "-DVX_CFG_CLUSTER_BARRIER_LATENCY=0" "-n4 -g2 -p8" stalled_warps assert

echo "=== F3: outstanding-remote-store credit (dsmem_storeflood) ==="
# NOTE: F3 is expected NOT to bind — the DsmemLsuAdapter's 1-flit/cycle egress
# throttles remote stores below any credit, so this is a 'report', not 'assert'.
diff_feature f3 dsmem_storeflood 2 "-DVX_CFG_DSMEM_STORE_CREDITS=999999999" "-n2 -l4" stall_lsu report

echo
[ $rc -eq 0 ] && echo "dsmem_feature_diff: OK (F1,F2 proven active; F3 reported)" \
              || echo "dsmem_feature_diff: FAIL"
rm -rf "$TMP"
exit $rc
