#!/bin/bash
# dsmem_testbench.sh <build_dir> [out_dir]
#
# The DSMEM/DXA test bench: runs the full DSMEM + DXA suite at the pocket-H100
# config (16 SMs / 8 KB SMEM / K<=8 / 2 MB L2), timeout-guarded, and reports
# per-run PASS/FAIL. Wires in the previously-orphaned DSMEM tests
# (dsmem_pingpong, dxa_multicast, dxa_tma_modes, cta_cluster) plus the three new
# F1-F3 tests, then runs the feature-active differential. A deadlock surfaces as
# a non-PASSED / non-zero-exit row (the SimX scheduler watchdog aborts a hang).
set -u
BUILD=${1:?usage: dsmem_testbench.sh <build_dir> [out_dir]}
OUT=${2:-$(mktemp -d)}
mkdir -p "$OUT"
fails=0; total=0

DSMEM="-DVX_CFG_DSMEM_ENABLE"
DXA="-DVX_CFG_EXT_DXA_ENABLE"
DXA_DSMEM="-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_DSMEM_ENABLE"
TCU_MC="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_DSMEM_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE"

# run <name> <app> <cores> <configs> <flags> <args>
run() {
  local name=$1 app=$2 cores=$3 cfg=$4 flags=$5 args=$6
  total=$((total+1))
  local argopt=(); [ -n "$args" ] && argopt=(--args="$args")
  ( cd "$BUILD" && timeout 120 env CONFIGS="$cfg" \
      ./ci/blackbox.sh --driver=simx --app="$app" --cores="$cores" $flags \
      --perf=1 "${argopt[@]}" > "$OUT/$name.log" 2>&1 )
  local ec=$?
  if grep -q '^PASSED' "$OUT/$name.log" && [ $ec -eq 0 ]; then
    printf "  PASS  %-34s (exit %s)\n" "$name" "$ec"
  else
    printf "  FAIL  %-34s (exit %s)  %s\n" "$name" "$ec" "$(grep -m1 -iE 'FAILED|error:|deadlock|stall-timeout' "$OUT/$name.log" | head -c 80)"
    fails=$((fails+1))
  fi
}

echo "### DSMEM direct addressing (F1 replica + F2 barrier + F3 drain) ###"
run pingpong_K2_c16       dsmem_pingpong 16 "$DSMEM" "--l2cache" "-n2"
run pingpong_K4_c8_w2     dsmem_pingpong 8  "$DSMEM" "--l2cache" "-n4 -w2"
run pingpong_K8_c8        dsmem_pingpong 8  "$DSMEM" "--l2cache" "-n8"
run pingpong_K8_c16       dsmem_pingpong 16 "$DSMEM" "--l2cache" "-n8"
run pingpong_K8_c8_nol2   dsmem_pingpong 8  "$DSMEM" ""          "-n8"

echo "### F1 multi-cluster isolation (K sweep + retire/re-admit) ###"
run multicluster_K2_G16   dsmem_multicluster 8  "$DSMEM" "--l2cache" "-n2 -g16"
run multicluster_K4_G4    dsmem_multicluster 8  "$DSMEM" "--l2cache" "-n4 -g4"
run multicluster_K8_G2    dsmem_multicluster 8  "$DSMEM" "--l2cache" "-n8 -g2"

echo "### F2 barrier stress (multi-warp, multi-cluster, deadlock stress) ###"
run barrier_K4_G2         dsmem_barrier_stress 8  "$DSMEM" "--l2cache" "-n4 -g2 -p8"
run barrier_K4_G2_w2      dsmem_barrier_stress 8  "$DSMEM" "--l2cache" "-n4 -g2 -p8 -w2"
run barrier_K8_c16_w2     dsmem_barrier_stress 16 "$DSMEM" "--l2cache" "-n8 -g2 -p8 -w2"

echo "### F3 store flood (credit sweep; correctness+no-deadlock) ###"
run storeflood_cred64     dsmem_storeflood 2 "$DSMEM"                              "--l2cache" "-n2 -l4"
run storeflood_cred32     dsmem_storeflood 2 "$DSMEM -DVX_CFG_DSMEM_STORE_CREDITS=32" "--l2cache" "-n2 -l4"
run storeflood_credhuge   dsmem_storeflood 2 "$DSMEM -DVX_CFG_DSMEM_STORE_CREDITS=999999999" "--l2cache" "-n2 -l4"

echo "### DXA multicast + TMA modes (F1 replica fan-out) ###"
run dxa_multicast_c8      dxa_multicast  8  "$DXA_DSMEM" "" ""
run dxa_tma_modes         dxa_tma_modes  1  "$DXA"       "--l2cache" ""
# cta_cluster is a SAME-CORE multicast showcase (core-local group_barrier, DXA
# only) — it co-resides all K CTAs on one core, so it runs single-core; DSMEM
# gang placement would scatter the cluster and hang the core-local barrier.
run cta_cluster           cta_cluster    1  "$DXA"       "--l2cache" ""

echo "### Registered DXA copy/gemm at pocket shape ###"
run dxa_copy_d2_c16       dxa_copy       16 "$DXA" "--l2cache" "-d2"
run dxa_copy_d5_c16       dxa_copy       16 "$DXA" "--l2cache" "-d5"
run dxa_copy_mcast_c1     dxa_copy_mcast 1  "$DXA" "--l2cache" ""
run dxa_kmajor_check      dxa_kmajor_check 1 "$DXA" "--l2cache" ""
run sgemm_wg_dxa_mcast_c8 sgemm_tcu_wg_dxa_mcast 8 "$TCU_MC" "--l2cache" "-m128 -n64 -k32"

echo
echo "### feature-active differential (F1/F2 proven, F3 reported) ###"
bash "$(dirname "$0")/dsmem_feature_diff.sh" "$BUILD" || fails=$((fails+1))

echo
echo "==== dsmem_testbench: $((total-fails))/$total runs PASSED, $fails failed ===="
exit $fails
