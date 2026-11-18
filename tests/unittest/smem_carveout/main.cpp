// smem_carveout.h on the host: the no-hint rule against a brute-force search,
// the hint rule on the spec's method cells, the measured driver rows, the L1
// left over, the occupancy formula and the refusal.

#include <smem_carveout.h>

#include <algorithm>
#include <cstdio>

namespace {

int g_fail = 0;
long g_pass = 0;

#define CHECK(c, ...)                                   \
  do {                                                  \
    if (!(c)) {                                         \
      if (++g_fail < 20) {                              \
        std::printf("FAIL %s:%d ", __FILE__, __LINE__); \
        std::printf(__VA_ARGS__);                       \
        std::printf("\n");                              \
      }                                                 \
    } else {                                            \
      ++g_pass;                                         \
    }                                                   \
  } while (0)

// The H100 shape (05 §2, 03 §3): 256 KiB array, 228 KiB max.
SmemCarveCfg h100(const uint32_t (&kb)[SMEM_CARVE_NUM_CONFIGS], uint32_t tpw, uint32_t warps) {
  SmemCarveCfg c = {{}, 262144, 0, 256, 233472, warps < 32 ? warps : 32, warps < 8 ? warps : 8, warps, tpw};
  for (uint32_t i = 0; i < SMEM_CARVE_NUM_CONFIGS; ++i)
    c.configs_kb[i] = kb[i];
  return c;
}

// Most resident blocks, then the smallest configuration.
int32_t brute(uint32_t lmem, uint32_t bt, bool cl, const SmemCarveCfg& cfg) {
  uint32_t fp = smem_cta_footprint(lmem, cfg);
  if (fp > cfg.configs_kb[SMEM_CARVE_NUM_CONFIGS - 1] * 1024u)
    return -1;
  uint32_t w = (bt + cfg.threads_per_warp - 1) / cfg.threads_per_warp;
  uint32_t lim = cfg.warps_per_sm / w;
  uint32_t cap = cl ? cfg.max_ctas_per_sm_cluster : cfg.max_ctas_per_sm;
  if (cap < lim)
    lim = cap;
  int32_t best = -1;
  uint32_t bestb = 0;
  for (uint32_t i = 0; i < SMEM_CARVE_NUM_CONFIGS; ++i) {
    uint32_t c = cfg.configs_kb[i] * 1024, b = 0;
    while ((b + 1) * uint64_t(fp) <= c && b < lim)
      ++b;
    if (best < 0 || b > bestb || (b == bestb && int32_t(c) < best)) {
      best = int32_t(c);
      bestb = b;
    }
  }
  return best;
}

const uint32_t kH100List[SMEM_CARVE_NUM_CONFIGS] = {8, 16, 32, 64, 100, 132, 164, 196, 228};
const uint32_t kShortList[SMEM_CARVE_NUM_CONFIGS] = {16, 16, 32, 64, 96, 128, 160, 192, 228};

void brute_sweep(const uint32_t (&kb)[SMEM_CARVE_NUM_CONFIGS]) {
  const uint32_t tpws[] = {16, 32}, wps[] = {4, 32, 64};
  for (uint32_t tpw : tpws)
    for (uint32_t wp : wps)
      for (int cl = 0; cl < 2; ++cl) {
        SmemCarveCfg cfg = h100(kb, tpw, wp);
        for (uint32_t bt = 32; bt <= 1024; bt += 32)
          for (uint32_t k = 0; k <= 227; ++k) {
            int32_t a = smem_config_choose(k * 1024, bt, cl, -1, cfg);
            int32_t b = brute(k * 1024, bt, cl, cfg);
            CHECK(a == b, "brute list0=%u tpw=%u w=%u cl=%d bt=%u kb=%u: got %d want %d", kb[0], tpw, wp, cl, bt, k,
                  a, b);
          }
      }
}

}  // namespace

int main() {
  // (a) the no-hint rule at the default list and at a shorter list with repeats.
  brute_sweep(kH100List);
  brute_sweep(kShortList);

  SmemCarveCfg c = h100(kH100List, 32, 64);

  // (b) 05 §2's method: dyn = config - 1 KiB with hint floor(c * 100 / 228).
  for (uint32_t k : kH100List) {
    int32_t h = int32_t(k * 100 / 228);
    int32_t s = smem_config_choose((k - 1) * 1024, 256, false, h, c);
    CHECK(s == int32_t(k * 1024), "method c=%u hint=%d: got %d", k, h, s);
  }

  // (c) the measured no-hint rows (E12b_h100 §8.5) count the H100's 1 KiB per-CTA reserve, which Vortex
  // does not keep; (a) checks our rule, the H100's choices are reference data.
  const uint32_t dyn[] = {0, 1, 4, 16, 48, 99};
  const uint32_t want32[] = {32, 64, 164, 228, 196, 228};
  const uint32_t want1024[] = {8, 8, 16, 64, 100, 228};
  const uint32_t h256[] = {32, 32, 100, 164, 196, 228};
  for (int i = 0; i < 6; ++i) {
    std::printf("H100REF metric=carveout_kib_t32_dyn%u h100=%u vortex=%d\n", dyn[i], want32[i],
                smem_config_choose(dyn[i] * 1024, 32, false, -1, c) / 1024);
    std::printf("H100REF metric=carveout_kib_t256_dyn%u h100=%u vortex=%d\n", dyn[i], h256[i],
                smem_config_choose(dyn[i] * 1024, 256, false, -1, c) / 1024);
    std::printf("H100REF metric=carveout_kib_t1024_dyn%u h100=%u vortex=%d\n", dyn[i], want1024[i],
                smem_config_choose(dyn[i] * 1024, 1024, false, -1, c) / 1024);
  }
  std::printf("H100REF metric=carveout_kib_t32_dyn4_clustered h100=64 vortex=%d\n",
              smem_config_choose(4096, 32, true, -1, c) / 1024);
  std::printf("H100REF metric=carveout_kib_t32_dyn0_hint0 h100=32 vortex=%d\n",
              smem_config_choose(0, 32, false, 0, c) / 1024);
  std::printf("H100REF metric=carveout_kib_t1024_dyn0_hint0 h100=8 vortex=%d\n",
              smem_config_choose(0, 1024, false, 0, c) / 1024);
  const int32_t hints[] = {13, 28, 43};
  const uint32_t hwant[] = {64, 132, 196};
  for (int i = 0; i < 3; ++i)
    std::printf("H100REF metric=carveout_kib_t32_dyn1_hint%d h100=%u vortex=%d\n", hints[i], hwant[i],
                smem_config_choose(1024, 32, false, hints[i], c) / 1024);
  // A launch that addresses no shared memory is bounded by its other limits only: the smallest configuration.
  CHECK(smem_config_choose(0, 32, false, -1, c) == int32_t(kH100List[0] * 1024), "dyn 0 takes the smallest");

  // (d) the L1 left over and the occupancy floor(228 / smem) for 256-thread blocks (03 §3 counts
  // smem + 1 KiB on the H100).
  for (uint32_t k : kH100List)
    CHECK(smem_usable_l1(k * 1024, c) == (256 - k) * 1024, "usable l1 at %u KiB", k);
  const uint32_t occ[][2] = {{0, 8}, {16, 8}, {48, 4}, {76, 3}, {100, 2}, {114, 2}, {115, 1}};
  for (auto& o : occ) {
    uint32_t want = o[0] ? 228 / o[0] : 8;
    if (want > 8)
      want = 8;
    uint32_t got = smem_occupancy(o[0] * 1024, 256, false, c);
    CHECK(got == want && want == o[1], "occupancy %u KB: got %u want %u", o[0], got, o[1]);
    std::printf("H100REF metric=occupancy_t256_smem%ukib h100=%u vortex=%u\n", o[0],
                std::min<uint32_t>(8, 228 / (o[0] + 1)), got);
  }

  // (e) one block above the largest configuration is refused, with or without a hint.
  CHECK(smem_config_choose(228 * 1024, 32, false, -1, c) == 228 * 1024, "228 KiB fits");
  CHECK(smem_config_choose(228 * 1024 + 1, 32, false, -1, c) == -1, "refusal");
  CHECK(smem_config_choose(228 * 1024 + 1, 32, false, 50, c) == -1, "refusal with hint");

  std::printf("smem_carveout: %ld passed, %d failed\n", g_pass, g_fail);
  std::printf(g_fail ? "FAILED\n" : "PASSED\n");
  return g_fail ? 1 : 0;
}
