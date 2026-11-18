// A kernel the standalone mains run without the runtime: VX_DCR_KMU_SMEM_CONFIG keeps its reset
// value, the largest shared region. Thread 0 stores and reloads the last shared word of that
// region, then chases an L1 working set of the reset configuration's capacity: the second pass
// must not miss, and one step more must. A timed load is a miss above the midpoint of the first
// line's timed L1 hit and the fastest first-pass load after the fence (which clears the L1).

#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <smem_carveout.h>

#define LINE VX_CFG_DCACHE_LINE_SIZE
#define ALIGN ((uint64_t)VX_CFG_L1_TAG_SETS * LINE)
#define STEP_BYTES 1024

static uint8_t pool[VX_CFG_SM_SRAM_BYTES + 2 * ALIGN];

static uint64_t ring_base() { return ((uint64_t)(uintptr_t)pool + ALIGN - 1) / ALIGN * ALIGN; }

static uint32_t gcd(uint32_t a, uint32_t b) {
  while (b) {
    uint32_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

// One cycle over n lines, visited by a stride coprime with n so that neighbours in time are far apart.
static void build_ring(uint32_t n) {
  const uint64_t base = ring_base();
  uint32_t stride = n * 5 / 8 + 1;
  while (gcd(stride, n) != 1) ++stride;
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t a = (uint32_t)(((uint64_t)i * stride) % n);
    uint32_t b = (uint32_t)(((uint64_t)(i + 1) * stride) % n);
    *(volatile uint64_t*)(base + (uint64_t)a * LINE) = base + (uint64_t)b * LINE;
  }
}

static uint64_t timed_ld(uint64_t* p) {
  uint64_t t0 = vx_rdcycle();
  uint64_t v;
  __asm__ volatile ("ld %0, 0(%1)\n add %0, %0, zero\n" : "=&r"(v) : "r"(*p) : "memory");
  uint64_t t1 = vx_rdcycle();
  *p = v;
  return t1 - t0;
}

static uint32_t chase(uint32_t n) {
  build_ring(n);
  uint64_t p = ring_base();
  __asm__ volatile ("fence iorw, iorw" ::: "memory");
  uint64_t q = p;
  uint64_t miss = timed_ld(&p);
  const uint64_t hit = timed_ld(&q);
  for (uint32_t i = 1; i < n; ++i) {
    const uint64_t c = timed_ld(&p);
    miss = c < miss ? c : miss;
  }
  const uint64_t thr = (hit + miss) / 2;
  uint32_t misses = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint64_t t0 = vx_rdcycle();
    uint64_t v;
    __asm__ volatile ("ld %0, 0(%1)\n add %0, %0, zero\n" : "=&r"(v) : "r"(p) : "memory");
    uint64_t t1 = vx_rdcycle();
    p = v;
    misses += (t1 - t0 > thr);
  }
  return misses;
}

int main() {
  int errors = 0;
  volatile uint32_t* smem = (volatile uint32_t*)csr_read(VX_CSR_CTA_LMEM_ADDR);
  const uint32_t top = VX_CFG_SMEM_PER_SM_MAX / 4 - 1;
  smem[top] = 0x5eed0000u + top;
  if (smem[top] != 0x5eed0000u + top) {
    vx_printf("shared word at %d B: read back wrong\n", top * 4);
    ++errors;
  }
  const uint32_t cap = smem_usable_l1(VX_CFG_SMEM_PER_SM_MAX, smem_carve_cfg()) / LINE;
  uint32_t at = chase(cap);
  uint32_t over = chase(cap + STEP_BYTES / LINE);
  vx_printf("reset configuration: L1 %d lines; second-pass misses %d at capacity, %d one step above\n", cap, at, over);
  errors += (at != 0);
  errors += (over == 0);
  return errors;
}
