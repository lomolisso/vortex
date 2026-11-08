#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <dxa_meta.h>
#include <dxa_tmap.h>

// Jobs, run in order by every warp of the block: a job is executed by its
// warp only, J_SYNC by all.
#define J_LOAD    0   // tensor load through `sel` (tile or im2col by the map), complete_tx `tx`
#define J_STORE   1   // tensor store of the box from local memory `dst`
#define J_RED     2   // tensor reduction `op` of the box from local memory `dst`, typed by the map
#define J_BWAIT   3   // commit_group + wait_group 0
#define J_PF      4   // prefetch.tensormap
#define J_ACQ     5   // fence.proxy.tensormap::generic.acquire
#define J_REL     6   // fence.proxy.tensormap::generic.release
#define J_COPY    7   // local memory `dst` = the 128 bytes of the global map at `map_off` (generic loads)
#define J_ZERO    8   // local memory [dst, dst + 128) = 0
#define J_RADDR   9   // tensormap.replace global_address = val
#define J_RSTRIDE 10  // tensormap.replace global_stride ord `arg` = val
#define J_RALL    11  // every field of the map from the argument's spec (a zero buffer first)
#define J_CP      12  // tensormap.cp_fenceproxy: global map at `map_off` = local `dst`, lanes in `mask` (0: all)
#define J_SYNC    13  // __syncthreads

// Where a job's map is (J_LOAD, J_STORE, J_RED, J_PF, J_ACQ, J_RADDR, J_RSTRIDE, J_RALL).
#define SEL_PARAM  0  // &arg->map, the __grid_constant__ form
#define SEL_GLOBAL 1  // maps buffer + map_off
#define SEL_LMEM   2  // local memory + map_off
#define SEL_SLOT   3  // descriptor slot `slot` (no map)

#define K_JOBS  0
#define K_MCAST 1     // the cluster multicast through the param map, then the jobs
#define K_PERF  2     // -p: warp 0 caches maps 0-7, then PERF_PREFETCHES prefetches (maps 8+r or r%8), a load
#define PERF_PREFETCHES 48  // K_PERF: the timed batch
#define K_PF    3     // -m prefetch: cold, prefetched and cached loads, timed (`linger`: the idle cycles)

typedef struct {
  uint32_t kind;
  uint32_t warp;
  uint32_t sel;
  uint32_t map_off;
  uint32_t slot;
  uint32_t rank;
  uint32_t im2col;
  uint32_t dst;       // local-memory byte offset
  uint32_t tx;
  uint32_t op;
  uint32_t mask;
  uint32_t arg;
  int32_t  coord[5];
  uint32_t offs[3];
  uint64_t val;
} job_t;

// J_RALL: the fields, in PTX's operand meanings.
typedef struct {
  uint64_t gaddr;
  uint64_t gstride[4];
  uint32_t rank;
  uint32_t box[5];
  uint32_t gdim[5];
  uint32_t es[5];
  uint32_t elemtype;
  uint32_t interleave;
  uint32_t swizzle;
  uint32_t fill;
} rspec_t;

typedef struct {
  vx_tensor_map_t map;  // first: the argument buffer is 64-byte aligned
  rspec_t  spec;
  uint64_t jobs_addr;
  uint64_t maps_addr;
  uint64_t err_addr;    // ERR_WORDS uint32 per thread
  uint64_t dump_addr;   // lmem_bytes per block
  uint32_t num_jobs;
  uint32_t kernel;      // K_*
  uint32_t lmem_bytes;
  uint32_t perf_miss;   // K_PERF: map 8 + r of prefetch r, else map r % 8
  uint32_t neg;
  uint32_t linger;
  uint32_t mc_tx;       // K_MCAST: the box bytes every block receives
  int32_t  mc_coord[2];
  uint32_t mc_dst;
  uint32_t pad;
} __attribute__((aligned(64))) tmap_arg_t;

#define LM_MBAR 0
#define LM_SMAP 128     // a map in local memory (J_COPY, J_ZERO, J_RALL, J_CP)
#define LM_DATA 256

#define ERR_DONE   0
#define ERR_WAIT   1    // an mbarrier phase that did not complete
#define ERR_ARG_LO 2    // thread 0: &arg->map
#define ERR_ARG_HI 3
#define ERR_T0     4    // thread 0: K_PERF the timed cycles; K_PF cold, prefetched, cached
#define ERR_BOX    7    // K_PERF, thread t: word t % 4 of the landed box
#define ERR_TMO    8    // K_PERF: try_waits that ended at the time limit
#define ERR_WORDS  12

#define WAIT_LIMIT 256


static inline uint32_t mix32(uint32_t a, uint32_t b) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

// Word w of block b's local memory before the jobs.
static inline uint32_t canary_word(uint32_t b, uint32_t w) { return mix32(0x7A0000u + b, w); }

#endif
