#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <dxa_meta.h>
#include <dxa_tmap.h>

// Forms, each issued by lane `lane` of warp 0 with every other lane inactive.
#define F_BULK_G2S   0   // cp.async.bulk.shared::cluster.global (funct3 3)
#define F_BULK_MC    1   // the same, .multicast::cluster to the issuer's own rank
#define F_COPY_S2S   2   // cp.async.bulk.shared::cluster.shared::cta (funct3 1)
#define F_TLOAD_1D   3   // cp.async.bulk.tensor.{1..5}d.shared::cluster.global (funct3 2), F_TLOAD_1D + rank - 1
#define F_TLOAD_5D   7
#define F_BSTORE     8   // cp.async.bulk.global.shared::cta.bulk_group (funct3 5)
#define F_BRED       9   // cp.reduce.async.bulk.global.shared::cta.bulk_group .add.u32
#define F_CRED       10  // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier .add.u32
#define F_TSTORE     11  // cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
#define F_TRED       12  // cp.reduce.async.bulk.tensor.2d.global.shared::cta.add (u32 map)
#define F_BPF        13  // cp.async.bulk.prefetch.L2.global (funct3 4)
#define F_TPF        14  // cp.async.bulk.prefetch.tensor.2d.L2.global (funct3 2, prefetch bit)
#define F_TMAP_PF    15  // prefetch.tensormap (funct3 7, op 0), then a 2D load through the map
#define F_TMAP_INV   16  // fence.proxy.tensormap::generic.acquire (funct3 7, op 1), then a 2D load
#define F_COUNT      17

// Local-memory layout (bytes).
#define LM_MBAR      0x0000            // NUM_MBARS mbarriers: one per form (K_FORMS), 4 per warp (K_COST)
#define NUM_MBARS    32
#define LM_SRC       0x0100            // the store/reduce/copy source pattern, SRC_BYTES
#define SRC_BYTES    512
#define LM_AREA      0x0400            // one result area per form, AREA_BYTES each
#define AREA_BYTES   512
#define LM_BYTES     (LM_AREA + F_COUNT * AREA_BYTES)

#define BULK_BYTES   256

// Tensors: u32 elements, dims TDIM, box TBOX (the first `rank` entries), coordinates TC.
#define TDIM0 8
#define TDIM1 4
#define TDIM2 3
#define TDIM3 2
#define TDIM4 2
#define TBOX0 8
#define TBOX1 2
#define TBOX2 2
#define TBOX3 2
#define TBOX4 1
#define TC0 0
#define TC1 1
#define TC2 1
#define TC3 0
#define TC4 1
#define TENSOR_WORDS (TDIM0 * TDIM1 * TDIM2 * TDIM3 * TDIM4)

// Maps in the maps buffer: rank 1..5 loads over `gsrc` (map r - 1), the 2D
// store map over `gdst_t` (5), the 2D reduce map over `gred_t` (6).
#define MAP_STORE    5
#define MAP_RED      6
#define NUM_MAPS     7

// Cost forms (K_COST): one timed issue loop per DXA form.
#define C_BULK         0
#define C_TENSOR       1
#define C_BULK_STORE   2
#define C_BULK_REDUCE  3
#define C_TENSOR_STORE 4
#define C_TENSOR_RED   5
#define C_PF_BULK      6
#define C_PF_TENSOR    7
#define C_TMAP_INVAL   8
#define C_TMAP_PF      9
#define C_COUNT        10
#define COST_N1        8
#define COST_N2        24
#define COST_BYTES     16

#define K_FORMS   0
#define K_COST    1
#define K_VISIBLE 2
#define K_VISIBLE_L1 3

// K_VISIBLE: the issuing thread's loads right after vx_bulk_wait_all() of a
// bulk store and of a bulk reduce, each VIS_BYTES from LM_AREA.
#define VIS_BYTES 4096
static_assert(LM_AREA + VIS_BYTES <= LM_BYTES, "K_VISIBLE sources from LM_AREA");
static inline uint32_t vis_word(uint32_t i) { return 0x3C000000u ^ (i * 40503u); }
static inline uint32_t vis_red_init(uint32_t i) { return 0x00100000u + i * 11u; }

// K_VISIBLE_L1 (H100 SM path): the issuing thread loads VL1_BYTES of the
// visibility buffer into its L1, bulk-stores VL1_WBYTES (vis_word) over their
// start and reads after vx_bulk_wait_group<0> (the written word, an unwritten
// line, an unrelated gsrc line: all miss, current values); then the same at
// VL1_R2 with vx_bulk_wait_group_read<0> (the L1 copy stays stale), a bulk
// reduce (add vis_word) at VL1_R3 with vx_bulk_wait_group<0> (current sums),
// then the cost of vx_l1_inval against a fence that goes to the L2.
#define VL1_BYTES  512
#define VL1_WBYTES 64
#define VL1_R2     1024
#define VL1_R3     2048
static_assert(VL1_R3 + VL1_BYTES <= 2 * VIS_BYTES, "K_VISIBLE_L1 regions lie in the visibility buffer");
static inline uint32_t vl1_old(uint32_t i) { return 0x5A000000u ^ (i * 2654435761u >> 8); }

// Per-thread output words.
#define O_DONE   0
#define O_ALIVE  1   // a value each thread holds across the issue
#define O_WAIT   2   // an mbarrier wait gave up
#define O_T1     3   // K_COST: cycles of COST_N1 copies (issuing lane)
#define O_T2     4   // K_COST: cycles of COST_N2 copies
#define O_STALE  6   // K_VISIBLE: stale words seen, store (low half) and reduce (high half);
                     // K_VISIBLE_L1: wrong words after wait_group 0 (low half), stale words kept after .read (high half)
#define O_TW     8   // K_VISIBLE_L1 cycles: the last written word after wait_group 0
#define O_TN     9   //   a word of the next, unwritten line
#define O_TU     10  //   the first word of gsrc (unrelated, resident before)
#define O_TH     11  //   the same word again
#define O_TS     12  //   the first written word after wait_group.read 0
#define O_TINV   13  //   vx_l1_inval
#define O_TFENCE 14  //   a fence (L1 drain, invalidate, L2 confirmation)
#define O_WORDS  16
#define MAX_COST_WARPS 8
static_assert(4 * MAX_COST_WARPS <= NUM_MBARS, "K_COST uses 4 mbarriers per warp");

#define WAIT_LIMIT 2000000u

typedef struct {
  uint64_t maps_addr;
  uint64_t gsrc_addr;   // TENSOR_WORDS u32 (also the bulk-load source)
  uint64_t gdst_addr;   // bulk store / bulk reduce destinations: 2 x BULK_BYTES
  uint64_t gst_addr;    // the 2D store tensor, TENSOR_WORDS u32
  uint64_t gred_addr;   // the 2D reduce tensor, TENSOR_WORDS u32
  uint64_t vis_addr;    // K_VISIBLE: store then reduce destination, VIS_BYTES each
  uint64_t out_addr;    // O_WORDS u32 per thread
  uint64_t dump_addr;   // LM_BYTES
  uint32_t kernel;      // K_*
  uint32_t lane;        // the issuing lane
  uint32_t form;        // K_COST: C_*
  uint32_t warps;       // K_COST: issuing warps
} tma_arg_t;

#endif
