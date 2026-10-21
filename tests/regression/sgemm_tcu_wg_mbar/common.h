#ifndef _SGEMM_TCU_WG_MBAR_COMMON_H_
#define _SGEMM_TCU_WG_MBAR_COMMON_H_

#include <stdint.h>

#ifndef WGMMA_NRC
  #define WGMMA_NRC 8
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

#define MODE_WAIT   0   // commit_group, wait_group<1>, then arrive on the previous stage's EMPTY
#define MODE_NOWAIT 1   // arrive on the stage's EMPTY right after its wgmma issues (no groups)
#define MODE_LAT    2   // MODE_WAIT, the last k-tile waiting with wait_group<0>, plus cycle probes
#define MODE_GROUPS 3   // MODE_WAIT, k-tile 0 adding blocking waits on a large group and on empty groups
#define MODE_REFILL 4   // one group per k-tile, waited with wait_group<0>; one stage allowed

typedef struct {
  uint32_t M, N, K;
  uint32_t mode;
  uint32_t stages;     // S >= 2 slab pairs, with FULL[S] and EMPTY[S]
  uint32_t parity;     // no spin: the warps meet in a register barrier where they would wait
  uint32_t slab_bytes; // one stage: the A slab, then the B slab, padded to a line
  uint32_t slab_base;  // byte offset of stage 0 in local memory
  uint32_t zero_base;  // byte offset of a zeroed slab, after the stages
  uint32_t extra;      // refill: wgmmas on the zero slab after each arrive
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t err_addr;   // ERR_WORDS per block and warp
  uint64_t lat_addr;   // LAT_WORDS per block and consumer warp
} kernel_arg_t;

#define ERR_TIMEOUT 0   // phases given up after SPIN_LIMIT polls, or not complete at a parity meet
#define ERR_TOKEN   1   // arrive tokens of the wrong phase
#define ERR_WORDS   2

#define LAT_WAIT    0   // cycles from the last wgmma's issue to wait_group<0> returning
#define LAT_AFTER   1   // cycles to then read the first eight accumulator registers
#define LAT_REF     2   // the same reads again, all registers long written
#define LAT_WORDS   3

#define SPIN_LIMIT (1u << 16)

// groups: wgmmas in k-tile 0's large group. Above 33, so done has passed its
// first boundary by more than half of a 6-bit count when the group commits.
#ifndef BIG_GROUP
#define BIG_GROUP 48
#endif

#endif
