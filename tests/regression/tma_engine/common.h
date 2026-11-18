#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Modes (-m).
#define M_INTAKE 0   // warps 0..W-1, lane 0 each: P bulk loads of B bytes back to back
#define M_STORE  1   // one thread: G groups of S bulk stores of B bytes, a commit each, then wait_group K, wait_group 0
#define M_REDUCE 2   // one thread: one bulk .add.u32 reduction of B bytes, then wait_group 0
#define M_READ   3   // one thread: one bulk store of B bytes, wait_group.read 0, then wait_group 0

// M_READ -c: the timed store's offset in gdst, past the lines the warm-up pass wrote.
#define COLD_OFFSET(b) (((b) + 4095u) / 4096u * 4096u)

#define MAX_WARPS  32
#define MAX_COPIES 16
#define MAX_GROUPS 64
#define SRC_STRIDE 4096           // M_INTAKE: the largest per-copy source stride

#define LM_MBAR    0x0000         // one mbarrier per warp
#define LM_TS      0x0100         // M_STORE: commit times, copied out after the timed waits
#define LM_DATA    0x0400         // load destination (shared by every warp) / store and reduce source

// Output words (u32): per warp {t_first, t_last, t_done} as lo/hi pairs, then the single-thread results.
#define O_WARP     0
#define O_WARP_W   6
#define O_COMMIT   (O_WARP + MAX_WARPS * O_WARP_W)   // M_STORE: cycles after each commit, from the first issue
#define O_WAITK    (O_COMMIT + MAX_GROUPS)           // wait_group K
#define O_WAIT0    (O_WAITK + 1)                     // the following wait_group 0
#define O_T1       (O_WAIT0 + 1)                     // M_REDUCE: issue to wait_group 0; M_READ: to wait_group.read 0
#define O_T2       (O_T1 + 1)                        // M_READ: wait_group.read 0 to wait_group 0
#define O_T3       (O_T2 + 1)                        // M_READ: issue to wait_group 0 alone, same lines
#define O_ERR      (O_T3 + 1)
#define O_KNOBS    (O_ERR + 1)                       // the build's knobs, 4 words
#define O_WAITB    (O_KNOBS + 4)                     // M_STORE: wait_group K with nothing pending
#define O_WORDS    (O_WAITB + 1)

#define WAIT_LIMIT 4000000u

typedef struct {
  uint64_t gsrc_addr;
  uint64_t gdst_addr;
  uint64_t out_addr;
  uint32_t mode;
  uint32_t warps;
  uint32_t copies;   // M_INTAKE: P per warp; M_STORE: S per group
  uint32_t bytes;
  uint32_t groups;
  uint32_t waitk;
  uint32_t src_stride; // M_INTAKE: each copy reads its own global lines
  uint32_t cold;       // M_READ: the timed store goes to lines nothing touched before
} eng_arg_t;

#endif
