#ifndef _MBAR_COST_COMMON_H_
#define _MBAR_COST_COMMON_H_

#include <stdint.h>

enum {
  MC_LAT    = 0,  // thread 0: dependent chases of each op (MC_OP_*), then the chase's ALU cost after a load
  MC_ISSUE  = 1,  // lane 0 of every warp: back-to-back expect_tx / complete_tx streams
  MC_UPDATE = 2,  // every lane arrives (relaxed) on one of nbars barriers in different bank pairs
  MC_BULK   = 3,  // thread 0: 128-B bulk copies completing on one mbarrier, alone or beside an op stream (nbars = 1)
};

enum {
  MC_OP_ARRIVE = 0, MC_OP_ARRIVE_TX, MC_OP_ARRIVE_DROP, MC_OP_TEST_WAIT, MC_OP_TRY_WAIT, MC_OP_TEST_PENDING,
  MC_OP_ALU, MC_LAT_N
};

enum { MC_IS_EXPECT_TX = 0, MC_IS_COMPLETE_TX, MC_ISSUE_N };

#define MC_UNROLL 64

typedef struct {
  uint32_t kind;
  uint32_t reps;      // timed groups of MC_UNROLL ops per warp (lat, issue) or of 16 arrives per lane (update)
  uint32_t nbars;     // update: barriers, one per bank pair; bulk: 1 = warp 1 streams complete_tx meanwhile
  uint32_t pad;
  uint64_t out_addr;  // lat: cycles per kind [MC_LAT_N]; issue: [MC_ISSUE_N]; update, bulk: [1]
  uint64_t src_addr;  // bulk: 128 B of global memory
} kernel_arg_t;

#endif
