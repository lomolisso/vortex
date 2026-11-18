#ifndef _L1_SETOVER_COMMON_H_
#define _L1_SETOVER_COMMON_H_

#include <stdint.h>

#define SO_LANES 32  // the widest warp

enum {
  SO_CHASE = 0,  // per-lane ring over host-written pointers: each hop a ld of the next pointer
  SO_MIX   = 1,  // the same ring, each hop also a sd of the hop number to the line's word 1
};

typedef struct {
  uint32_t kind;
  uint32_t lanes;
  uint32_t hops;
  uint64_t start_addr;  // [lanes] first pointers
  uint64_t out_addr;    // lane 0's last pointer
} kernel_arg_t;

#endif
