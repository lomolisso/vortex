#ifndef _SM_HOL_COMMON_H_
#define _SM_HOL_COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t lines;      // full-line global stores warp 0 of rank 0 issues before the timed access
  uint32_t pad;
  uint64_t gmem_addr;  // lines x 128 B
  uint64_t out_addr;   // [8] cycles, see kernel.cpp
} kernel_arg_t;

#endif
