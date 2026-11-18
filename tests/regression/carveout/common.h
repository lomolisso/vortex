#ifndef _CARVEOUT_COMMON_H_
#define _CARVEOUT_COMMON_H_

#include <stdint.h>

enum {
  CV_CHASE = 0,   // thread 0: check/write the shared marker words, gpu fence, an untimed and a timed pass of a line chase
  CV_FILL  = 1,   // every thread writes the shared pattern over lmem_words
  CV_TOUCH = 2,   // the block loads every word of [start, start + n lines)
  CV_DUMP  = 3,   // every thread copies lmem_words shared words to out
  CV_OCC   = 4,   // warp 0 of each block: core id, start cycle, end = the next multiple of spin; all warps held by barriers
  CV_STAMP = 5,   // thread 0: cycles since the previous stamp kernel's last one, and its own spin end, at out[2n], out[2n + 1]
};

#define CV_MARK_WORDS 1792  // 7 KiB: fits the smallest configuration with the per-block reserve

typedef struct {
  uint32_t kind;
  uint32_t n;          // chase: lines in the ring; touch: lines
  uint32_t lmem_words; // fill/dump: shared words
  uint32_t mark_old;   // chase: expected marker (0 = do not check)
  uint32_t mark_new;   // chase: marker written
  uint32_t spin;       // occ: period; stamp: cycles
  uint32_t line;       // touch, occ: bytes between lines
  uint64_t start_addr; // chase: first line of the ring; touch, occ: base
  uint64_t out_addr;   // chase: [misses, timed cycles, first-hop cycles, marker mismatches]; dump: words; occ: [block][3]
} kernel_arg_t;

#endif
