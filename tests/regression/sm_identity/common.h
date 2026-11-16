#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint64_t rec_addr;
  uint64_t arrive_addr;  // one counter all blocks arrive on
  uint32_t blocks;
  uint32_t bound;        // rendezvous polls before giving up
} id_arg_t;

#define REC_SMID   0  // %smid
#define REC_NSMID  1  // %nsmid
#define REC_POLLS  2  // polls until every block arrived; 0 on a timeout
#define REC_RUNS   3
#define REC_WORDS  4

#endif
