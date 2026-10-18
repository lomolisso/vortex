#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define MODE_BAR      0   // the producer's store is ordered by the group barrier
#define MODE_FENCE    1   // the producer's store is ordered by vx_fence() before a flag store
#define MODE_FENCEBAR 2   // vx_fence() and then the group barrier, with no polling

typedef struct {
  uint64_t res_addr;     // RES_WORDS per thread, then one copy count per streamer warp
  uint32_t mode;
  uint32_t rounds;
  uint32_t dxa_on;       // the streamer warps keep DXA tile copies in flight
  uint32_t tile_elems;   // 32-bit elements per DXA tile
  uint32_t num_src_tiles;
  uint32_t streamers;    // warps 0 .. streamers - 1 stream DXA copies, each into its own tile
  uint32_t x_word;       // doubleword index of X past the tiles; the flag sits on the next bank
  uint32_t delay;        // producer idle iterations before its stores
  uint32_t burst;        // stores per group lane and round into X's bank; the producer's last carries X
  uint32_t banks;        // local-memory banks, interleaved by doubleword
} kernel_arg_t;

#define RES_STALE     0   // reads of X older than the round; ~0 if the thread never finished
#define RES_FIRST     1   // the first round with a stale read, 0 if none
#define RES_TIMEOUT   2   // fence mode: rounds whose flag never arrived
#define RES_WORDS     3

// Polls of the flag before a consumer gives the round up.
#define SPIN_LIMIT    (1u << 16)

// Lane l's burst word is X + banks * (SINK_ROWS(banks) + l): on X's bank, past the flag and done words.
#define SINK_ROWS(banks) (((banks) + 2) / (banks))

#endif
