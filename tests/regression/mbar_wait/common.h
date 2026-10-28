#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define MODE_LATE  0   // a late arriver: the waiters poll with try_wait until it arrives
#define MODE_SLEEP 1   // every warp sleeps for many time limits, the producer included
#define MODE_MANY  2   // every lane of every warp waits, on one object or on several
#define MODE_MIXED 3   // the phase completes in the middle of a warp's lanes

typedef struct {
  uint64_t err_addr;     // ERR_WORDS uint32 per thread
  uint64_t word_addr;    // uint64: the NUM_OBJS final object words
  uint32_t mode;
  uint32_t rounds;       // phases; sleep: the producer's time-outs
  uint32_t delay;        // idle iterations of the arriver before each arrive
  uint32_t objs;         // many: objects waited on
  uint32_t lanes;        // many: lanes of a warp wait on different objects
  uint32_t spin;         // poll with test_wait instead of try_wait
  uint32_t parity;       // no timing-dependent loop counts (late)
  uint32_t streamers;    // warps 0 .. streamers - 1 stream DXA copies, the rest are workers
  uint32_t tile_word;    // doubleword index of the first DXA tile
  uint32_t tile_elems;   // 32-bit elements per DXA tile
  uint32_t num_src_tiles;
  uint32_t pad;
} mbar_arg_t;

#define ERR_COUNT    0   // failed checks of the thread
#define ERR_FIRST    1   // 1 + round of the first failed check, 0 if none
#define ERR_WHAT     2   // CHK_* code of the first failed check
#define ERR_CALLS    3   // wait calls (try_wait or test_wait polls)
#define ERR_TIMEOUTS 4   // try_wait false after at least the time limit
#define ERR_EARLY    5   // try_wait false in less than the time limit (lanes on several objects)
#define ERR_MINFALSE 6   // fewest cycles a false try_wait took, ~0 if none
#define ERR_MIXED    7   // lane 0: rounds whose first try_wait split the warp's lanes
#define ERR_ELAPSED  8   // cycles of the thread's wait phase
#define ERR_LAGMIN   9   // arriver lane 0: fewest cycles from the workers' meeting to issuing an arrive
#define ERR_LAGMAX   10  // arriver lane 0: most cycles from the workers' meeting to an arrive's answer
#define ERR_SLEPT    11  // try_waits that suspended the warp
#define ERR_WORDS    12

#define CHK_DATA     1   // data released by the arrive was not visible after the wait
#define CHK_WAIT     2   // the phase never completed (poll limit)
#define CHK_SPURIOUS 3   // a try_wait on the warp's one object returned false before the time limit
#define CHK_NEVER    4   // a try_wait on a phase nobody completes returned true, or before the time limit
#define CHK_CALLS    5   // parity: a wait took another number of calls than the fixed schedule allows

// Doubleword indexes in the block's local memory.
#define OBJ_BAR    0    // late, sleep, mixed: the phase barrier, one arrival per phase
#define OBJ_NEVER  1    // an object nobody arrives on
#define OBJ_MANY   2    // many: MAX_OBJS objects
#define MAX_OBJS   16
#define NUM_OBJS   (OBJ_MANY + MAX_OBJS)
#define LM_DONE    NUM_OBJS   // the streamers stop once the workers set it
#define LM_DATA    (NUM_OBJS + 2)   // one tag per object, stored before each arrive
#define LM_TILES   (LM_DATA + NUM_OBJS)

// mixed: the longest delay of a waiter warp's first poll, in nops, and the
// test_waits it sends ahead of that poll.
#define MIXED_NOPS 256
#define MIXED_LEAD 3

// Polls before a thread gives a wait up.
#define SPIN_LIMIT (1u << 16)
#define TRY_LIMIT  (1u << 12)

#endif
