#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define MODE_LOCAL    0   // every step by the owner on its own object
#define MODE_REMOTE   1   // the peer steps by rank 1 through the window, two cores of one socket
#define MODE_REMOTE2S 2   // the same over two sockets
#define MODE_RELAXED  3   // local, relaxed arrives and drops, relaxed waits
#define MODE_HINT     4   // local, waits with a suspend-time hint, then the hint timing probe

#define NEG_NONE      0
#define NEG_NOCOMPLETE 1  // a noComplete arrive that completes the phase
#define NEG_DROPZERO  2   // an arrive_drop of the whole expected count
#define NEG_SAMECORE  3   // a test_wait on a peer's object on the same core

// A script step: kind | side << 4 | n << 8.
#define K_INIT    0   // plain store of mbar_ref::init(n)
#define K_ARR     1   // arrive(n)
#define K_DROP    2   // arrive_drop(n)
#define K_NC      3   // arrive.noComplete(n)
#define K_DNC     4   // arrive_drop.noComplete(n)
#define K_DROPTX  5   // arrive_drop.expect_tx(n bytes)
#define K_CTX     6   // complete_tx(n bytes)
#define K_CLSDROP 7   // vortex::mbarrier::arrive_and_drop(n): no returned word
#define K_TW      8   // test_wait.parity(n): 0 or 1
#define K_WAITP   9   // try_wait.parity(n) until true: 1
#define SIDE_OWN  0
#define SIDE_PEER 1   // remote modes: rank 1 on rank 0's object through the window
#define STEP(k, side, n) ((uint32_t)(k) | ((uint32_t)(side) << 4) | ((uint32_t)(n) << 8))
#define STEP_KIND(s) ((s) & 0xf)
#define STEP_SIDE(s) (((s) >> 4) & 0xf)
#define STEP_N(s)    ((s) >> 8)

#define MAX_SCRIPTS 16
#define MAX_STEPS   32
#define NUM_HINTS   8
#define HINT_REPS   3

typedef struct {
  uint64_t steps_addr;   // uint32[MAX_SCRIPTS][MAX_STEPS]
  uint64_t rec_addr;     // uint64[MAX_SCRIPTS][MAX_STEPS]: returned word, or the wait's 0/1
  uint64_t word_addr;    // uint64[MAX_SCRIPTS]: each script object's final word
  uint64_t info_addr;    // uint64[INFO_WORDS]
  uint64_t hint_addr;    // uint64[HINT_REPS][NUM_HINTS + 2]: cycles | ok << 63
  uint32_t mode;
  uint32_t neg;
  uint32_t num_scripts;
  uint32_t spin_limit;
} ops_arg_t;

#define INFO_CORE0  0   // rank 0's core
#define INFO_CORE1  1   // rank 1's core (remote modes)
#define INFO_SPINS  2   // try_wait calls of K_WAITP steps that returned false
#define INFO_DONE   3   // 1 once rank 0 finished
#define INFO_WORDS  4

// Hint probe: H[0] is no hint, then the hints in ns; the last column times a test_wait.
#define HINT_NS { 0u, 10u, 100u, 1000u, 10000u, 100000u, 1000000u, 10000000u }

// Doubleword indexes in the block's local memory.
#define LM_OBJ   0             // MAX_SCRIPTS objects
#define LM_HINT  MAX_SCRIPTS   // the hint probe's object, never completed
#define LM_WORDS (MAX_SCRIPTS + 2)

#endif
