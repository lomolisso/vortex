#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define MODE_BASIC 0   // every worker thread arrives with 1, then waits on its token
#define MODE_PC    1   // producer and consumer warps over a double buffer
#define MODE_TX    2   // a scripted op sequence, then software tx accounting by every worker warp
#define MODE_COUNT 3   // one elected lane per worker warp arrives with a count
#define MODE_GLOB  4   // pc over a global double buffer: the arrive must release global stores

#define NEG_NONE      0
#define NEG_UNDERFLOW 1   // more arrivals than pending
#define NEG_TXRANGE   2   // tx past its 21-bit range
#define NEG_GLOBAL    3   // an arrive on a global address
#define NEG_REMOTE    4   // an arrive on another block's object through the window

typedef struct {
  uint64_t err_addr;     // ERR_WORDS uint32 per thread
  uint64_t word_addr;    // uint64: the NUM_OBJS final object words, then the script's returned words
  uint64_t script_addr;  // tx: script_len pairs of uint64 {op, operand}
  uint32_t mode;
  uint32_t neg;
  uint32_t rounds;       // phases (basic, count, tx) or buffer fills (pc)
  uint32_t streamers;    // warps 0 .. streamers - 1 stream DXA copies, the rest are workers
  uint32_t tile_word;    // doubleword index of the first DXA tile
  uint32_t tile_elems;   // 32-bit elements per DXA tile
  uint32_t num_src_tiles;
  uint32_t script_len;
  uint32_t delay;        // idle iterations that skew the warps
  uint32_t parity;       // no spin: the workers meet in a register barrier, then test_wait must pass at once
  uint64_t glob_addr;    // glob: 2 * PC_ELEMS doublewords
  uint32_t burst;        // glob: stores per producer element and fill, the last one the tag
} mbar_arg_t;

#define ERR_COUNT 0   // failed checks of the thread
#define ERR_FIRST 1   // 1 + round of the first failed check, 0 if none
#define ERR_WHAT  2   // CHK_* code of the first failed check
#define ERR_NEG   3   // returned words with a negative tx (tx mode)
#define ERR_WORDS 4

#define CHK_TOKEN_PHASE 1   // a returned word's phase is not the round's
#define CHK_TOKEN_FIELD 2   // a returned word's pending or tx is outside its possible range
#define CHK_DATA        3   // data released by the barrier was not visible after the wait
#define CHK_WAIT        4   // the phase never completed (spin limit), or not by the register barrier (parity)

// Doubleword indexes in the block's local memory.
#define OBJ_BASE   0   // NUM_OBJS mbarrier objects, one per word
#define NUM_OBJS   8
#define LM_DONE    8   // the streamers stop once the workers set it
#define LM_DATA    16  // the mode's data words, then the DXA tiles
#define PC_ELEMS   64  // pc: doublewords per buffer
#define DATA_WORDS(block) (2 * (block) > 2 * PC_ELEMS ? 2 * (block) : 2 * PC_ELEMS)

// Objects of each mode.
#define OBJ_BAR    0   // basic, count: the phase barrier; tx: the concurrent tx barrier
#define OBJ_SCRIPT 1   // tx: the scripted sequence
#define OBJ_FULL   2   // pc: FULL[2]
#define OBJ_EMPTY  4   // pc: EMPTY[2]

// Script op codes are mbar_ref::Op, plus a plain store of the operand.
#define SCRIPT_STORE 0
#define MAX_SCRIPT   64

// tx mode: worker warp w > 0 completes TX_CHUNK * w bytes per phase.
#define TX_CHUNK   8

// Polls of an incomplete phase before a thread gives the round up.
#define SPIN_LIMIT (1u << 16)

// Fields of an object word (layout in vx_mbar.h).
#define MB_PENDING(w) ((uint32_t)((w) & 0xFFFFF))
#define MB_TX(w)      ((int64_t)((uint64_t)(w) << 3) >> 43)
#define MB_PHASE(w)   ((uint32_t)((uint64_t)(w) >> 63))

#endif
