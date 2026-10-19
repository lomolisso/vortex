#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define MODE_MP    0   // producer amoswap.d.rl, consumer plain flag load
#define MODE_MPAQ  1   // producer amoswap.d.rl, consumer amoor.d.aq on the flag
#define MODE_LAT   2   // fixed-count latency probes of .aq and .rl
#define MODE_WIDE  3   // every lane of two warps per block: mixed .aq/.rl atomics

typedef struct {
  uint64_t out_addr;     // OUT_WORDS per block, then rank 0's LAT_KINDS x LAT_ITERS samples
  uint32_t mode;
  uint32_t iters;        // producer rounds
  uint32_t fence;        // control: vx_fence() and a plain amoswap.d instead of .rl
  uint32_t flood;        // rank 3 floods rank 1 with window stores, rank 1 hammers its banks
  uint32_t dxa;          // rank 1 streams DXA copies into its own local memory
  uint32_t tile_elems;   // 32-bit elements per DXA tile
} amo_arg_t;

#define OUT_RAN       0
#define OUT_RANK      1
#define OUT_CORE      2
#define OUT_VIOL      3   // consumer: FLAG = f seen, then DATA < f read; wide: wrong words
#define OUT_READS     4   // consumer: new flag values observed
#define OUT_WORK      5   // flood stores, hammer atomics or DXA copies issued
#define OUT_FIRST_F   6   // flag value at the first violation
#define OUT_FIRST_D   7   // DATA read at the first violation (low word)
#define OUT_WORDS     8

// Doubleword indexes in each block's local memory. In mp the producer stores
// DATA into rank 1 and swaps FLAG into rank 2; in mpaq DATA is in rank 2 and
// FLAG in rank 1.
#define AO_DATA   0
#define AO_FLAG   8
#define AO_DONE   16
#define AO_FLOOD  24   // 64 doublewords that rank 3 floods in rank 1
#define AO_HAMMER 88   // 64 doublewords that rank 1 hammers with local AMOs
#define AO_LAT    152  // lat: the remote word (in rank 1) and the local AMO word
#define AO_TILE   160  // DXA tile
#define AO_WORDS  AO_TILE

// lat samples, each the cycles of one probe sequence.
#define LAT_R     0   // amoadd.d rd to rank 1, then a use of rd: the round trip
#define LAT_A     1   // amoadd.d.aq zero to rank 1, then the next instruction
#define LAT_C     2   // control: A without .aq
#define LAT_B     3   // amoswap.d rd on the own word, then a use of rd
#define LAT_S     4   // store to rank 1, fence, then B: an upper bound for L
#define LAT_L     5   // store to rank 1, then B with amoswap.d.rl
#define LAT_L0    6   // control: L without .rl
#define LAT_KINDS 7
#define LAT_ITERS 16

// wide: per-lane counters, region k at doubleword k * blockDim.x, each ending
// at the round count; then the block's error count. The global counter is the
// doubleword after the blocks' OUT_WORDS.
#define WIDE_MIX   0   // amoadd.d.aqrl, odd lanes own block, even lanes next rank
#define WIDE_AQ    1   // amoadd.d.aq on the own word; the old value is checked
#define WIDE_RL    2   // amoadd.d.rl into the next rank
#define WIDE_ST    3   // plain store to the own word
#define WIDE_STR   4   // plain store into the next rank
#define WIDE_LRSC  5   // lr.d.aq / sc.d.rl increment of the own word
#define WIDE_KINDS 6
#define WIDE_ERR   WIDE_KINDS
#define WIDE_WORDS(nb) ((WIDE_KINDS) * (nb) + 1)
#define WIDE_LRSC_TRIES 64

#endif
