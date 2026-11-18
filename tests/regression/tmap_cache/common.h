#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <dxa_meta.h>
#include <dxa_tmap.h>

// Tests (-t). A hop = one map-form 2D load of a 16-B box by lane 0 of warp 0, from issue to its mbarrier phase.
#define T_RR       0   // M maps round robin, R rounds; the last round's hops
#define T_LRU      1   // maps 0-7, map 0, map 8, then map 0 timed
#define T_INVAL    2   // {in level 1, only in level 2} -> fence acquire -> use; control: only in level 2, no fence
#define T_PREFETCH 3   // never-used map: prefetch, spin, use; control: never-used map used at once
#define T_PARAM    4   // the argument's map: first use, second use; the second argument map prefetched, spin, use
#define T_LAUNCH   5   // launch 0 uses maps 0..M-1 twice, launch 1 uses each once
#define T_SHARE    6   // block 0 primes maps 0..M-1; block 1, on another SM, uses them after a flag
#define T_PLRU     7   // a stream of M distinct maps with map M every I-th hop; the probe hops
#define T_RELAUNCH 8   // launch 0 without a copy, its argument maps another tensor; launch 1 = T_PARAM

#define MAX_VALS   4096
#define WAIT_LIMIT 4000u
#define ISSUE_SPIN 400u  // cycles from the hop barrier to the copy issue: the waiter is asleep by then

#define LM_MBAR    0x0000
#define LM_T0      0x0008  // the issuing warp's clock before its copy
#define LM_FAIL    0x0010  // a hop timed out: both warps stop timing
#define LM_DATA    0x0040

// Output words (u32).
#define O_BASE     0   // the level-1 hit: a reused map's fourth hop
#define O_ERR      1
#define O_CORE     2   // T_SHARE: block 0's and block 1's core ids
#define O_KNOBS    4   // 6 words: L2 trip of map 0 and of the argument map, L1_ENTRIES, L1_REPL, L2_ENTRIES, L2_WAYS
#define O_T        10  // 8 single results
#define O_N        18  // number of hop values
#define O_VALS     20
#define O_WORDS    (O_VALS + MAX_VALS)

// The serial trip of a level-2 hit (VX_tmap_l2): tag, compare, data and answer
// for each L1-line beat of the 128-B map, plus the lookup's request register,
// its park and the head's re-lookup of the filled entry. The DXA core adds it
// whole; the hop's wake-up path can absorb part of it, so it bounds a level-2
// hit's cost over level 1 from above.
#define TMAP_L2_BEAT_CYCLES  4
#define TMAP_L2_FIXED_CYCLES 3
static inline uint32_t tmap_l2_trip(uint64_t map, uint32_t l1_line) {
  return TMAP_L2_BEAT_CYCLES * (uint32_t)((map % l1_line + 127) / l1_line + 1) + TMAP_L2_FIXED_CYCLES;
}

typedef struct {
  vx_tensor_map_t map;    // first: the argument buffer is 64-byte aligned
  vx_tensor_map_t map2;
  uint64_t maps_addr;     // the global maps, 128 B apart
  uint64_t out_addr;
  uint64_t flag_addr;
  uint32_t test;
  uint32_t nmaps;         // M
  uint32_t rounds;        // R (T_RR), I (T_PLRU)
  uint32_t spin;          // cycles (T_PREFETCH, T_PARAM)
  uint32_t launch;        // T_LAUNCH: 0 or 1
  uint32_t pad;
} tc_arg_t;

#endif
