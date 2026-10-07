#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Each mode races accesses to one word against the fill of its line, so that
// an access arrives while older accesses to the word still wait in the line's
// miss chain. The data cache must apply the accesses to a word in the order
// it accepted them.
enum {
  MODE_RAW,       // store, then a load of the word (one hart)
  MODE_WAR,       // load, then a store to the word (one hart)
  MODE_STORE_AMO, // store, then an atomic add on the word (one hart)
  MODE_STORE_LR,  // store, then LR of the word (one hart)
  MODE_AMO_STORE, // atomic add by one warp racing a store by another
  MODE_AMO_LOAD,  // atomic add, then a load of the word (one hart)
  MODE_AMO_STORM, // every hart: a load, then atomics on a cold and a resident line
  NUM_MODES
};

typedef struct {
  uint64_t buf_addr;   // line-aligned test lines
  uint64_t out_addr;   // per-probe results
  uint32_t mode;
  uint32_t iters;      // per hart (rounds for the atomic modes)
  uint32_t num_harts;
  uint32_t seed;
  uint32_t pool_lines; // amo_load: test-line pool, a power of two
  uint32_t pad;
} kernel_arg_t;

// raw/war: every hart walks its own fresh 64-byte lines, interleaved so
// concurrent harts hit different lines, banks and DRAM rows.
#define PROBE_LINE     64
#define PROBE_VARIANTS 12
#define PROBE_GROUP    4

static inline uint32_t probe_line(uint32_t h, uint32_t i, uint32_t num_harts) {
  return i * num_harts + h;
}

static inline uint32_t probe_variant(uint32_t h, uint32_t i) {
  return (i + (h / PROBE_GROUP) * 5) % PROBE_VARIANTS;
}

// war: the line's first word before the store, and the two stores.
static inline uint32_t war_init(uint32_t ln)   { return 0xA0000000u | ln; }
static inline uint32_t war_marker(uint32_t ln) { return 0xB0000000u | ln; }
static inline uint32_t war_side(uint32_t ln)   { return 0x50000000u | ln; }
// raw: the store whose value the following load must return.
static inline uint32_t raw_value(uint32_t ln)  { return 0xC0000000u | ln; }

// store_amo/store_lr: one warp; thread t stores to dcache word t of a fresh
// line after two loads of another word, then issues the atomic on its word.
#define STORE_AMO_BASE 0x5a000000u

// amo_store: slot i is the first word of its own 256-byte block, so it sits
// alone in its line at every cache level (L1 64 B, L2/L3 128 B lines).
#define SLOT_STRIDE_WORDS 64
static inline uint32_t slot_init(uint32_t i) { return 0x10000000u + i * 0x101u; }
static inline uint32_t slot_inc(uint32_t i)  { return 0x00000100u + i; }
static inline uint32_t slot_val(uint32_t i)  { return 0xA5000000u ^ (i * 0x01010101u); }
// Issue slots the store (atomic = 0) or the atomic waits after the barrier:
// rounds alternate in groups of 16 between delaying one side by 0..15.
static inline uint32_t slot_delay(uint32_t i, uint32_t atomic) {
  return (((i / 16) & 1) == atomic) ? (i % 16) : 0;
}

// amo_load: one warp of AMO_LOAD_LANES lanes; per lane {violations, first bad
// round, the atomic's old value and the load's value in that round}.
#define AMO_LOAD_LANES 4
#define AMO_LOAD_RES   4

// amo_storm: per hart and round one fresh line; per hart one resident line.
#define STORM_LINE 64

#endif
