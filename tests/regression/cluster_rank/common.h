#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// rank_k: one record of REC_WORDS words per CTA, at its linear block index.
typedef struct {
  uint64_t rec_addr;
  uint64_t map_addr;    // 0, or num_blocks uint64 addresses per CTA (the -m check)
  uint32_t num_blocks;  // K, the launch's cluster size
  uint32_t lmem_bytes;  // the launch's per-CTA local memory
} rank_arg_t;

#define REC_BLOCK_X      0
#define REC_BLOCK_Y      1
#define REC_BLOCK_Z      2
#define REC_CTA_ID       3  // VX_CSR_CTA_ID: the dispatcher slot
#define REC_RANK         4  // vx_cluster_block_rank()
#define REC_GROUP_RANK   5  // cluster_group::block_rank()
#define REC_SPAWN_RANK   6  // vx_spawn2 get_cluster_rank()
#define REC_NUM_BLOCKS   7  // vx_cluster_num_blocks()
#define REC_WORDS        8

// lmem_k: per thread {errors, first bad index}; per CTA its local memory base.
typedef struct {
  uint64_t err_addr;
  uint64_t base_addr;
  uint32_t lmem_bytes;
} lmem_arg_t;

// Offset of the object a CTA maps into its peers, distinct per CTA.
static inline uint32_t map_offset(uint32_t block, uint32_t lmem_bytes) {
  return ((block * 40u) + 8u) % lmem_bytes & ~7u;
}

// Unique per (CTA, word), so a word left by an earlier CTA in the slot never matches.
static inline uint64_t lmem_pattern(uint32_t block, uint32_t index) {
  return ((uint64_t)(block + 1) << 40) ^ ((uint64_t)index * 0x9E3779B97F4A7C15ull);
}

#endif
