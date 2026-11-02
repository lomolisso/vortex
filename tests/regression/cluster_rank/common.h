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

// ids_k (-x ids): one record of IDS_WORDS words per block, at its linear index.
typedef struct {
  uint64_t rec_addr;
} ids_arg_t;

#define IDS_BLOCK        0   // blockIdx x, y, z
#define IDS_CTAID        3   // %cluster_ctaid x, y, z
#define IDS_NCTAID       6   // %cluster_nctaid x, y, z
#define IDS_CLUSTERID    9   // %clusterid x, y, z
#define IDS_NCLUSTERID   12  // %nclusterid x, y, z
#define IDS_EXPLICIT     15  // %is_explicit_cluster
#define IDS_DYN_SMEM     16  // %dynamic_smem_size
#define IDS_TOTAL_SMEM   17  // %total_smem_size
#define IDS_AGGR_SMEM    18  // %aggr_smem_size
#define IDS_RANK         19  // %cluster_ctarank
#define IDS_CVTA         20  // cvta.shared::cluster of the block's object
#define IDS_RANK_LOCAL   21  // getctarank of the object's generic pointer
#define IDS_RANK_WINDOW  22  // getctarank of its window pointer in the next rank
#define IDS_RANK_U32     23  // getctarank.u32 of mapa.u32 into the next rank
#define IDS_MAPA         24  // mapa.u32 of the object into the next rank
#define IDS_FLAGS        25  // IDS_F_* that held
#define IDS_PEER_VALUE   26  // the next rank's object, read through cvta of mapa.u32
#define IDS_CG_THREADS   27  // cluster_group::num_threads()
#define IDS_CG_TRANK     28  // cluster_group::thread_rank() of thread 0
#define IDS_CG_DIMTX     29  // cluster_group::dim_threads().x
#define IDS_CG_QRANK     30  // cluster_group::query_shared_rank() of the window pointer
#define IDS_RESERVED     31  // %reserved_smem_offset
#define IDS_WORDS        32

#define IDS_F_SPACEP_LOCAL   (1u << 0)  // isspacep.shared::cluster of the object
#define IDS_F_SPACEP_WINDOW  (1u << 1)  // isspacep.shared::cluster of the window pointer
#define IDS_F_SPACEP_GLOBAL  (1u << 2)  // isspacep.shared::cluster of a global pointer (must not hold)
#define IDS_F_CTA_LOCAL      (1u << 3)  // isspacep.shared::cta of the object
#define IDS_F_CTA_WINDOW     (1u << 4)  // isspacep.shared::cta of the window pointer (must not hold)
#define IDS_F_GENERIC        (1u << 5)  // cvta of mapa.u32 == map_shared_rank()
#define IDS_F_MAPA_WINDOW    (1u << 6)  // mapa of a window pointer == map_shared_rank()
#define IDS_F_CG_DIMS        (1u << 7)  // dim_blocks() and block_index() match the registers
#define IDS_F_TIMER          (1u << 8)  // %globaltimer advanced
#define IDS_F_VALID          (1u << 9)  // cluster_group::is_valid()
#define IDS_F_ALL            0x3EBu     // every flag but the two that must not hold

// The object every block of a cluster holds at the same offset, and the tag
// each block stores there.
#define IDS_OBJ_OFFSET 40
static inline uint64_t ids_tag(uint32_t block) { return 0x1D5000000000ull | block; }

// Offset of the object a CTA maps into its peers, distinct per CTA.
static inline uint32_t map_offset(uint32_t block, uint32_t lmem_bytes) {
  return ((block * 40u) + 8u) % lmem_bytes & ~7u;
}

// Unique per (CTA, word), so a word left by an earlier CTA in the slot never matches.
static inline uint64_t lmem_pattern(uint32_t block, uint32_t index) {
  return ((uint64_t)(block + 1) << 40) ^ ((uint64_t)index * 0x9E3779B97F4A7C15ull);
}

#endif
