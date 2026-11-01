#ifndef _DXA_COPY_MW_COMMON_H_
#define _DXA_COPY_MW_COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// Intra-core multicast copy test arguments.
typedef struct {
  uint32_t tile_rows;
  uint32_t tile_cols;
  uint32_t src_row_stride;  // elements between source rows (descriptor stride)
  uint32_t num_recv;        // multicast group size (= CTAs sharing the tile)
  uint64_t src_addr;
  uint64_t dst_addr;        // tightly packed: num_recv × tile_rows × tile_cols
} kernel_arg_t;

// The checked modes (-m, -M, -x): clusters of k blocks, block b = rank b % k
// of cluster b / k; each block's whole local memory is dumped for the host.
#define MC_LEGACY   0   // a barrier multicast into the slots issuer + bit (DEFAULT clusters)
#define MC_CLUSTER  1   // an mbarrier multicast into the ranks of the mask

#define NEG_NONE    0
#define NEG_RANK    1   // a mask bit at or past the cluster size
#define NEG_SPREAD  2   // a barrier multicast from a block of a SPREAD cluster
#define NEG_BOX     3   // the box starts inside the block and runs past its end
#define NEG_RETIRED 4   // a receiver has exited when the copy reaches it
#define NUM_NEGS    5

#define MAX_DIMS    5

typedef struct {
  uint64_t err_addr;     // ERR_WORDS uint32 per thread
  uint64_t rec_addr;     // REC_WORDS uint32 per block
  uint64_t dump_addr;    // lmem_bytes per block: its local memory at the end
  uint32_t mode;
  uint32_t neg;
  uint32_t k;
  uint32_t mask;
  uint32_t issuer;       // the rank that issues
  uint32_t helper;       // issue through vortex::dxa_multicast_cluster_Nd
  uint32_t ndim;
  uint32_t copies;       // multicasts the issuer sends, each into the next tile slot
  uint32_t ntiles;       // source tiles
  uint32_t box;          // bytes per copy
  uint32_t tstride;      // bytes between the tile slots in local memory
  uint32_t lmem_bytes;
  uint32_t linger;       // negatives: cycles each block stays alive around its issue
  uint32_t neg_off;      // -x box: the block offset of the destination
  uint32_t tiles[MAX_DIMS];  // tile extents, elements
  uint32_t grids[MAX_DIMS];  // source tiles per dimension
} mc_arg_t;

// Byte offsets in the block's local memory: the mbarrier (cluster mode),
// then the tile slots, each followed by at least a 64-byte gap. Everything
// else starts as canary bytes, so a stray write shows.
#define LM_MBAR     0
#define LM_TILE     64
#define DESC_SLOT   0

#define REC_BLOCK   0   // blockIdx.x
#define REC_CORE    1
#define REC_SLOT    2   // the dispatcher slot
#define REC_RANK    3
#define REC_RAN     4   // times the block ran
#define REC_WORDS   5

#define ERR_DONE    0   // 1 once the thread finished
#define ERR_WAIT    1   // the copies did not complete within WAIT_LIMIT waits
#define ERR_TRIES   2   // waits of the thread
#define ERR_WORDS   4

#define WAIT_LIMIT  64

static inline uint32_t mix32(uint32_t a, uint32_t b) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

// Byte i of the source array.
static inline uint8_t src_byte(uint32_t i) {
  return (uint8_t)(mix32(0xD3A0u, i) >> 8);
}

static inline uint32_t canary_word(uint32_t b, uint32_t w) {
  return mix32(0xCB0000u + b, w);
}

static inline uint8_t canary_byte(uint32_t b, uint32_t i) {
  return (uint8_t)(canary_word(b, i >> 2) >> (8 * (i & 3)));
}

// The source tile of copy i of cluster g.
static inline uint32_t src_tile(uint32_t g, uint32_t i, uint32_t copies, uint32_t ntiles) {
  return ((g * copies + i) * 3 + 1) % ntiles;
}

#endif
