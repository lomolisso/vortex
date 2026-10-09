#ifndef _COMMON_H_
#define _COMMON_H_

// Slots in the compare-and-swap ladder. The shared buffer is sized from the
// hart count, and one hart still gives 64 bytes, so 16 words always fit
// whatever the launch shape or the iteration count.
#define CAS_LADDER_SLOTS 16

// Slots in the local-memory compare-and-swap ladder (64 bytes of a group's
// local memory), and the marker a group leader folds into its published win
// count so the host can count groups. Wins never exceed the slot count, so
// the marker is unambiguous.
#define CAS_LMEM_SLOTS      16
#define CAS_LMEM_GROUP_MARK 0x10000u

// Reservations live in the last-level cache, one per line, so the LR/SC tests
// give each word they reserve a line of its own there.
#if VX_CFG_L3_ENABLED
#define LLC_LINE_SIZE VX_CFG_L3_LINE_SIZE
#elif VX_CFG_L2_ENABLED
#define LLC_LINE_SIZE VX_CFG_L2_LINE_SIZE
#else
#define LLC_LINE_SIZE VX_CFG_DCACHE_LINE_SIZE
#endif
#define LLC_LINE_WORDS (LLC_LINE_SIZE / 4)

// Cross-thread store-conditional tests: each thread of the warp owns the word
// at the start of its own LLC line of the shared buffer, seeded with
// LRSC_FOREIGN_SEED + tid, so a successful SC on one word breaks no other
// thread's.
#define LRSC_FOREIGN_SEED 0x5c000000u

// Threads below LRSC_FOREIGN_DIST store-conditional into the word of the
// thread LRSC_FOREIGN_DIST above them. When a warp is split into LSU packets,
// that thread rides the same LSU lane, so the lane cannot stand in for the
// thread. A warp that fits one packet aims half a warp away instead.
#if VX_CFG_NUM_LSU_LANES < VX_CFG_NUM_THREADS
#define LRSC_FOREIGN_DIST VX_CFG_NUM_LSU_LANES
#else
#define LRSC_FOREIGN_DIST (VX_CFG_NUM_THREADS / 2)
#endif

// Reservation stations are direct-mapped, so threads whose words share one
// displace each other and an owner's own SC may legally fail. Consecutive LLC
// lines or bank words interleave across banks, then stations, so the words
// stay apart while the warp has no more threads than banks x stations. Software
// sees AMO_RS_SIZE resolved without the cache request counts, which can
// overstate it, and not the LLC bank count; NUM_WARPS and one bank are floors.
constexpr uint32_t lrsc_bank_stations(uint32_t entries) {
  uint32_t n = 2;
  while (n < entries) n <<= 1;
  return n;
}
#define LRSC_RS_SIZE_FLOOR \
  ((VX_CFG_AMO_RS_SIZE < VX_CFG_NUM_WARPS) ? VX_CFG_AMO_RS_SIZE : VX_CFG_NUM_WARPS)
#define LRSC_LLC_STATIONS  lrsc_bank_stations(LRSC_RS_SIZE_FLOOR)
#define LRSC_LMEM_STATIONS (VX_CFG_LMEM_NUM_BANKS \
  * lrsc_bank_stations((LRSC_RS_SIZE_FLOOR < 4) ? LRSC_RS_SIZE_FLOOR : 4))

// Store-conditional by a peer hart (lrsc_peer). Each slot is the first word of
// its own LLC line of the shared buffer; the out slots hold one result word per
// LRSC_PEER_* index below.
#define LRSC_PEER_X       0  // the holder's reserved word
#define LRSC_PEER_Y       1  // the peer's own reservation and its control LR/SC
#define LRSC_PEER_HELD    2  // flag: the holder's LR is in
#define LRSC_PEER_TRIED   3  // flag: the peer's SC on X is in
#define LRSC_PEER_REHELD  4  // flag: the holder's live SC and second LR are in
#define LRSC_PEER_WROTE   5  // flag: the peer's swap and control are in
#define LRSC_PEER_A_OUT   6
#define LRSC_PEER_B_OUT   7
#define LRSC_PEER_LINES   8

#define LRSC_PEER_RAN     0  // out words, both roles
#define LRSC_PEER_TIMEOUT 1
#define LRSC_PEER_POLLS   2
#define LRSC_PEER_A_LR    3  // holder: value its first LR read
#define LRSC_PEER_A_SC    4  // holder: its last SC's result
#define LRSC_PEER_A_LIVE  5  // holder: result of its SC right after the peer's
#define LRSC_PEER_A_LR2   6  // holder: value its second LR read
#define LRSC_PEER_B_SC    3  // peer: result of its SC on the holder's word
#define LRSC_PEER_B_PREV  4  // peer: the word its swap replaced
#define LRSC_PEER_B_CTRL  5  // peer: result of its control SC
#define LRSC_PEER_B_SEEN  6  // peer: X as its plain load read it after its SC

// Mixed-lane tests: odd lanes of one access hit local memory, even lanes global
// memory. Thread t's local word is MIXED_LMEM_OWN + t, clear of the hot word 0
// that every lane hammers; the seeds tell a local word from a global one.
#define MIXED_LMEM_OWN     16
#define MIXED_GLOBAL_SEED  0x6d000000u
#define MIXED_LOCAL_SEED   0x1c000000u

#define LRSC_PEER_X_SEED        0x7e000001u
#define LRSC_PEER_Y_SEED        0x7e000002u
#define LRSC_PEER_A_SC_VALUE    0x7e0000a0u
#define LRSC_PEER_A_LIVE_VALUE  0x7e0000a1u
#define LRSC_PEER_B_SC_VALUE    0x7e0000b1u
#define LRSC_PEER_B_SWAP_VALUE  0x7e0000b2u

// Args shared between host (main.cpp) and the GPU kernel (kernel.cpp).
// Layout matches dogfood's, with `iters` added for the contention loops.
typedef struct {
  uint32_t testid;
  uint32_t num_harts;     // total harts hammering the shared word
  uint32_t iters;         // per-hart iteration count
  uint64_t shared_addr;   // single-word AMO target (4 or 8 bytes)
  uint64_t per_hart_addr; // per-hart scratch / observed-old buffer
  uint32_t peer_core;     // lrsc_peer: global core id of the peer hart
  uint32_t peer_warp;     // lrsc_peer: its warp (thread 0)
} kernel_arg_t;

#endif
