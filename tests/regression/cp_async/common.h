#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define MODE_ZFILL  0   // every copy size, src-size, ignore-src and offset pair, byte-exact
#define MODE_GROUPS 1   // 4 groups per thread, wait_group 2, wait_all
#define MODE_BAR    2   // the H100 probe's cp.async.mbarrier.arrive modes 0-3
#define MODE_STREAM 3   // rows streamed into a double buffer, completed on mbarriers
#define MODE_PIPE   4   // an RMSNorm-style double buffer with pipeline_wait_prior<1>
#define MODE_ORPHAN 5   // blocks exit with copies and an arrive in flight; the next block checks its memory
#define MODE_MIX    6   // copies and global loads in flight together

#define NEG_NONE       0
#define NEG_REMOTE_DST 1   // destination in a peer block, through the window
#define NEG_GLOBAL_DST 2   // destination in global memory
#define NEG_MISALIGNED 3   // an 8-byte copy to a 4-byte aligned destination
#define NEG_SRCSIZE    4   // src-size 17 on a 16-byte copy
#define NEG_REMOTE_MBAR 5  // cp.async.mbarrier.arrive.noinc on a peer's object
#define NEG_LOCAL_SRC  6   // a source in local memory
#define NEG_ENCODING   7   // the reserved funct3 7

typedef struct {
  uint64_t src_addr;     // global source bytes
  uint64_t out_addr;     // mode output
  uint64_t aux_addr;     // zfill: cases; pipe: weights
  uint64_t info_addr;    // uint64[INFO_WORDS]
  uint32_t mode;
  uint32_t neg;
  uint32_t count;        // zfill: cases; stream, pipe: rows
  uint32_t parity;       // stream: no prefetch, one try_wait per row
  uint32_t row_elems;    // pipe: valid int32 elements per row (the rest zero-filled)
  uint32_t spin_limit;
} cpa_arg_t;

// zfill: one case per 64-byte local slot, prefilled with ZF_FILL.
#define ZF_SLOT  64
#define ZF_FILL  0xAA
#define ZF_CASE(size, ss, ign, soff, doff) \
  ((uint32_t)(size) | ((uint32_t)(ss) << 5) | ((uint32_t)(ign) << 10) | ((uint32_t)(soff) << 11) | ((uint32_t)(doff) << 17))
#define ZF_SIZE(c) ((c) & 31)
#define ZF_SS(c)   (((c) >> 5) & 31)
#define ZF_IGN(c)  (((c) >> 10) & 1)
#define ZF_SOFF(c) (((c) >> 11) & 63)
#define ZF_DOFF(c) (((c) >> 17) & 63)
#define ZF_CG      (1u << 23)    // the .cg spelling
#define ZF_MAX     128

// groups: per thread 4 groups of one 16-byte copy, GR_STRIDE bytes apart in
// the source; out: per thread the groups seen after wait_group 2 (bits 0-3),
// then after wait_all (bits 4-7).
#define GR_GROUPS 4
#define GR_STRIDE 4096

// bar: the four H100 modes, one after another; out per mode.
#define BAR_MODES   4
#define BAR_REC     8     // single-thread modes: the recorded values
#define BAR_OUT(m)  ((m) * (2 + BAR_REC))
// out[BAR_OUT(m)] = bad data elements, [+1] = the object's final word, [+2..] records
// mode 1: out[BAR_PC + t] = thread t's pending count
#define BAR_PC      (BAR_MODES * (2 + BAR_REC))
#define BAR_SRC_SINGLE 1000   // the single-thread modes copy 16-byte element 1000

// stream and pipe: rows of 16 bytes per thread.
#define INFO_SPINS  0   // try_waits that returned false
#define INFO_OPEN   1   // stream -p: rows whose phase was still open after the wait
#define INFO_WORD0  2   // stream: final words of FULL[0..1]
#define INFO_DONE   4
#define INFO_WORDS  8

// orphan: per thread OR_STORES global stores, one line each, then OR_COPIES
// copies over its own 16 bytes, each from its own line (OR_STRIDE apart),
// whose reads queue behind the stores and land late; out per block: [0] the
// words not holding the block's marks after the wait, [1] its mbarrier word
// then. count: the iterations the block waits before it checks.
#define OR_STORES   32
#define OR_COPIES   4
#define OR_STRIDE   4160
#define OR_OUT      2

// mix: per row a 16-byte copy and MIX_LOADS loads per thread.
#define MIX_LOADS   4

#endif
