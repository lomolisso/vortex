#ifndef _DXA_TMA_MODES_COMMON_H_
#define _DXA_TMA_MODES_COMMON_H_

#include <stdint.h>

// TMA copy-mode test arguments. One CTA; the DXA copies one tile into SMEM
// under a mode-specific descriptor (element strides / 3D K-major / swizzle),
// then the kernel dumps the RAW SMEM tile image to GMEM and the host
// compares it against a reference model of the expected destination layout.
typedef struct {
  uint32_t ndim;        // descriptor rank (2 or 3)
  uint32_t tile_words;  // tile size in 32-bit words
  uint32_t block_size;  // threads per CTA (dump stride)
  uint64_t dst_addr;    // tile_words * 4 bytes
} kernel_arg_t;

#endif
