// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mempool.h>
#include "types.h"
#include "instr.h"
#ifdef VX_CFG_H100_FE_ENABLE
#include <array>
#endif

namespace vortex {

// Stateless ISA decoder.
class Decoder : public SimObject<Decoder> {
public:
  Decoder(const SimContext& ctx, const char* name, PoolAllocator<Instr, 64>& instr_pool);
  ~Decoder();

  // Decode an instruction word into an Instr. `code` carries either a
  // 32-bit RV32I encoding (low2 == 0b11) or a 16-bit RVC in the low half
  // (low2 != 0b11); the decoder detects RVC, calls rvc_decompress, and
  // stamps IntrBrArgs.is_rvc accordingly.
  Instr::Ptr decode(uint32_t code, uint64_t uuid);

#ifdef VX_CFG_H100_FE_ENABLE
  // Called for every instruction of warp `wid` before decode(): true when `code` is a UPFX,
  // which is held and applied to the warp's next instruction (it issues nothing itself).
  bool fe_prefix(uint32_t wid, uint32_t code);

  // The H100 front-end encodings, and the held UPFX of the current warp; nullptr otherwise.
  Instr::Ptr decode_fe(uint32_t code, uint64_t uuid);
#endif

private:
  PoolAllocator<Instr, 64>& instr_pool_;
#ifdef VX_CFG_H100_FE_ENABLE
  struct Prefix {
    bool     valid = false;
    uint32_t slot  = 0;
    uint32_t ur    = 0;
  };
  std::array<Prefix, VX_CFG_NUM_WARPS> prefix_{};
  uint32_t cur_wid_ = 0;
#endif
};

}

// Decoder::decode() starts with this hook; it expands to nothing with the switch off.
#ifdef VX_CFG_H100_FE_ENABLE
#define VX_FE_DECODE_HOOK if (auto fe_instr = this->decode_fe(code, uuid)) return fe_instr;
#else
#define VX_FE_DECODE_HOOK
#endif
