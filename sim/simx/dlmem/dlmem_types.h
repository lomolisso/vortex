// Copyright © 2019-2026
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

#include <cstdint>
#include <iostream>
#include <memory>
#include "types.h"

namespace vortex {

// Request network packet (VX_dlmem_pkg dlmem_req_pkt_t); the fabric routes it
// to `dst`, the target core.
struct DlmemReqPkt {
  uint32_t dst = 0;
  uint32_t src = 0;                    // requesting core, the response's dst
  MemOp    op = MemOp::LD;             // LD, ST or a read-modify-write atomic
  uint64_t lmem_addr = 0;              // byte address in the target's local memory
  std::shared_ptr<mem_block_t> data;   // store data or AMO rhs, block-positioned
  uint64_t byteen = 0;
  uint64_t amo_cmp = 0;
  uint32_t hart_id = 0;
  MemFlags flags;
  uint32_t tag = 0;                    // LD/AMO: source pending index; ST: source warp id
  uint64_t uuid = 0;                   // trace only; the RTL packet carries none

  friend std::ostream& operator<<(std::ostream& os, const DlmemReqPkt& p) {
    os << "src=" << p.src << ", dst=" << p.dst << ", op=" << p.op
       << ", addr=0x" << std::hex << p.lmem_addr << std::dec
       << ", tag=" << p.tag << " (#" << p.uuid << ")";
    return os;
  }
};

// Response network packet (dlmem_rsp_pkt_t); `dst` is the requesting core.
struct DlmemRspPkt {
  uint32_t dst = 0;
  bool     ack = false;                // store committed; data unused
  uint32_t tag = 0;
  std::shared_ptr<mem_block_t> data;   // the bank's response block for LD/AMO
  uint64_t uuid = 0;

  friend std::ostream& operator<<(std::ostream& os, const DlmemRspPkt& p) {
    os << "dst=" << p.dst << ", " << (p.ack ? "ack" : "data")
       << ", tag=" << p.tag << " (#" << p.uuid << ")";
    return os;
  }
};

} // namespace vortex
