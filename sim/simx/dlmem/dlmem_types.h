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

// VX_dlmem_pkg's endpoint sizes.
constexpr uint32_t DLMEM_INGRESS_SLOTS = 4;
constexpr uint32_t DLMEM_WT_MAX = VX_CFG_DLMEM_STORE_CREDITS + VX_CFG_DLMEM_PENDING_SIZE
                                + (VX_CFG_LSU_QUEUE_IN_SIZE + 3) * VX_CFG_NUM_LSU_LANES;

constexpr uint32_t dlmem_clog2(uint64_t x) {
  uint32_t n = 0;
  while ((uint64_t(1) << n) < x) {
    ++n;
  }
  return n;
}

// The access tracker's largest count, 2^DLMEM_WT_W - 1.
constexpr uint32_t DLMEM_WT_LIMIT = (1u << dlmem_clog2(uint64_t(DLMEM_WT_MAX) + 1)) - 1;

// VX_dlmem_pkg's bound on one warp's local plain-store lanes between the
// translation and their bank write, with the RTL crossbar's arbiter groups
// (VX_stream_arb's MAX_FANOUT is 8).
constexpr uint32_t DLMEM_LST_XB_IN = VX_CFG_NUM_LSU_BLOCKS * VX_CFG_NUM_LSU_LANES + 1;
constexpr uint32_t DLMEM_LST_XB_G1 = (DLMEM_LST_XB_IN > 12) ? (DLMEM_LST_XB_IN + 7) / 8 : 0;
constexpr uint32_t DLMEM_LST_XB_G2 = (DLMEM_LST_XB_G1 > 12) ? (DLMEM_LST_XB_G1 + 7) / 8 : 0;
static_assert(DLMEM_LST_XB_G2 <= 12, "the local-store bound covers at most three crossbar arbiter levels");
constexpr uint32_t DLMEM_LST_MAX = (VX_CFG_LSU_QUEUE_IN_SIZE + 3) * VX_CFG_NUM_LSU_LANES
                                 + (3 + 2 * (DLMEM_LST_XB_G1 + DLMEM_LST_XB_G2)) * VX_CFG_LMEM_NUM_BANKS;

// The local-store tracker's largest count, 2^DLMEM_LST_W - 1.
constexpr uint32_t DLMEM_LST_LIMIT = (1u << dlmem_clog2(uint64_t(DLMEM_LST_MAX) + 1)) - 1;

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
