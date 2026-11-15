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

// C++20 side of DramSim: included only by TUs of libvxdram.a, never by a C++17 simulator TU.

#include <cstdint>

#include "ramulator/base/type.h"

namespace Ramulator {
class IMemorySystem;
}

namespace vortex::dram {

// One DramSim request becomes k 32-B Ramulator transactions; Request::ingress_id carries
// seq << 9 | head << 8 | write << 7 | k. The head is sent first; seq ties tails to their head.
struct Visit {
  uint32_t seq;
  bool head;
  bool write;
  uint32_t k;
};

constexpr uint32_t VISIT_MAX_K = 4;
constexpr uint32_t VISIT_SEQ_BITS = 22;
constexpr uint32_t VISIT_SEQ_MASK = (1u << VISIT_SEQ_BITS) - 1;

inline int visit_encode(const Visit& v) {
  return static_cast<int>(((v.seq & VISIT_SEQ_MASK) << 9) | (uint32_t(v.head) << 8) | (uint32_t(v.write) << 7) | (v.k & 0x7f));
}

inline Visit visit_decode(int ingress_id) {
  auto x = static_cast<uint32_t>(ingress_id);
  return Visit{(x >> 9) & VISIT_SEQ_MASK, ((x >> 8) & 1) != 0, ((x >> 7) & 1) != 0, x & 0x7f};
}

// Called for every command a VxHBM3 controller issues; addr_vec is in Ramulator's level order
// (Channel, PseudoChannel, Sid, BankGroup, Bank, Row, Column), column in transactions.
using vx_dram_issue_hook_t = void (*)(void* ctx, int channel, const char* command,
                                      const Ramulator::AddrVec_t& addr_vec, int ingress_id, int64_t clk);

// Hooks are per memory system: a process holds one per partition, each with its own observer.
void vx_dram_set_issue_hook(Ramulator::IMemorySystem* ms, vx_dram_issue_hook_t hook, void* ctx);

struct CtrlStats {
  uint64_t dir_switches;
  uint64_t max_rd_heads;  // requests waiting for their visit (the controller queue)
  uint64_t max_wr_heads;
};

// nullptr when that channel's controller is not a VxHBM3 (the legacy HBM12 model).
const CtrlStats* vx_dram_ctrl_stats(Ramulator::IMemorySystem* ms, int channel);

// Drops the registry entries of a memory system; call after deleting it.
void vx_dram_release(Ramulator::IMemorySystem* ms);

}  // namespace vortex::dram
