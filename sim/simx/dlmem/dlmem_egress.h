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
#include <memory>
#include <vector>
#include "types.h"
#include "dlmem_types.h"

namespace vortex {

// VX_dlmem_egress: serialises the remote lanes of the LSU blocks into request
// packets, one per cycle, and returns their responses. Loads and atomics hold
// a pending entry from issue to LSU delivery, stores a credit until their ack,
// so the response sink never refuses a packet.
//
// NetReqOut feeds the request switch, which must tick after this object so a
// slot it frees is seen here the cycle after, as the buffer's registered
// ready. The response sink is always ready, so its order is free.
class DlmemEgress : public SimObject<DlmemEgress> {
public:
  using Ptr = std::shared_ptr<DlmemEgress>;

  struct Config {
    uint32_t core_id;        // index inside the cluster
    uint32_t ncc;            // cores of the cluster
    uint32_t lmem_log;       // log2 of a core's local memory size
    uint64_t dlmem_base;
    uint32_t num_blocks;
    uint32_t num_lanes;
    uint32_t store_credits;
    uint32_t pending_size;
    bool     lrsc_assert;    // abort on a remote LR/SC instead of the defined fallback
  };

  std::vector<SimChannel<LsuReq>> ReqIn;    // remote lanes only; popped after the last lane
  std::vector<SimChannel<LsuRsp>> RspOut;   // the OUT_BUF=3 response buffers
  SimChannel<DlmemReqPkt> NetReqOut;
  SimChannel<DlmemRspPkt> NetRspIn;         // never refused
  SimChannel<uint32_t>    AckOut;           // warp id of each store ack

  DlmemEgress(const SimContext& ctx, const char* name, const Config& config);

  bool empty() const;

  uint32_t credits() const {
    return credits_;
  }

  uint32_t pending() const;

protected:
  void on_reset();
  void on_tick();

private:
  struct Entry {
    bool valid = false;
    bool data_valid = false;
    uint32_t block = 0;
    uint32_t lane = 0;
    uint64_t lsu_tag = 0;
    uint32_t cid = 0;
    uint64_t uuid = 0;
    std::shared_ptr<mem_block_t> data;
  };

  void issue();
  void deliver(std::vector<uint32_t>& freed);
  void sink();

  Config config_;
  std::vector<Entry> entries_;
  uint32_t credits_;
  bool locked_;
  uint32_t lock_block_;
  BitVector<> rem_mask_;
  uint32_t rr_last_;

  friend class SimObject<DlmemEgress>;
};

} // namespace vortex
