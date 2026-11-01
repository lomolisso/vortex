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

#include "types.h"
#include "instr_trace.h"

namespace vortex {

class Core;

// DxaReq — per-SFU dispatch packet. The SFU owns the outbound SimChannel;
// DxaUnit is a plain helper sub-class of SfuUnit that decodes lanes and
// pushes onto that channel.
struct DxaReq {
  Core*    core;          // routes barrier-release back at completion
  uint64_t uuid;
  uint32_t wid;
  uint32_t desc_slot;     // descriptor table index
  uint32_t bar_id;        // RAW (pre bar_decode_id) — kept raw so that
                          // multicast offset arithmetic (bar_id + cta_idx)
                          // increments cta_no in the encoded low byte.
                          // Decoded only at the release call site.
  uint32_t cta_mask;      // multicast warp mask (>1 bit ⇒ multicast)
  uint64_t smem_addr;
  uint32_t coords[5];
#ifdef DXA_MBAR_ENABLE
  bool     mode;          // meta[30]: complete_tx on an mbarrier
  uint32_t mbar_off;      // the mbarrier's doubleword offset in the destination's local memory
  uint32_t slot;          // the destination block's CTA slot and its generation
  uint32_t gen;
  // A destination block on another core of the cluster: the rows travel over
  // the DLMEM network, and smem_addr is in that core's local memory.
  bool     remote;
  uint32_t src_core;      // the issuing and the destination core, cluster-relative
  uint32_t dst_core;
  bool     win;           // a DLMEM window destination
  uint32_t win_room;      // and its bytes up to its block's end
  // A cluster multicast (meta[30] with a rank mask): one copy into each masked
  // rank's block, in increasing rank, each at the issuer's block-relative
  // offset and completing on the mbarrier at the issuer's mbarrier offset.
  struct Rcv {
    uint32_t core;        // cluster-relative
    uint32_t slot;
    uint32_t gen;
    uint64_t smem_addr;
    uint32_t mbar_off;
    bool     remote;
  };
  bool     cluster;
  uint32_t num_rcv;
  std::array<Rcv, VX_CFG_DLMEM_MAX_CLUSTER> rcvs;
#else
#line 43
#endif
};

// DXA sub-unit of the SFU. Plain (non-SimObject) class owned by SfuUnit;
// lane-decodes a DxaType trace and forwards it onto the SFU's outbound
// DxaReq channel.
class DxaUnit {
public:
  // The outbound channel lives on SfuUnit (which is the SimObject).
  DxaUnit(Core* core, SimChannel<DxaReq>& req_out)
    : core_(core), req_out_(req_out) {}

  // Decode lanes 0..3 from the trace's source operands and try to push a
  // DxaReq on the SFU's outbound channel. Returns the trace on success
  // (caller falls through to writeback) or nullptr on backpressure
  // (caller retries next cycle without side effects).
  instr_trace_t* process(instr_trace_t* trace);

#ifdef DXA_MBAR_ENABLE
  // Mirrors the dispatcher's slot generations and the warps' slots.
  void slot_evt(const CtaSlotEvt& e);
#endif

private:
  Core*               core_;
  SimChannel<DxaReq>& req_out_;
#ifdef DXA_MBAR_ENABLE
  // Resolves a DLMEM window destination and mbarrier through the issuing
  // block's peer row, as the RTL reads it from the core's peer table.
  void translate(const instr_trace_t* trace, uint32_t meta, uint32_t cta_mask, DxaReq& req) const;
  void cluster_receivers(const instr_trace_t* trace, uint32_t meta, uint32_t cta_mask, DxaReq& req) const;

  [[noreturn]] void issue_abort(const instr_trace_t* trace, const char* msg) const;

  std::vector<uint32_t> warp_slot_ = std::vector<uint32_t>(VX_CFG_NUM_WARPS, 0);
  std::vector<uint32_t> slot_gen_  = std::vector<uint32_t>(VX_CFG_NUM_WARPS, 0);
  std::vector<dlmem_ref::Row> slot_row_ = std::vector<dlmem_ref::Row>(VX_CFG_NUM_WARPS);
  uint32_t stride_ = 0;
#else
#line 72
#endif
};

} // namespace vortex
