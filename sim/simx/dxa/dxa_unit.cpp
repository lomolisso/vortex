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

#include "dxa_unit.h"
#include "core.h"
#include "constants.h"
#include "debug.h"
#ifdef DXA_MBAR_ENABLE
#include <iostream>
#endif

using namespace vortex;

static_assert(VX_CFG_NUM_THREADS >= 4, "DXA requires at least 4 threads per warp");

instr_trace_t* DxaUnit::process(instr_trace_t* trace) {
  // A warp wider than the SFU arrives as several packets, all dispatched even
  // when inactive, that carry the whole operand set; only the one holding
  // lane 3 issues the request.
  if (trace->pid >= 0 && trace->pid != int(3 / VX_CFG_NUM_SFU_LANES)) {
    return trace;
  }
  if (req_out_.full()) {
    return nullptr;
  }

  // 4-lane wgather encoding:
  //   Lane 0: rs1=smem_addr, rs2=coord2
  //   Lane 1: rs1=meta,      rs2=coord3
  //   Lane 2: rs1=coord0,    rs2=coord4
  //   Lane 3: rs1=coord1,    rs2=cta_mask
  auto& rs1 = trace->src_data[0];
  auto& rs2 = trace->src_data[1];

  uint64_t smem_addr = static_cast<uint64_t>(rs1.at(0).u);
  uint32_t meta      = rs1.at(1).u;
  uint32_t coords[5] = {
    static_cast<uint32_t>(rs1.at(2).u),
    static_cast<uint32_t>(rs1.at(3).u),
    static_cast<uint32_t>(rs2.at(0).u),
    static_cast<uint32_t>(rs2.at(1).u),
    static_cast<uint32_t>(rs2.at(2).u),
  };
  uint32_t cta_mask  = rs2.at(3).u;
  uint32_t desc_slot = meta & 0x0fu;
#ifdef DXA_MBAR_ENABLE
  // meta[30] selects the completion; meta[29:4] is its barrier or mbarrier.
  const bool mode    = (meta >> 30) & 1u;
  uint32_t raw_bar   = mode ? 0u : ((meta >> 4) & 0x03ffffffu);
  if (mode && __builtin_popcount(cta_mask) > 1) {
    std::cerr << SimPlatform::instance().cycles() << ": *** core" << core_->id()
              << " DXA multicast with an mbarrier is not supported: wid=" << trace->wid
              << " cta_mask=0x" << std::hex << cta_mask << std::dec << std::endl;
    std::abort();
  }
#else
  uint32_t raw_bar   = (meta >> 4) & 0x07ffffffu;
#endif

  DxaReq req;
  req.core      = core_;
  req.uuid      = trace->uuid;
  req.wid       = trace->wid;
  req.desc_slot = desc_slot;
  // Keep raw bar_id; multicast offset arithmetic relies on encoded form
  // (cta_no in low 8 bits → bar_id + cta_idx targets next CTA's same bar).
  // Release call site decodes via bar_decode_id().
  req.bar_id    = raw_bar;
  req.cta_mask  = cta_mask;
  req.smem_addr = smem_addr;
  for (int i = 0; i < 5; ++i) req.coords[i] = coords[i];
#ifdef DXA_MBAR_ENABLE
  req.mode      = mode;
  req.mbar_off  = mode ? ((meta >> 4) & ((1u << (VX_CFG_LMEM_LOG_SIZE - 3)) - 1u)) : 0u;
  req.slot      = warp_slot_.at(trace->wid);
  req.gen       = slot_gen_.at(req.slot);
#endif

  // Barrier pre-registration is the kernel's responsibility via
  // vx_barrier_expect_tx(). The DXA pipeline only emits release events on
  // completion; pre-registration happens explicitly per-CTA so multicast
  // destinations correctly wait.

  req_out_.send(req);
  DT(4, "dxa-unit submit: core=" << core_->id() << ", wid=" << trace->wid
     << ", slot=" << desc_slot << ", bar=" << raw_bar
     << ", cta_mask=0x" << std::hex << cta_mask << std::dec);
  return trace;
}

#ifdef DXA_MBAR_ENABLE
void DxaUnit::slot_evt(const CtaSlotEvt& e) {
  switch (e.kind) {
  case CtaSlotEvt::ADMIT: slot_gen_.at(e.slot) = e.gen; break;
  case CtaSlotEvt::WARP:  warp_slot_.at(e.wid) = e.slot; break;
  default: break;
  }
}
#endif
