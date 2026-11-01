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
#include <deque>
#include <memory>
#include <vector>
#include "types.h"
#include "dlmem_types.h"

namespace vortex {

// VX_dlmem_ingress: serves remote requests on the core's local memory. A
// request is accepted only into a free slot, which holds its response space
// until the response enters the response network, so the local memory never
// waits on the network.
//
// NetReqIn is the request network's ejection buffer: this object must tick
// after the request switch so its pops free a slot the cycle after, as the
// switch's registered ready. NetRspOut feeds the response switch, which must
// tick after this object for the same reason.
class DlmemIngress : public SimObject<DlmemIngress> {
public:
  using Ptr = std::shared_ptr<DlmemIngress>;

  struct Config {
    uint32_t num_slots;
    uint64_t lmem_base;   // MemReq.addr of local-memory byte 0
  };

  SimChannel<DlmemReqPkt> NetReqIn;
  SimChannel<MemReq>      LmemReqOut;   // the OUT_BUF=3 request buffer
  SimChannel<MemRsp>      LmemRspIn;    // never refused
  SimChannel<DlmemRspPkt> NetRspOut;
#ifdef DXA_MBAR_ENABLE
  // A peer's DXA row flits: a DATA word is a DMA-class write on LmemDmaOut,
  // which has no buffer of its own, so the next flit waits until the local
  // memory has taken it toward its bank; the COMPL then becomes a posted
  // complete_tx on LmemReqOut. Must tick before the local memory.
  SimChannel<MemReq>     LmemDmaOut;
  // The core's CTA slot admissions and frees, for the flits' occupancy check.
  SimChannel<CtaSlotEvt> SlotIn;
#endif

  DlmemIngress(const SimContext& ctx, const char* name, const Config& config);

  bool empty() const;

  uint32_t busy_slots() const;

protected:
  void on_reset();
  void on_tick();

private:
  struct Slot {
    bool busy = false;
    uint32_t src = 0;
    uint32_t tag = 0;
    bool ack = false;
    std::shared_ptr<mem_block_t> data;
    uint64_t uuid = 0;
#ifdef DXA_MBAR_ENABLE
    bool posted = false;   // a COMPL's complete_tx: its response is dropped
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
    // An st.async / red.async: complete_tx of `bytes` on the mbarrier at
    // mbar_addr follows its access's commit, before its ack.
    bool     mbar = false;
    uint64_t mbar_addr = 0;
    uint32_t bytes = 0;
    bool     cx_sent = false;
#else
#line 77
#endif
  };

  Config config_;
  std::vector<Slot> slots_;
  std::deque<uint32_t> done_;   // completion FIFO of slot indices
#ifdef DXA_MBAR_ENABLE
  // Takes the head DXA flit if it can; false leaves it at the head.
  bool accept_dxa(const DlmemReqPkt& pkt);
  void check_slot(const DlmemReqPkt& pkt) const;

  std::vector<bool> slot_valid_;
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
  bool cx_valid() const;
  MemReq complete_tx(uint32_t s) const;
#else
#line 89
#endif

  friend class SimObject<DlmemIngress>;
};

} // namespace vortex
