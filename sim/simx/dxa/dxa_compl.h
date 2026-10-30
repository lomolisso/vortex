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

#include <deque>
#include <unordered_map>
#include <vector>
#include "types.h"

namespace vortex {

// A core's DXA completion unit. A copy is complete once every one of its
// writes has committed at the local-memory banks; its queue entry then
// releases a barrier event or, for an mbarrier copy, becomes a complete_tx of
// the box bytes on the mbarrier word, which the bank's atomic unit executes.
class DxaCompl : public SimObject<DxaCompl> {
public:
  // The queue depth: an intra-core multicast lands one copy per warp.
  static constexpr uint32_t DEPTH = (VX_CFG_NUM_WARPS < 2) ? 2 : VX_CFG_NUM_WARPS;

  // The local memory's committed DXA writes, in the cycle they commit.
  SimChannel<DxaWrCommit> WrIn;

  // Barrier events, to the core's barrier unit.
  SimEventLink<uint32_t> TxbarOut;

  // The entries free next cycle, to the local memory's DXA input, which holds
  // a copy's final beat while there is none.
  SimEventLink<uint32_t> FreeOut;

  // Raw barrier ids of copies the engine finished without a write.
  SimChannel<uint32_t> ReleaseIn;

#ifdef DXA_MBAR_ENABLE
  // complete_tx requests into the local memory, and their responses, dropped.
  SimChannel<MemReq> ReqOut;
  SimChannel<MemRsp> RspIn;

  // The core's CTA slots, for the retired-slot and bounds checks.
  SimChannel<CtaSlotEvt> SlotIn;
#endif

  DxaCompl(const SimContext& ctx, const char* name, uint32_t num_banks);
  ~DxaCompl();

protected:
  void on_reset();
  void on_tick();

private:
  struct Entry {
    DxaWrTag tag;
    uint32_t bar_id;
    uint64_t uuid;
    uint64_t push;   // the cycle it was queued
  };

  struct Xfer {
    uint32_t committed = 0;
    uint32_t writes = 0;   // known once the final write has committed
    uint32_t bar_id = 0;
  };

  bool entry_is_mbar(const Entry& e) const;
  void send_free();

#ifdef DXA_MBAR_ENABLE
  void check_live(uint32_t slot, uint32_t gen, uint64_t uuid, const char* what) const;
  void check_complete_tx(const Entry& e) const;
  void send_complete_tx(const Entry& e);

  std::vector<bool>     slot_valid_;
  std::vector<uint32_t> slot_gen_;
  uint32_t              slot_stride_;
  bool                  sent_;    // the head's complete_tx waits at the local memory's input
#endif

  std::deque<Entry> fifo_;
  std::unordered_map<uint32_t, Xfer> xfers_;   // copies with writes committed
  uint32_t landing_;     // copies whose final write has committed but not all the others
  uint32_t free_sent_;

  friend class SimObject<DxaCompl>;
};

} // namespace vortex
