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

#include "types.h"

#ifdef DXA_MBAR_ENABLE

#include <array>
#include "dxa/dxa_bg.h"

namespace vortex {

// The bulk async-groups of one core, per warp. Each group slot, indexed by
// its sequence number, counts the operations whose local-memory reads and
// whose global writes are still pending; the open group is the slot after the
// committed ones. A wait_group releases its warp once every committed group
// but the N newest has nothing pending of the kind it waits for.
class BgUnit : public SimObject<BgUnit> {
public:
  using Ptr = std::shared_ptr<BgUnit>;

  static constexpr uint32_t SEQ_MASK = DXA_BG_GROUPS - 1;
  static_assert((DXA_BG_GROUPS & SEQ_MASK) == 0, "the bulk groups fill a sequence space");

  SimChannel<DxaBgEvt>   EvtIn;      // from the DXA unit, in its send cycle
  SimChannel<DxaBgCompl> ComplIn;    // from the socket's engine, always accepted
  SimChannel<DxaBgState> StateOut;   // a warp's registered state, when it changes
  SimChannel<uint32_t>   UnlockOut;  // a warp whose wait is satisfied

  BgUnit(const SimContext& ctx, const char* name);

  // On the registered counts, as the exit hold and the core's busy read them:
  // some group of the warp (the open one included) has a source read
  // pending; some group of any warp has a write not yet performed.
  bool rd_pending(uint32_t wid) const;
  bool wr_pending() const;

protected:
  void on_reset();
  void on_tick();

private:
  struct Group {
    uint32_t rd = 0;  // operations whose source reads are pending
    uint32_t wr = 0;  // operations whose writes are not performed
  };

  struct Warp {
    std::array<Group, DXA_BG_GROUPS> groups;
    uint32_t seq     = 0;  // the open group
    uint32_t ncommit = 0;  // committed groups not yet retired
    bool     waiting = false;
    bool     wait_rd = false;
    uint32_t wait_n  = 0;
    bool     ready   = false;
  };

  static bool satisfied(const Warp& w, bool rd, uint32_t n);
  static DxaBgState state(uint32_t wid, const Warp& w);
  void apply_evt(const DxaBgEvt& e);
  void apply_compl(const DxaBgCompl& c);

  std::vector<Warp> warps_;
  std::vector<DxaBgState> sent_;

  friend class SimObject<BgUnit>;
};

} // namespace vortex

#endif
