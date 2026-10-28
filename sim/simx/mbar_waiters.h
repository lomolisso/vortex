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

#ifdef VX_CFG_EXT_MBAR_ENABLE

#include "types.h"
#include "local_mem.h"

namespace vortex {

// The warps of one core suspended in an mbarrier try_wait, one entry each,
// fed by the local-memory banks' event streams and the LSU blocks' retires.
// An object lives in one bank, which orders every op on it, so its flip either
// precedes a lane's read (which then reads the phase complete) or follows that
// lane's insert: no wake-up is lost. A retire follows all of its warp's inserts.
class MbarWaiterTable : public SimObject<MbarWaiterTable> {
public:
  using Ptr = std::shared_ptr<MbarWaiterTable>;

  // Per bank, in bank-op order, each taken in its cycle.
  std::vector<SimChannel<MbarBankEvt>> BankEvtIn;
  // Per LSU block, taken one cycle late: the commit that pops a try_wait's last
  // result ticks after this table.
  std::vector<SimChannel<uint32_t>> RetireIn;
  // The warps released in a cycle, sent in the next; the channel's delay is the
  // scheduler's stall register.
  SimChannel<WarpMask> UnlockOut;

  MbarWaiterTable(const SimContext& ctx, const char* name, uint32_t num_banks,
                  uint32_t num_blocks, uint32_t suspend_cycles);

protected:
  void on_reset();
  void on_tick();

private:
  struct entry_t {
    bool     valid   = false;
    bool     woke    = false;
    bool     retired = false;
    uint32_t bank    = 0;
    uint32_t row     = 0;
    uint32_t timer   = 0;
  };

  void release(uint32_t wid);

  std::vector<entry_t> entries_;
  // Released in the cycle being completed, on the unlock port next cycle.
  WarpMask released_;
  uint32_t suspend_cycles_;

  friend class SimObject<MbarWaiterTable>;
};

} // namespace vortex

#endif
