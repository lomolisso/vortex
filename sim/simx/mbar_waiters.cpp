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

#ifdef VX_CFG_EXT_MBAR_ENABLE

#include "mbar_waiters.h"
#include <util.h>
#include "debug.h"

using namespace vortex;

MbarWaiterTable::MbarWaiterTable(const SimContext& ctx, const char* name, uint32_t num_banks,
                                 uint32_t num_blocks, uint32_t suspend_cycles)
  : SimObject<MbarWaiterTable>(ctx, name)
  , BankEvtIn(num_banks, SimChannel<MbarBankEvt>(this))
  , RetireIn(num_blocks, SimChannel<uint32_t>(this))
  , UnlockOut(this)
  , entries_(VX_CFG_NUM_WARPS)
  , suspend_cycles_(suspend_cycles)
{}

void MbarWaiterTable::on_reset() {
  std::fill(entries_.begin(), entries_.end(), entry_t{});
  released_.reset();
}

void MbarWaiterTable::release(uint32_t wid) {
  entries_.at(wid) = entry_t{};
  released_.set(wid);
}

// Cycle T's events apply in order: inserts, flips, retires, then the timers of
// the retired entries. T's bank events are taken in tick T and its retires in
// tick T+1, so tick T+1 completes T (its retires, and the timer step of an
// entry they retire) before the port carries T's releases and T+1 begins. An
// entry retired before T is not touched by T's retires, so its timer step runs
// in tick T.
void MbarWaiterTable::on_tick() {
  // Cycle T-1's retires.
  WarpMask retiring;
  for (auto& in : RetireIn) {
    while (!in.empty()) {
      uint32_t wid = in.peek();
      in.pop();
      __assert(!retiring.test(wid), this->name() << " mbarrier waiter retires from two LSU blocks: wid=" << wid);
      retiring.set(wid);
      auto& e = entries_.at(wid);
      __assert(!e.retired, this->name() << " mbarrier waiter retired twice: wid=" << wid);
      if (!e.valid || e.woke) {
        this->release(wid);
        continue;
      }
      e.retired = true;
      e.timer = suspend_cycles_;
      if (e.timer == 0) {
        DT(2, this->name() << " mbar-timeout: wid=" << wid);
        this->release(wid);
      } else {
        --e.timer;
      }
    }
  }

  // The unlock port, registered.
  if (released_.any()) {
    DT(2, this->name() << " mbar-unlock: mask=0x" << std::hex << released_.to_ullong() << std::dec);
    UnlockOut.send(released_, 1);
    released_.reset();
  }

  // Cycle T's inserts (the highest bank's insert of a warp wins), then its flips.
  for (uint32_t b = 0; b < BankEvtIn.size(); ++b) {
    auto& in = BankEvtIn.at(b);
    if (in.empty() || in.peek().flip)
      continue;
    auto& evt = in.peek();
    DT(2, this->name() << " mbar-insert: bank=" << b << ", wid=" << evt.wid << ", row=0x" << std::hex << evt.row << std::dec);
    auto& e = entries_.at(evt.wid);
    __assert(!e.retired, this->name() << " mbarrier waiter insert after retire: wid=" << evt.wid);
    e.valid = true;
    e.bank = b;
    e.row = evt.row;
    in.pop();
  }
  for (uint32_t b = 0; b < BankEvtIn.size(); ++b) {
    auto& in = BankEvtIn.at(b);
    if (in.empty())
      continue;
    auto& evt = in.peek();
    __assert(evt.flip, this->name() << "-bank" << b << ": two mbarrier waiter events in one cycle");
    DT(2, this->name() << " mbar-flip: bank=" << b << ", row=0x" << std::hex << evt.row << std::dec);
    for (auto& e : entries_) {
      if (e.valid && e.bank == b && e.row == evt.row) {
        e.woke = true;
      }
    }
    in.pop();
  }

  // Cycle T's timers.
  bool waiting = false;
  for (uint32_t w = 0; w < entries_.size(); ++w) {
    auto& e = entries_.at(w);
    if (!e.valid)
      continue;
    if (e.retired) {
      if (e.woke) {
        this->release(w);
        continue;
      }
      if (e.timer == 0) {
        DT(2, this->name() << " mbar-timeout: wid=" << w);
        this->release(w);
        continue;
      }
      --e.timer;
    }
    waiting = true;
  }

  // size() counts the packets still in flight, which a sleeping table would miss.
  bool busy = waiting || released_.any();
  for (auto& in : BankEvtIn) {
    busy |= (in.size() != 0);
  }
  for (auto& in : RetireIn) {
    busy |= (in.size() != 0);
  }
  if (!busy) {
    this->tick_sleep();
  }
}

#endif
