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

#ifdef VX_CFG_EXT_MBAR_ENABLE

#include <algorithm>
#include <deque>
#include "types.h"

namespace vortex {

// One lane of a cp.async uop: where its bytes go, how many come from the
// source, and the source's offset in the read block.
struct CpaLane {
  uint64_t dst     = 0;
  uint32_t ss      = 0;
  uint32_t src_off = 0;
};

// What the LSU block tells its CpaUnit at execute, in the same cycle: one
// event per packet of a copy (the first takes the slot, the last closes its
// FIFO entry); one per commit, wait or arrive, at its last packet.
struct CpaIssue {
  enum Kind : uint8_t { COPY, COMMIT, WAIT, ARRIVE };
  Kind     kind  = COPY;
  uint32_t wid   = 0;
  uint64_t uuid  = 0;
  bool     sop   = true;            // COPY: the uop's first and last packets
  bool     eop   = true;
  uint64_t tmask = 0;               // COPY, ARRIVE: the active threads
  uint32_t slot  = 0;               // COPY: the copy slot of the uop
  uint32_t size  = 0;               // COPY: bytes per lane, 4 or 8
  std::vector<CpaLane> lanes;       // COPY: per thread
  uint32_t n     = 0;               // WAIT: groups that may stay pending
  bool     noinc = false;           // ARRIVE: ARRIVE(1) rather than COMPLETE_TX(1)
  std::vector<uint64_t> addrs;      // ARRIVE: per thread, the mbarrier
};

// The unit's registered state the LSU block stalls on.
struct CpaStatus {
  WarpMask fifo_full;
  WarpMask groups_full;
  WarpMask arrive_pending;
  uint64_t free_slots = 0;

  static CpaStatus idle();

  bool operator==(const CpaStatus& o) const {
    return fifo_full == o.fifo_full && groups_full == o.groups_full
        && arrive_pending == o.arrive_pending && free_slots == o.free_slots;
  }
  bool operator!=(const CpaStatus& o) const { return !(*this == o); }
};

// The cp.async completion unit of one LSU block. A copy takes a slot, whose
// read responses land here and leave as local-memory writes; each warp keeps
// its copies in an in-order FIFO, the commit_group boundaries over it, and one
// pending cp.async.mbarrier.arrive, sent once the head passes it. An eligible
// arrive goes before landed writes: a warp holds one, while its copies can
// keep the writer busy for as long as it issues them.
class CpaUnit : public SimObject<CpaUnit> {
public:
  using Ptr = std::shared_ptr<CpaUnit>;

  static constexpr uint32_t DEPTH  = 8;
  static constexpr uint32_t GROUPS = 8;
  // A slot id is a write tag; the all-ones tag is the arrives'.
  static constexpr uint32_t SLOTS  = std::min<uint32_t>(VX_CFG_LSU_PENDING_SIZE,
                                       (1u << log2ceil(VX_CFG_LSU_PENDING_SIZE)) - 1);
  static constexpr uint32_t NUM_PIDS = (VX_CFG_NUM_THREADS + VX_CFG_NUM_LSU_LANES - 1) / VX_CFG_NUM_LSU_LANES;
  static constexpr uint32_t ARRIVE_TAG = (1u << log2ceil(SLOTS + 1)) - 1;
  static_assert(SLOTS <= 64, "copy slots exceed the status mask");

  // The tag of a read response, relative to the LSU's cp.async tag base.
  static uint32_t rsp_tag(uint32_t slot, uint32_t pid) { return slot * NUM_PIDS + pid; }

  SimChannel<CpaIssue>  IssueIn;   // from the LSU block, in its execute cycle
  SimChannel<LsuRsp>    RspIn;     // its cp.async read responses, in their cycle
  SimChannel<LsuReq>    ReqOut;    // writes and arrives toward the local memory
  SimChannel<LsuRsp>    AckIn;     // their acknowledgements
  SimChannel<CpaStatus> StatusOut; // to the LSU block, registered
  SimChannel<uint32_t>  UnlockOut; // a warp whose wait_group is satisfied

  CpaUnit(const SimContext& ctx, const char* name, uint32_t block);

  // No copy of the warp in its FIFO, no arrive of it pending and no arrive of
  // the unit still in the local memory: the warp may exit.
  bool idle(uint32_t wid) const;

protected:
  void on_reset();
  void on_tick();

private:
  struct Slot {
    bool     valid = false;
    uint32_t wid   = 0;
    uint32_t seq   = 0;
    uint32_t size  = 0;
    uint64_t uuid  = 0;
    struct Lane {
      bool     active = false;
      bool     landed = false;
      bool     sent   = false;
      CpaLane  info;
      uint64_t data   = 0;
    };
    std::vector<Lane> lanes;
  };

  struct Entry {
    uint32_t count  = 0;          // lane writes issued and not yet acked
    bool     closed = false;      // its last packet issued
  };

  struct Warp {
    std::deque<Entry> fifo;
    uint32_t head = 0;            // sequence numbers
    uint32_t tail = 0;
    std::deque<uint32_t> bounds;  // commit_group boundaries: the tail at each commit
    bool     waiting = false;
    uint32_t wait_n  = 0;
    bool     ready   = false;
    bool     arrive  = false;
    bool     noinc   = false;
    uint32_t arrive_tail = 0;
    uint32_t arrive_pid  = 0;
    uint64_t arrive_mask = 0;
    uint64_t arrive_uuid = 0;
    std::vector<uint64_t> arrive_addrs;
  };

  static uint32_t beyond(const Warp& w, uint32_t head);
  bool satisfied(const Warp& w, uint32_t head, uint32_t n) const;
  void apply_issue(const CpaIssue& e, const std::vector<uint32_t>& head_r);
  bool select_write();
  bool select_arrive(const std::vector<uint32_t>& head_r);
  void accept_held();
  void land(const LsuRsp& rsp);
  void ack(const LsuRsp& rsp);
  CpaStatus status() const;

  // The request chosen for the writer port. The adapter takes a packet's
  // lanes as its outputs free up, so the next is chosen once it has all left.
  struct Held {
    bool     valid = false;
    bool     arrive = false;
    uint32_t slot = 0;   // a write: its (slot, pid) and lanes
    uint32_t pid  = 0;
    uint64_t lanes = 0;
    uint32_t wid  = 0;   // an arrive: its warp
    LsuReq   req{VX_CFG_NUM_LSU_LANES};
  };

  uint32_t block_;
  Held held_;
  uint32_t arr_inflight_ = 0;  // arrive lanes sent, not yet answered
  std::vector<Slot> slots_;
  std::vector<Warp> warps_;
  CpaStatus sent_status_;

  friend class SimObject<CpaUnit>;
};

} // namespace vortex

#endif
