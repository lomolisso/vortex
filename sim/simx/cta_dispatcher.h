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

#include <array>
#include <cstdint>
#include <vector>
#include "types.h"
#include "kmu.h"
#ifdef VX_CFG_EXT_DLMEM_ENABLE
#include <optional>
#include <dlmem_ref.h>
#include "barrier_unit.h"
#endif

namespace vortex {

class Core; // forward declaration

struct cta_warp_record_t {
  bool       do_init;
  Word       PC;
  Word       entry;
  ThreadMask tmask;
  Word       mscratch;
  uint32_t   cta_id;
  uint32_t   cta_rank;
  uint32_t   cta_size;
  uint32_t   thread_idx[3];
  uint32_t   block_idx[3];
  uint32_t   block_dim[3];
  uint32_t   grid_dim[3];
  uint64_t   param;
  uint64_t   lmem_addr;
  uint32_t   cluster_size;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  uint32_t   cluster_rank;
#endif
};

#ifdef VX_CFG_EXT_DLMEM_ENABLE
// A SPREAD block left its slot: the event its gang unit releases the slot on.
struct CtaDone {
  uint32_t slot;
  uint32_t gang_id;
  uint32_t epoch;
};
#endif

// CTA dispatcher: takes CTAs off the launch bus and launches their warps, one
// step() per cycle, with the RTL dispatcher's timing.
class CtaDispatcher : public SimObject<CtaDispatcher> {
public:
  // The launch bus's last skid (two entries): CTAs routed to this core and not
  // yet admitted.
  SimChannel<kmu_req_t> bus_in;

  // One credit per admitted CTA, back to the launch bus.
  SimChannel<uint8_t> credit_out;

#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // The admitted block's peer row, for the core's aperture translation table.
  SimChannel<dlmem_ref::RowWrite> PeerWrOut;
#endif
#ifdef DXA_MBAR_ENABLE
  // Slot admissions and frees, and the warps' slots: [0] to the core's DXA
  // completion unit, [1] to the SFU that issues the DXA requests, [2] (no warps) to the DLMEM ingress.
  std::vector<SimChannel<CtaSlotEvt>> DxaSlotOut;
#else
#line 75
#endif

  CtaDispatcher(const SimContext& ctx, const char* name, Core* core);
  ~CtaDispatcher();

  // Advance one cycle. Returns true with the warp that becomes active this
  // cycle.
  bool step(const WarpMask& active_warps, uint32_t* wid_out, cta_warp_record_t* rec_out);

  // Notify that a warp has exited; frees its lmem slot when all warps of the CTA are done.
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // A SPREAD block's last warp also returns the block's done event.
  std::optional<CtaDone> warp_done(uint32_t wid);

  // The peer row recorded when the slot's block was admitted.
  const dlmem_ref::Row& peer_row(uint32_t slot) const { return peer_row_.at(slot); }

  // The cluster-barrier record of the block admitted by the last step(), once.
  bool take_cbar_wr(uint32_t* slot, CbarSlot* info);
#else
  void warp_done(uint32_t wid);
#endif

  // Startup-stub bookkeeping for launches that do not come through step().
  // A fragment warp lands in the same slots as a compute CTA and owes the same
  // per-lane startup, so it has to read and update this state rather than keep
  // its own: a slot the compute path has already initialized must not run the
  // stub again, and vice versa.
  bool slot_needs_init(uint32_t wid, const ThreadMask& tmask) const;
  void mark_slot_inited(uint32_t wid, const ThreadMask& tmask);

  // True while a CTA is being dispatched or waits in the launch lane.
  bool running() const {
    return has_cta_ || fire_valid_ || act_valid_ || (bus_in.size() != 0);
  }

  // Kernel start PC of the most recently dispatched CTA. The Fragment Work
  // Distributor reuses it as the entry PC for injected fragment-wave warps
  // (which run the same kernel as the host-launched driver warp).
  Word kernel_pc() const { return cur_kernel_pc_; }

protected:
  void on_reset();

private:
  void admit(uint64_t now);

  void next_warp(bool is_init, cta_warp_record_t* out);

  // Usable slot count for a given per-CTA stride: floor(capacity/stride) capped
  // at NUM_WARPS (all slots when stride is 0). This is the occupancy bound that
  // replaces the byte ring's free_size — slots at index >= this never fit.
  uint32_t usable_slots(uint32_t stride) const;

  Core*     core_;
  uint32_t  num_threads_;
  uint32_t  num_warps_;
  uint64_t  lmem_base_;
  uint32_t  lmem_capacity_;
  // One-shot latch so an over-capacity kernel reports once instead of per CTA.
  mutable bool lmem_overrun_reported_;

  // Fixed-stride slot allocation. Every resident CTA gets LMEM base
  // slot × stride, where stride = align(lmem_size, MEM_BLOCK_SIZE) is uniform
  // within a kernel. The occupancy bound floor(capacity/stride) (capped at
  // NUM_WARPS) replaces the byte ring's free_size accounting, so no wrap or
  // padding arithmetic is needed. Slots are allocated round-robin via a tail
  // pointer and freed immediately when a CTA's last warp exits.
  std::vector<uint32_t> slot_rem_warps_;  // remaining active warps; slot free when 0
  // Reverse lookup: wid → slot index
  std::vector<uint32_t> wid_to_slot_;
  uint32_t  tail_slot_;   // next slot to allocate (round-robin over usable slots)

  // A warp's slot frees, and the dispatcher takes a CTA again, this many
  // cycles after it exits: the retirement pipeline, then the count write-back.
  static constexpr uint64_t RETIRE_DELAY = 3;
  std::vector<uint64_t> slot_free_at_;
  std::array<uint64_t, RETIRE_DELAY + 1> done_cycle_;  // exit cycle + 1, by cycle

  // The selected warp (warp_fire_r) and the one that goes active this cycle.
  bool      fire_valid_;
  bool      fire_last_;
  uint32_t  fire_wid_;
  cta_warp_record_t fire_rec_;
  bool      act_valid_;
  uint32_t  act_wid_;
  cta_warp_record_t act_rec_;
  WarpMask  dispatched_;
  WarpMask  exited_;    // warps that exited this cycle

  // Currently-in-flight CTA (being dispatched warp-rank by warp-rank)
  bool      has_cta_;
  uint32_t  cur_slot_;    // slot index assigned to the current in-flight CTA
  kmu_req_t cta_;
  uint32_t  cta_size_;
  uint32_t  rank_;
  uint32_t  block_size_rem_;
  uint32_t  thread_idx_[3];
  uint64_t  lmem_addr_;

  Word      cur_kernel_pc_;
  uint8_t   cur_ctx_id_;
  // Which lanes of each warp slot have run the startup stub. Per LANE, not per
  // warp: the stub derives a lane's stack from its hart id, so a slot relaunched
  // with lanes it has not run before must run it again.
  std::vector<ThreadMask> warp_init_lanes_;

#ifdef VX_CFG_EXT_DLMEM_ENABLE
  uint32_t my_core_;                       // index within the hardware cluster
  std::vector<bool>     evt_;              // the slot's block owes a done event
  std::vector<uint32_t> gang_;
  std::vector<uint32_t> epoch_;
  std::vector<dlmem_ref::Row> peer_row_;
  uint32_t legacy_base_;                   // first slot of the same-core cluster being admitted
  bool     cbar_wr_valid_;
  uint32_t cbar_wr_slot_;
  CbarSlot cbar_wr_info_;
#endif
#ifdef DXA_MBAR_ENABLE
  std::vector<uint32_t> dxa_gen_;   // per slot, bumped at each admission
  void dxa_slot_evt(const CtaSlotEvt& e, uint64_t delay, bool to_compl);
#else
#line 192
#endif

  // Cluster co-residency: the first CTA of a cluster reserves K consecutive
  // usable slots (pre-wrapping the tail to 0 if the window would overrun the
  // usable range), so members occupy consecutive slots — member r at LMEM base
  // issuer_base + r × stride, exactly what multicast resolves. The following
  // K-1 members (KMU clears is_first_of_cluster) take tail slots in order.

  friend class SimObject<CtaDispatcher>;
};

} // namespace vortex
