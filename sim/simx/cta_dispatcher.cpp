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

#include <VX_types.h>
#include "cta_dispatcher.h"
#include "core.h"
#include <VX_config.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include "debug.h"

using namespace vortex;

CtaDispatcher::CtaDispatcher(const SimContext& ctx, const char* name, Core* core)
  : SimObject<CtaDispatcher>(ctx, name)
  , bus_in(this, 2)
  , credit_out(this)
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  , PeerWrOut(this)
#endif
#ifdef DXA_MBAR_ENABLE
  , DxaSlotOut(3, SimChannel<CtaSlotEvt>(this, 4 * VX_CFG_NUM_WARPS + 8))
#else
#line 31
#endif
  , core_(core)
  , num_threads_(VX_CFG_NUM_THREADS)
  , num_warps_(VX_CFG_NUM_WARPS)
  , lmem_base_(VX_MEM_LMEM_BASE_ADDR)
  , lmem_capacity_(1u << VX_CFG_LMEM_LOG_SIZE)
  , lmem_overrun_reported_(false)
  , slot_rem_warps_(num_warps_, 0)
  , wid_to_slot_(num_warps_, num_warps_)  // num_warps_ = invalid/unassigned
  , tail_slot_(0)
  , slot_free_at_(num_warps_, 0)
  , fire_valid_(false)
  , fire_last_(false)
  , fire_wid_(0)
  , act_valid_(false)
  , act_wid_(0)
  , has_cta_(false)
  , cur_slot_(0)
  , cta_size_(0)
  , rank_(0)
  , block_size_rem_(0)
  , lmem_addr_(0)
  , cur_kernel_pc_(0)
  , cur_ctx_id_(0)
  , warp_init_lanes_(num_warps_)
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  , my_core_(core->id() % (NUM_SOCKETS * VX_CFG_SOCKET_SIZE))
  , evt_(num_warps_, false)
  , gang_(num_warps_, 0)
  , epoch_(num_warps_, 0)
  , peer_row_(num_warps_)
  , legacy_base_(0)
  , cbar_wr_valid_(false)
  , cbar_wr_slot_(0)
#endif
#ifdef DXA_MBAR_ENABLE
  , dxa_gen_(VX_CFG_NUM_WARPS, 0)
#else
#line 65
#endif
{
  thread_idx_[0] = thread_idx_[1] = thread_idx_[2] = 0;
  for (auto& lanes : warp_init_lanes_) {
    lanes.resize(num_threads_);
  }
  done_cycle_.fill(0);
}

CtaDispatcher::~CtaDispatcher() {}

void CtaDispatcher::on_reset() {
  has_cta_    = false;
  fire_valid_ = false;
  act_valid_  = false;
  dispatched_.reset();
  exited_.reset();
  tail_slot_  = 0;
  for (uint32_t i = 0; i < num_warps_; ++i) {
    slot_rem_warps_[i] = 0;
    slot_free_at_[i]   = 0;
    wid_to_slot_[i]    = num_warps_;  // invalid
  }
  done_cycle_.fill(0);
  cur_kernel_pc_ = 0;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  for (uint32_t i = 0; i < num_warps_; ++i) {
    evt_[i] = false;
  }
  legacy_base_ = 0;
  cbar_wr_valid_ = false;
#endif
#ifdef DXA_MBAR_ENABLE
  std::fill(dxa_gen_.begin(), dxa_gen_.end(), 0);
#else
#line 96
#endif
  // warp_init_lanes_ and cur_ctx_id_ are excluded: which lanes of a slot have
  // already run the startup is keyed on the context, and a launch is not a
  // device reset. Clearing them here would make the context check dead and
  // re-run the startup on every launch, including one that dispatches no CTAs
  // of its own. A context change in step() remains the only invalidator, and
  // both are zero-initialized at construction.
}

uint32_t CtaDispatcher::usable_slots(uint32_t stride) const {
  if (stride == 0) {
    return num_warps_;
  }
  uint32_t fit = uint32_t(lmem_capacity_ / stride);
  if (fit > num_warps_) {
    fit = num_warps_;
  }
  if (fit == 0) {
    // The CTA does not fit. One slot is still granted so the launch makes
    // forward progress, but it will write past that slot, and a silent
    // overrun surfaces later as unattributable data corruption.
    if (!lmem_overrun_reported_) {
      lmem_overrun_reported_ = true;
      std::cerr << "Warning: CTA shared-memory footprint " << stride
                << " bytes exceeds the " << lmem_capacity_
                << "-byte capacity; the CTA is admitted anyway and will "
                   "overrun its slot." << std::endl;
    }
    fit = 1;
  }
  return fit;
}

bool CtaDispatcher::step(const WarpMask& active_warps, uint32_t* wid_out, cta_warp_record_t* rec_out) {
  uint64_t now = SimPlatform::instance().cycles();
  // A warp goes active the cycle after its selection is registered.
  bool out = act_valid_;
  if (act_valid_) {
    *wid_out = act_wid_;
    *rec_out = act_rec_;
    act_valid_ = false;
  }
  if (fire_valid_) {
    act_valid_ = true;
    act_wid_   = fire_wid_;
    act_rec_   = fire_rec_;
  }
  bool fire = false;
  if (!has_cta_) {
    // Idle: take the head CTA; its first warp is picked from the next cycle.
    this->admit(now);
  } else if (fire_valid_ && fire_last_) {
    // The CTA's last warp was picked last cycle: idle again from the next.
    has_cta_ = false;
  } else {
    // Lowest warp slot not active, not given a rank, not exited this cycle (registered in the RTL).
    int free_wid = -1;
    for (int wid = 0; wid < (int)num_warps_; ++wid) {
      if (!active_warps.test(wid) && !dispatched_.test(wid) && !exited_.test(wid)) {
        free_wid = wid;
        break;
      }
    }
    if (free_wid >= 0) {
      // Skip the startup only if every lane this launch activates has already run it.
      uint32_t active = (block_size_rem_ >= num_threads_) ? num_threads_ : block_size_rem_;
      auto& inited = warp_init_lanes_[free_wid];
      bool needs_init = false;
      for (uint32_t t = 0; t < active; ++t) {
        if (!inited.test(t)) {
          needs_init = true;
          break;
        }
      }
      this->next_warp(needs_init, &fire_rec_);
      for (uint32_t t = 0; t < active; ++t) {
        inited.set(t);
      }
      wid_to_slot_[free_wid] = cur_slot_;
#ifdef DXA_MBAR_ENABLE
      this->dxa_slot_evt({CtaSlotEvt::WARP, cur_slot_, 0, 0, uint32_t(free_wid), {}}, 1, false);
#else
#line 174
#endif
      dispatched_.set(free_wid);
      fire_wid_  = uint32_t(free_wid);
      fire_last_ = (rank_ >= cta_size_);
      fire = true;
    }
  }
  fire_valid_ = fire;
  exited_.reset();
  return out;
}

void CtaDispatcher::admit(uint64_t now) {
  // The count write-back of a warp that exited RETIRE_DELAY cycles ago holds
  // the table's write port.
  uint64_t blk = now - RETIRE_DELAY;
  if (bus_in.empty() || (now >= RETIRE_DELAY && done_cycle_[blk % done_cycle_.size()] == blk + 1)) {
    return;
  }
  const kmu_req_t& next = bus_in.peek();

  // Fixed-stride admission. Round the per-CTA LMEM size up to a
  // MEM_BLOCK_SIZE multiple — this uniform stride is the slot pitch.
  // Multicast resolves receiver destinations as `issuer_base + r * stride`
  // and the LMEM model is block-addressed (byteen-masked), so a non-aligned
  // stride would truncate the destination; the descriptor handler rounds
  // its stride the same way to stay consistent.
#if defined(VX_CFG_EXT_DXA_ENABLE) && (VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8) > VX_CFG_MEM_BLOCK_SIZE)
  // A DXA multicast offsets its receivers by whole DMA words (a banked
  // local-memory row), so every slot must start on one.
  constexpr uint32_t kSlotAlign = VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8);
  uint32_t stride = (next.lmem_size + kSlotAlign - 1u) & ~(kSlotAlign - 1u);
#else
  uint32_t stride = (next.lmem_size + VX_CFG_MEM_BLOCK_SIZE - 1u) & ~uint32_t(VX_CFG_MEM_BLOCK_SIZE - 1u);
#endif
#line 202
  uint32_t max_slots = usable_slots(stride);

  // Round-robin slot allocation over the usable range. Normalize the tail in
  // case a kernel transition shrank the usable count.
  if (tail_slot_ >= max_slots) {
    tail_slot_ = 0;
  }
  uint32_t base = tail_slot_;

#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // A SPREAD block takes the slot its gang reserved, past the window and tail.
  bool is_pinned = (next.policy == KMU_POLICY_SPREAD);
  __assert(next.pinned == is_pinned, "a SPREAD block must arrive pinned");
  __assert(!is_pinned || next.dest_core == my_core_, "a pinned block reached core " << my_core_ << ", not " << next.dest_core);
#endif
  auto slot_busy = [&](uint32_t s) {
    return slot_rem_warps_[s] != 0 || now < slot_free_at_[s];
  };

  // Cluster start: reserve K consecutive usable slots. Pre-wrap to 0 if the
  // window would overrun the usable range so the cluster stays contiguous
  // (members must occupy consecutive slots for multicast). All K must be free
  // up front so the following members never stall mid-cluster.
  uint32_t k = 1;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  if (is_pinned) {
    base = next.forced_slot;
    // The gang's mirror frees a slot only after its done event left, so this cannot fire.
    __assert(base < num_warps_ && !slot_busy(base), "a pinned block's forced slot is not free");
  } else
#endif
  if (next.is_first_of_cluster) {
    k = next.cluster_dim[0]
      * next.cluster_dim[1]
      * next.cluster_dim[2];
    if (k == 0) {
      k = 1;
    }
    if (base + k > max_slots) {
      base = 0;
    }
    if (k > max_slots) {
      k = max_slots;  // cluster larger than co-residency: clamp (degenerate)
    }
    for (uint32_t i = 0; i < k; ++i) {
      if (slot_busy(base + i)) {
        return;  // window not free yet — wait
      }
    }
  } else if (slot_busy(base)) {
    // Standalone CTA, or a later cluster member (its window slot is reserved).
    return;
  }

  // A new launch (by id, not PC) is a new context: every slot owes the startup again.
  cur_kernel_pc_ = next.PC;
  if (next.ctx_id != cur_ctx_id_) {
    cur_ctx_id_ = next.ctx_id;
    for (uint32_t i = 0; i < num_warps_; ++i) {
      warp_init_lanes_[i].reset();
    }
  }

  // Accept the CTA into its slot; the skid's ready sees the pop next cycle.
  DT(1, this->name() << " kmu-accept: cta_id=" << next.cta_id << ", slot=" << base);
  cta_ = next;
  bus_in.pop();
  credit_out.send(1, 0);

  cta_size_       = (cta_.block_size + num_threads_ - 1) / num_threads_;
  rank_           = 0;
  block_size_rem_ = cta_.block_size;
  thread_idx_[0] = thread_idx_[1] = thread_idx_[2] = 0;

  cur_slot_  = base;
  lmem_addr_ = lmem_base_ + uint64_t(base) * stride;
  slot_rem_warps_[base] = cta_size_;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  if (!is_pinned) {
    tail_slot_ = (base + 1) % max_slots;
  }
  evt_[base]   = is_pinned;
  gang_[base]  = cta_.gang_id;
  epoch_[base] = cta_.epoch;
  cbar_wr_valid_ = true;
  cbar_wr_slot_  = base;
  cbar_wr_info_  = CbarSlot{};
  if (is_pinned) {
    peer_row_[base] = cta_.row;
    cbar_wr_info_.gang    = (cta_.row.k > 1);
    cbar_wr_info_.gang_id = cta_.gang_id;
    cbar_wr_info_.epoch   = cta_.epoch;
    cbar_wr_info_.base    = base;
    cbar_wr_info_.k       = cta_.row.k;
  } else {
    if (cta_.is_first_of_cluster) {
      legacy_base_ = base;
    }
    // the row spans the clamped window, as the RTL builds it
    uint32_t K = cta_.cluster_dim[0] * cta_.cluster_dim[1] * cta_.cluster_dim[2];
    K = std::min((K == 0) ? 1u : K, max_slots);
    peer_row_[base] = dlmem_ref::legacy_row(my_core_, legacy_base_, K, VX_CFG_DLMEM_MAX_CLUSTER);
    cbar_wr_info_.base = legacy_base_;
    cbar_wr_info_.k    = K;
  }
  // The RTL registers the row write one cycle after admission.
  PeerWrOut.send({base, peer_row_[base], stride}, 1);
#else
  tail_slot_ = (base + 1) % max_slots;
#endif
#ifdef DXA_MBAR_ENABLE
  dxa_gen_[base] = (dxa_gen_[base] + 1) & ((1u << DXA_CTA_GEN_BITS) - 1);
  this->dxa_slot_evt({CtaSlotEvt::ADMIT, base, dxa_gen_[base], stride, 0, peer_row_[base]}, 1, true);
#else
#line 311
#endif

  dispatched_.reset();
  has_cta_ = true;
}

#ifdef VX_CFG_EXT_DLMEM_ENABLE
std::optional<CtaDone> CtaDispatcher::warp_done(uint32_t wid) {
  exited_.set(wid);
  uint32_t slot = wid_to_slot_[wid];
  if (slot >= num_warps_) return std::nullopt;  // not a CTA-dispatcher warp
#else
void CtaDispatcher::warp_done(uint32_t wid) {
  exited_.set(wid);
  uint32_t slot = wid_to_slot_[wid];
  if (slot >= num_warps_) return;  // not a CTA-dispatcher warp
#endif
  wid_to_slot_[wid] = num_warps_;  // clear assignment
  assert(slot_rem_warps_[slot] > 0);
  // When the last warp of a CTA exits, its slot (and the LMEM region at
  // slot × stride) frees for reuse — no in-order reclaim.
  uint64_t now = SimPlatform::instance().cycles();
  done_cycle_[now % done_cycle_.size()] = now + 1;
  if (--slot_rem_warps_[slot] == 0) {
    slot_free_at_[slot] = now + RETIRE_DELAY;
  }
#ifdef DXA_MBAR_ENABLE
  if (slot_rem_warps_[slot] == 0) {
    // The slot is free once its warp count is written back.
    this->dxa_slot_evt({CtaSlotEvt::FREE, slot, 0, 0, 0, {}}, RETIRE_DELAY + (evt_[slot] ? 1 : 0), true);
  }
#else
#line 336
#endif
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  if (slot_rem_warps_[slot] != 0 || !evt_[slot]) {
    return std::nullopt;
  }
  // The slot stays taken until its done event leaves, a cycle later.
  evt_[slot] = false;
  slot_free_at_[slot] = now + RETIRE_DELAY + 1;
  return CtaDone{slot, gang_[slot], epoch_[slot]};
#endif
}

#ifdef VX_CFG_EXT_DLMEM_ENABLE
bool CtaDispatcher::take_cbar_wr(uint32_t* slot, CbarSlot* info) {
  if (!cbar_wr_valid_)
    return false;
  cbar_wr_valid_ = false;
  *slot = cbar_wr_slot_;
  *info = cbar_wr_info_;
  return true;
}
#endif

bool CtaDispatcher::slot_needs_init(uint32_t wid, const ThreadMask& tmask) const {
  const auto& inited = warp_init_lanes_[wid];
  for (uint32_t t = 0; t < num_threads_; ++t) {
    if (tmask.test(t) && !inited.test(t))
      return true;
  }
  return false;
}

void CtaDispatcher::mark_slot_inited(uint32_t wid, const ThreadMask& tmask) {
  auto& inited = warp_init_lanes_[wid];
  for (uint32_t t = 0; t < num_threads_; ++t) {
    if (tmask.test(t))
      inited.set(t);
  }
}

void CtaDispatcher::next_warp(bool do_init, cta_warp_record_t* out) {
  out->do_init  = do_init;
  out->PC       = Word(cta_.PC);
  out->entry    = Word(cta_.entry);
  out->mscratch = Word(cta_.param);
  out->param    = cta_.param;
  out->cta_id   = cur_slot_;
  out->cta_rank = rank_;
  out->cta_size = cta_size_;

  out->thread_idx[0] = thread_idx_[0];
  out->thread_idx[1] = thread_idx_[1];
  out->thread_idx[2] = thread_idx_[2];

  out->block_idx[0] = cta_.block_idx[0];
  out->block_idx[1] = cta_.block_idx[1];
  out->block_idx[2] = cta_.block_idx[2];

  out->block_dim[0] = cta_.block_dim[0];
  out->block_dim[1] = cta_.block_dim[1];
  out->block_dim[2] = cta_.block_dim[2];

  out->grid_dim[0] = cta_.grid_dim[0];
  out->grid_dim[1] = cta_.grid_dim[1];
  out->grid_dim[2] = cta_.grid_dim[2];

  out->lmem_addr = lmem_addr_;
  out->cluster_size = cta_.cluster_dim[0] * cta_.cluster_dim[1] * cta_.cluster_dim[2];
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  out->cluster_rank = cta_.cluster_rank;
#endif

  // Thread mask: full warp or partial last warp.
  out->tmask.resize(num_threads_);
  uint32_t active = (block_size_rem_ >= num_threads_) ? num_threads_ : block_size_rem_;
  for (uint32_t t = 0; t < active; ++t) {
    out->tmask.set(t);
  }

  // advance thread_idx with 3-D carry propagation (additive delta + carry)
  uint32_t next_x = thread_idx_[0] + cta_.warp_step[0];
  bool wrap_x = (next_x >= cta_.block_dim[0]);
  thread_idx_[0] = wrap_x ? (next_x - cta_.block_dim[0]) : next_x;

  uint32_t next_y = thread_idx_[1] + cta_.warp_step[1] + (wrap_x ? 1 : 0);
  bool wrap_y = (next_y >= cta_.block_dim[1]);
  thread_idx_[1] = wrap_y ? (next_y - cta_.block_dim[1]) : next_y;

  thread_idx_[2] = thread_idx_[2] + cta_.warp_step[2] + (wrap_y ? 1 : 0);

  block_size_rem_ -= (block_size_rem_ >= num_threads_) ? num_threads_ : block_size_rem_;

  ++rank_;
}

#ifdef DXA_MBAR_ENABLE
void CtaDispatcher::dxa_slot_evt(const CtaSlotEvt& e, uint64_t delay, bool to_compl) {
  if (to_compl) {
    DxaSlotOut.at(0).send(e, delay);
    DxaSlotOut.at(2).send(e, delay);
  }
  DxaSlotOut.at(1).send(e, delay);
}
#endif
