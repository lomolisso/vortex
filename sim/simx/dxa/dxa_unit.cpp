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
#include "kmu/cta_gang.h"
#else
#line 20
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
#else
#line 67
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
  req.src_core  = core_->id() % dlmem_cfg().ncc;
  req.dst_core  = req.src_core;
  req.remote    = false;
  req.win       = false;
  req.win_room  = 0;
  req.cluster   = false;
  req.num_rcv   = 0;
  req.bulk      = false;
  req.src_addr  = 0;
  req.size      = 0;
  req.sent      = SimPlatform::instance().cycles();
  if (std::get<DxaType>(trace->op_type) == DxaType::BULK) {
    this->translate_bulk(trace, req);
  } else if (mode && cta_mask != 0) {
    this->cluster_receivers(trace, meta, cta_mask, req);
  } else {
    this->translate(trace, meta, cta_mask, req);
  }
#else
#line 87
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
  case CtaSlotEvt::ADMIT:
    slot_gen_.at(e.slot) = e.gen;
    slot_row_.at(e.slot) = e.row;
    stride_ = e.stride;
    break;
  case CtaSlotEvt::WARP:  warp_slot_.at(e.wid) = e.slot; break;
  default: break;
  }
}

void DxaUnit::issue_abort(const instr_trace_t* trace, const char* msg) const {
  std::cerr << SimPlatform::instance().cycles() << ": *** core" << core_->id() << " " << msg
            << ": wid=" << trace->wid << " (#" << trace->uuid << ")" << std::endl;
  std::abort();
}

// A window destination selects rank a[LMEM_LOG +: RANK_BITS] at block offset
// a[LMEM_LOG-1:0]; a window mbarrier carries its rank in meta[28:24] and its
// block-relative doubleword offset in the low payload bits. Both resolve
// through the issuing block's peer row. A block on this core keeps the local
// path at that block's slot; another core's block is remote.
void DxaUnit::translate(const instr_trace_t* trace, uint32_t meta, uint32_t cta_mask, DxaReq& req) const {
  const auto cfg = dlmem_cfg();
  const uint32_t rbits = dlmem_ref::rank_bits(cfg);
  const uint32_t wl = cfg.lmem_log + rbits;
  const bool dst_win  = (req.smem_addr >> wl) == (cfg.dlmem_base >> wl);
  const bool mbar_win = req.mode && ((meta >> 29) & 1u);
  if (!req.mode && __builtin_popcount(cta_mask) > 1) {
    // A barrier multicast replays into consecutive slots of this core.
    for (const auto& p : slot_row_.at(req.slot).peers) {
      if (p.core != req.src_core) {
        this->issue_abort(trace, "DXA barrier multicast into a cluster spread over cores");
      }
    }
  }
  if (!dst_win && !mbar_win) {
    return;
  }
  const uint32_t issue_slot = req.slot;
  const auto& row = slot_row_.at(issue_slot);
  const uint32_t rank   = uint32_t((req.smem_addr >> cfg.lmem_log) & dlmem_ref::low_mask(rbits));
  const uint64_t off    = req.smem_addr & dlmem_ref::low_mask(cfg.lmem_log);
  const uint32_t m_rank = (meta >> 24) & 0x1fu;
  const uint64_t m_off  = uint64_t(req.mbar_off) << 3;
  const bool dst_oob  = dst_win && (rank >= row.k || off >= stride_);
  const bool mbar_oob = mbar_win && (m_rank >= row.k || m_off >= stride_);
  if (dst_oob || mbar_oob) {
    this->issue_abort(trace, "DXA destination rank outside the cluster");
  }
  const dlmem_ref::Peer self{req.src_core, issue_slot};
  const dlmem_ref::Peer dst  = dst_win ? row.peers.at(rank) : self;
  const dlmem_ref::Peer mbar = mbar_win ? row.peers.at(m_rank) : self;
  if (req.mode && (dst.core != mbar.core || dst.slot != mbar.slot)) {
    this->issue_abort(trace, "DXA mbarrier must be in the destination block");
  }
  if (!req.mode) {
    this->issue_abort(trace, "DXA cluster destination needs an mbarrier");
  }
  if (dst_win) {
    req.smem_addr = cfg.lmem_base + uint64_t(dst.slot) * stride_ + off;
    req.win       = true;
    req.win_room  = uint32_t(stride_ - off);
  }
  if (mbar_win) {
    req.mbar_off = uint32_t(((uint64_t(mbar.slot) * stride_) >> 3) + req.mbar_off);
  }
  req.slot     = dst.slot;
  req.gen      = slot_gen_.at(dst.slot);
  req.dst_core = dst.core;
  req.remote   = (dst.core != req.src_core);
  DT(4, "dxa-unit window: core=" << core_->id() << ", wid=" << trace->wid << ", rank=" << rank
     << ", dst={core=" << dst.core << ", slot=" << dst.slot << "}, off=0x" << std::hex << off
     << ", mbar_off=0x" << req.mbar_off << std::dec << ", remote=" << req.remote);
}

// A cluster multicast lands at the issuer's block-relative offsets in every
// masked rank's block. Receivers on this core take the local path at their own
// slot; the others are remote.
void DxaUnit::cluster_receivers(const instr_trace_t* trace, uint32_t meta, uint32_t cta_mask, DxaReq& req) const {
  const auto cfg = dlmem_cfg();
  const uint32_t wl = cfg.lmem_log + dlmem_ref::rank_bits(cfg);
  const bool dst_win  = (req.smem_addr >> wl) == (cfg.dlmem_base >> wl);
  const bool mbar_win = (meta >> 29) & 1u;
  if (dst_win || mbar_win) {
    this->issue_abort(trace, "DXA multicast needs a local address");
  }
  const auto& row = slot_row_.at(req.slot);
  if ((uint64_t(cta_mask) >> row.k) != 0) {
    this->issue_abort(trace, "DXA multicast rank outside the cluster");
  }
  // Block-relative offsets, modulo the local memory as the RTL computes them.
  const uint64_t lmem_mask = dlmem_ref::low_mask(cfg.lmem_log);
  const uint64_t base      = uint64_t(req.slot) * stride_;
  const uint64_t off       = (req.smem_addr - cfg.lmem_base - base) & lmem_mask;
  const uint32_t mbar_rel  = uint32_t((req.mbar_off - (base >> 3)) & (lmem_mask >> 3));
  if ((uint64_t(mbar_rel) << 3) >= stride_) {
    this->issue_abort(trace, "mbarrier offset outside the block");
  }
  req.cluster  = true;
  req.win      = true;
  req.win_room = (off < stride_) ? uint32_t(stride_ - off) : 0u;
  for (uint32_t r = 0; r < row.k; ++r) {
    if (((cta_mask >> r) & 1u) == 0)
      continue;
    const auto& p  = row.peers.at(r);
    const uint64_t pbase = uint64_t(p.slot) * stride_;
    auto& rcv     = req.rcvs.at(req.num_rcv++);
    rcv.core      = p.core;
    rcv.slot      = p.slot;
    rcv.gen       = slot_gen_.at(p.slot);
    rcv.smem_addr = cfg.lmem_base + pbase + off;
    rcv.mbar_off  = uint32_t(pbase >> 3) + mbar_rel;
    rcv.remote    = (p.core != req.src_core);
    DT(4, "dxa-unit multicast: core=" << core_->id() << ", wid=" << trace->wid << ", rank=" << r
       << ", dst={core=" << p.core << ", slot=" << p.slot << "}, off=0x" << std::hex << off
       << ", mbar_off=0x" << rcv.mbar_off << std::dec << ", remote=" << rcv.remote);
  }
}

// A bulk copy's lanes are rs1 = {dst, mbar, src, size}. dst and mbar are each a
// window address, resolved through the issuing block's peer row, or a local
// address in the issuer's own block; src is local. The bound of dst's offset
// plus the size against its block is the engine's, as for a box.
void DxaUnit::translate_bulk(const instr_trace_t* trace, DxaReq& req) const {
  const auto& rs1 = trace->src_data[0];
  const uint64_t dst  = static_cast<uint64_t>(rs1.at(0).u);
  const uint64_t mbar = static_cast<uint64_t>(rs1.at(1).u);
  const uint64_t src  = static_cast<uint64_t>(rs1.at(2).u);
  const uint64_t size = static_cast<uint64_t>(rs1.at(3).u);
  const auto cfg = dlmem_cfg();
  const uint32_t rbits = dlmem_ref::rank_bits(cfg);
  const uint32_t wl = cfg.lmem_log + rbits;
  const uint32_t own = req.slot;
  const auto& row = slot_row_.at(own);
  const uint64_t own_base = cfg.lmem_base + uint64_t(own) * stride_;
  const uint64_t lmem_mask = dlmem_ref::low_mask(cfg.lmem_log);

  struct Ptr {
    bool     win;
    uint32_t rank;
    uint64_t off;   // in the block
  };
  auto decode = [&](uint64_t a) {
    if ((a >> wl) == (cfg.dlmem_base >> wl)) {
      return Ptr{true, uint32_t((a >> cfg.lmem_log) & dlmem_ref::low_mask(rbits)), a & lmem_mask};
    }
    // Below the block it wraps past the stride, as above it.
    return Ptr{false, 0, a - own_base};
  };
  const Ptr d = decode(dst);
  const Ptr m = decode(mbar);
  const uint64_t s_off = src - own_base;

  if (size == 0 || (size % 16) != 0 || (src % 16) != 0 || (d.off % 16) != 0) {
    this->issue_abort(trace, "DXA bulk copy needs 16-byte size and alignment");
  }
  if (s_off >= stride_ || size > stride_ - s_off) {
    this->issue_abort(trace, "DXA bulk copy source outside the block");
  }
  if ((d.win && (d.rank >= row.k || d.off >= stride_))
   || (m.win && (m.rank >= row.k || m.off >= stride_))) {
    this->issue_abort(trace, "DXA destination rank outside the cluster");
  }
  const dlmem_ref::Peer self{req.src_core, own};
  const dlmem_ref::Peer dblk = d.win ? row.peers.at(d.rank) : self;
  const dlmem_ref::Peer mblk = m.win ? row.peers.at(m.rank) : self;
  const bool m_own = m.win || m.off < stride_;
  if (!m_own || dblk.core != mblk.core || dblk.slot != mblk.slot) {
    this->issue_abort(trace, "DXA mbarrier must be in the destination block");
  }

  req.bulk      = true;
  req.mode      = true;
  req.src_addr  = src;
  req.size      = uint32_t(size & dlmem_ref::low_mask(VX_CFG_LMEM_LOG_SIZE + 1));
  req.desc_slot = 0;
  req.bar_id    = 0;
  req.cta_mask  = 0;
  for (auto& c : req.coords) c = 0;
  req.smem_addr = cfg.lmem_base + uint64_t(dblk.slot) * stride_ + d.off;
  req.mbar_off  = uint32_t((uint64_t(mblk.slot) * stride_ + m.off) >> 3);
  req.slot      = dblk.slot;
  req.gen       = slot_gen_.at(dblk.slot);
  req.dst_core  = dblk.core;
  req.remote    = (dblk.core != req.src_core);
  req.win       = true;
  req.win_room  = (d.off < stride_) ? uint32_t(stride_ - d.off) : 0u;
  DT(4, "dxa-unit bulk: core=" << core_->id() << ", wid=" << trace->wid << ", src=0x" << std::hex << src
     << ", size=0x" << size << ", dst={core=" << std::dec << dblk.core << ", slot=" << dblk.slot << "}, off=0x"
     << std::hex << d.off << ", mbar_off=0x" << req.mbar_off << std::dec << ", remote=" << req.remote);
}
#endif
