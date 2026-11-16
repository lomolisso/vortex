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

#include "mem_coalescer.h"
#include "mem_block_pool.h"
#include <cstring>

using namespace vortex;

MemCoalescer::MemCoalescer(
  const SimContext& ctx,
  const char* name,
  uint32_t input_size,
  uint32_t output_size,
  uint32_t line_size,
  uint32_t queue_size,
  uint32_t delay,
  uint32_t port0_fence_delay
) : SimObject<MemCoalescer>(ctx, name)
  // Single-entry ingress: one request in build/drain flight at a time; a
  // second request waits upstream rather than in local queue slack.
  , ReqIn(this, 1)
  , RspOut(this)
  , ReqOut(output_size, this)
  , RspIn(output_size, this)
  , input_size_(input_size)
  , output_size_(output_size)
  , output_ratio_(input_size / output_size)
  , pending_rd_reqs_(queue_size)
  , sent_mask_(input_size)
  , line_size_(line_size)
  , delay_(delay)
  , port0_fence_delay_(port0_fence_delay)
{}

void MemCoalescer::on_reset() {
  sent_mask_.reset();
  out_round_.valid = false;
  out_round_.lanes.reset();
  out_round_.reqs.clear();
#ifdef VX_CFG_H100_SM_ENABLE
  lane_ch_.clear();
#else
#line 50
#endif
}

void MemCoalescer::on_tick() {
  // process outgoing responses: merge same-tag fragments arriving across
  // channels this tick into one uncoalesced response.
  for (uint32_t o = 0; o < output_size_; ++o) {
    if (RspIn.at(o).empty()) {
      continue;
    }
    if (RspOut.full()) {
      break;
    }
    auto rsp0 = RspIn.at(o).peek();
    auto& entry = pending_rd_reqs_.at(rsp0.tag);

    BitVector<> lane_mask(output_size_);
    std::vector<std::shared_ptr<mem_block_t>> lane_data(output_size_);
    for (uint32_t j = o; j < output_size_; ++j) {
      if (RspIn.at(j).empty()) {
        continue;
      }
      auto& r = RspIn.at(j).peek();
      if (r.tag != rsp0.tag) {
        continue;
      }
      lane_mask.set(j);
      lane_data.at(j) = r.data;
    }

#ifdef VX_CFG_H100_SM_ENABLE
    if (rsp0.tag < lane_ch_.size() && !lane_ch_.at(rsp0.tag).empty()) {
      auto& lane_ch = lane_ch_.at(rsp0.tag);
      LsuRsp out_rsp(input_size_);
      for (uint32_t i = 0; i < input_size_; ++i) {
        uint8_t ch = lane_ch.at(i);
        if (ch != LANE_NO_CH && lane_mask.test(ch) && entry.mask.test(i)) {
          out_rsp.mask.set(i);
          out_rsp.data.at(i) = lane_data.at(ch);
        }
      }
      out_rsp.tag  = entry.tag;
      out_rsp.cid  = rsp0.hart_id;
      out_rsp.uuid = rsp0.uuid;
      RspOut.send(out_rsp, 1);
      DT(4, this->name() << " mem-rsp: " << out_rsp);
      assert(!out_rsp.mask.none());
      entry.mask &= ~out_rsp.mask;
      if (entry.mask.none()) {
        lane_ch.clear();
        pending_rd_reqs_.release(rsp0.tag);
      }
      for (uint32_t j = 0; j < output_size_; ++j) {
        if (lane_mask.test(j)) {
          RspIn.at(j).pop();
        }
      }
      break;
    }
#else
#line 79
#endif
    BitVector<> rsp_mask(input_size_);
    for (uint32_t j = 0; j < output_size_; ++j) {
      if (!lane_mask.test(j)) {
        continue;
      }
      for (uint32_t r = 0; r < output_ratio_; ++r) {
        uint32_t i = j * output_ratio_ + r;
        if (entry.mask.test(i)) {
          rsp_mask.set(i);
        }
      }
    }

    // build memory response — replicate each output-lane data block to all
    // coalesced input lanes (shared_ptr aliasing, no copy)
    LsuRsp out_rsp(input_size_);
    out_rsp.mask = rsp_mask;
    out_rsp.tag = entry.tag;
    out_rsp.cid = rsp0.hart_id;
    out_rsp.uuid = rsp0.uuid;
    for (uint32_t j = 0; j < output_size_; ++j) {
      if (!lane_mask.test(j)) {
        continue;
      }
      for (uint32_t r = 0; r < output_ratio_; ++r) {
        uint32_t i = j * output_ratio_ + r;
        if (entry.mask.test(i)) {
          out_rsp.data.at(i) = lane_data.at(j);
        }
      }
    }

    // send memory response
    RspOut.send(out_rsp, 1);
    DT(4, this->name() << " mem-rsp: " << out_rsp);

    // track remaining responses
    assert(!entry.mask.none());
    entry.mask &= ~rsp_mask;
    if (entry.mask.none()) {
      // whole response received, release tag
      pending_rd_reqs_.release(rsp0.tag);
    }
    for (uint32_t j = 0; j < output_size_; ++j) {
      if (lane_mask.test(j)) {
        RspIn.at(j).pop();
      }
    }
    break; // one merged response per tick
  }

  // drain a pending coalesced round before building a new one
  if (out_round_.valid) {
    this->flush_out_round();
    return;
  }

  // process incoming requests
  if (ReqIn.empty()) {
    // sleep when idle: no queued or in-flight input on either direction and
    // no partial round to drain (out_round_ was handled above). Outstanding
    // fills in pending_rd_reqs_ re-arm the tick when their response is
    // reserved toward RspIn.
    bool idle = (ReqIn.size() == 0);
    for (uint32_t o = 0; idle && o < output_size_; ++o) {
      idle = (RspIn.at(o).size() == 0);
    }
    if (idle) {
      this->tick_sleep();
    }
    return;
  }

  auto& in_req = ReqIn.peek();
  assert(in_req.mask.size() == input_size_);
  assert(!in_req.mask.none());

  // ensure we can allocate a response tag
  if (pending_rd_reqs_.full()) {
    DT(4, this->name() << " queue-full: " << in_req);
    return;
  }

#ifdef VX_CFG_H100_SM_ENABLE
  if ((in_req.op == MemOp::LD || in_req.op == MemOp::ST) && !in_req.flags.fence) {
    this->build_line_round(in_req);
    this->flush_out_round();
    return;
  }
#else
#line 162
#endif
  uint64_t addr_mask = ~uint64_t(line_size_-1);

  const bool in_is_amo = in_req.is_amo();

  BitVector<> out_mask(output_size_);
  std::vector<uint64_t> out_addrs(output_size_);
  std::vector<std::shared_ptr<mem_block_t>> out_data(output_size_);
  std::vector<uint64_t> out_byteen(output_size_, 0);
  // Comparand rides with the lane that owns the output. AMOs never coalesce
  // across lanes, so there is exactly one owner and no merge to do.
  std::vector<uint64_t> out_amo_cmp(output_size_, 0);
  std::vector<uint32_t> out_tids(output_size_, 0);

  BitVector<> cur_mask(input_size_);

  for (uint32_t o = 0; o < output_size_; ++o) {
    for (uint32_t r = 0; r < output_ratio_; ++r) {
      uint32_t i = o * output_ratio_ + r;
      if (sent_mask_.test(i) || !in_req.mask.test(i))
        continue;

      uint64_t seed_addr = in_req.addrs.at(i) & addr_mask;
      cur_mask.set(i);

      // RVA gives no commutativity guarantee across AMO operands —
      // do not coalesce AMO lanes that share a line; each AMO lane
      // emits its own request. For non-AMO, matching addresses coalesce.
      if (!in_is_amo) {
        for (uint32_t s = r + 1; s < output_ratio_; ++s) {
          uint32_t j = o * output_ratio_ + s;
          if (sent_mask_.test(j) || !in_req.mask.test(j))
            continue;
          uint64_t match_addr = in_req.addrs.at(j) & addr_mask;
          if (match_addr == seed_addr) {
            cur_mask.set(j);
          }
        }
      }

      if (in_is_amo) {
        // Carry this lane's original tid through the output slot so the
        // hart id at the memory boundary names the requesting lane.
        // No coalescing across lanes for AMO.
        if (i < in_req.tids.size()) {
          out_tids.at(o) = in_req.tids.at(i);
        }
      }

      // For writes and AMOs, merge per-lane data + byteen into the coalesced
      // block. AMOs never coalesce across lanes (cur_mask only covers lane i),
      // so the merge collapses to a single lane's payload.
      if (in_req.is_write() || in_is_amo) {
        std::shared_ptr<mem_block_t> merged;
        uint64_t merged_byteen = 0;
        for (uint32_t s = r; s < output_ratio_; ++s) {
          uint32_t j = o * output_ratio_ + s;
          if (!cur_mask.test(j) || !in_req.data.at(j))
            continue;
          if (!merged) {
            merged = make_mem_block();
            std::memset(merged->data(), 0, merged->size());
          }
          uint64_t lane_be = in_req.byteen.at(j);
          for (uint32_t b = 0; b < VX_CFG_MEM_BLOCK_SIZE; ++b) {
            if (lane_be & (1ull << b)) {
              (*merged)[b] = (*in_req.data.at(j))[b];
            }
          }
          merged_byteen |= lane_be;
        }
        out_data.at(o) = merged;
        out_byteen.at(o) = merged_byteen;
      }

      out_mask.set(o);
      // AMOs need the byte-level address downstream so the bank can
      // place the RMW result at the correct offset within the line.
      // Non-AMO requests stay line-aligned (no semantic change).
      out_addrs.at(o) = in_is_amo ? in_req.addrs.at(i) : seed_addr;
      out_amo_cmp.at(o) = in_req.amo_cmp.at(i);
      break;
    }
  }

  assert(!out_mask.none());

  uint32_t tag = 0;
  if (!in_req.is_write() || in_is_amo) {
    // Allocate a response tag for read requests and AMOs (which always
    // return rd). Without the AMO branch the response would route through
    // the write path and the LSU MSHR replay would never fire.
    tag = pending_rd_reqs_.allocate(pending_req_t{in_req.tag, cur_mask});
  }

  // build per-channel memory requests
  out_round_.valid = true;
  out_round_.lanes = out_mask;
  out_round_.cur_mask = cur_mask;
  out_round_.reqs.assign(output_size_, MemReq{});
  for (uint32_t o = 0; o < output_size_; ++o) {
    if (!out_mask.test(o)) {
      continue;
    }
    auto& mr  = out_round_.reqs.at(o);
    mr.op     = in_req.op;
    mr.addr   = out_addrs.at(o);
    mr.data   = std::move(out_data.at(o));
    mr.byteen = out_byteen.at(o);
    mr.amo_cmp = out_amo_cmp.at(o);
    mr.tag    = tag;
    mr.hart_id = make_hart_id(in_req.cid, in_req.wid, out_tids.at(o));
    mr.uuid   = in_req.uuid;
    mr.flags  = in_req.flags;
    auto t = get_addr_type(mr.addr);
    mr.flags.io    = (t == AddrType::IO);
    mr.flags.local = (t == AddrType::Shared);
  }

  DT(4, this->name() << " mem-req: coalesced=" << cur_mask.count() << ", lanes=" << out_mask.count() << " (#" << in_req.uuid << ")");

  // track partial responses
  perf_stats_.misses += (cur_mask.count() != in_req.mask.count());

  // The built round is presented to the downstream on the next tick, not
  // this one: the coalescer alternates a build tick with a drain tick, so a
  // batch issues at most every other cycle. Draining is handled at the top
  // of on_tick when out_round_ is valid.
}

// Issue the pending round's per-channel requests; commit the round once
// every lane has been accepted.
void MemCoalescer::flush_out_round() {
  for (uint32_t o = 0; o < output_size_; ++o) {
    if (!out_round_.lanes.test(o)) {
      continue;
    }
    auto& req = out_round_.reqs.at(o);
    uint32_t delay = delay_ + ((o == 0 && req.flags.fence) ? port0_fence_delay_ : 0);
    if (ReqOut.at(o).try_send(req, delay)) {
      out_round_.lanes.reset(o);
    }
  }
  if (!out_round_.lanes.none()) {
    return;
  }

  out_round_.valid = false;
  auto& in_req = ReqIn.peek();
  sent_mask_ |= out_round_.cur_mask;
  if (sent_mask_ == in_req.mask) {
    ReqIn.pop();
    sent_mask_.reset();
  }
}

#ifdef VX_CFG_H100_SM_ENABLE
// The H100 SM's line coalescer: each round takes the line of the first lane
// still to send, with every lane of the request on that line, so a warp
// request costs one L1 access per distinct line. The line goes out as its
// memory blocks (the model's data granule), one output channel each, sent
// the cycle the round is built; blocks with no channel left wait for the
// next round.
void MemCoalescer::build_line_round(const LsuReq& in_req) {
  constexpr uint32_t BLOCK = VX_CFG_MEM_BLOCK_SIZE;
  static_assert(VX_CFG_DCACHE_LINE_SIZE % BLOCK == 0, "an L1 line holds whole memory blocks");
  const uint32_t blocks = line_size_ / BLOCK;
  const bool is_write = in_req.is_write();

  uint32_t seed = input_size_;
  for (uint32_t i = 0; i < input_size_; ++i) {
    if (in_req.mask.test(i) && !sent_mask_.test(i)) {
      seed = i;
      break;
    }
  }
  assert(seed < input_size_);
  const uint64_t line = in_req.addrs.at(seed) / line_size_;

  BitVector<> cur_mask(input_size_);
  BitVector<> out_mask(output_size_);
  std::vector<uint8_t> lane_ch(input_size_, LANE_NO_CH);
  out_round_.reqs.assign(output_size_, MemReq{});
  uint32_t o = 0;
  for (uint32_t blk = 0; blk < blocks && o < output_size_; ++blk) {
    std::shared_ptr<mem_block_t> merged;
    uint64_t merged_byteen = 0;
    bool any = false;
    for (uint32_t i = 0; i < input_size_; ++i) {
      if (!in_req.mask.test(i) || sent_mask_.test(i))
        continue;
      uint64_t a = in_req.addrs.at(i);
      if (a / line_size_ != line || (a % line_size_) / BLOCK != blk)
        continue;
      any = true;
      cur_mask.set(i);
      lane_ch.at(i) = uint8_t(o);
      uint64_t lane_be = in_req.byteen.at(i);
      merged_byteen |= lane_be;
      if (is_write && in_req.data.at(i)) {
        if (!merged) {
          merged = make_mem_block();
          std::memset(merged->data(), 0, merged->size());
        }
        for (uint32_t b = 0; b < BLOCK; ++b) {
          if (lane_be & (1ull << b)) {
            (*merged)[b] = (*in_req.data.at(i))[b];
          }
        }
      }
    }
    if (!any)
      continue;
    auto& mr   = out_round_.reqs.at(o);
    mr.op      = in_req.op;
    mr.addr    = line * line_size_ + blk * BLOCK;
    mr.data    = std::move(merged);
    mr.byteen  = merged_byteen;
    mr.hart_id = make_hart_id(in_req.cid, in_req.wid, 0);
    mr.uuid    = in_req.uuid;
    mr.flags   = in_req.flags;
    auto t = get_addr_type(mr.addr);
    mr.flags.io    = (t == AddrType::IO);
    mr.flags.local = (t == AddrType::Shared);
    out_mask.set(o);
    ++o;
  }

  uint32_t tag = 0;
  if (!is_write) {
    tag = pending_rd_reqs_.allocate(pending_req_t{in_req.tag, cur_mask});
    if (lane_ch_.size() <= tag) {
      lane_ch_.resize(tag + 1);
    }
    lane_ch_.at(tag) = std::move(lane_ch);
  }
  for (uint32_t c = 0; c < output_size_; ++c) {
    if (out_mask.test(c)) {
      out_round_.reqs.at(c).tag = tag;
    }
  }
  out_round_.valid    = true;
  out_round_.lanes    = out_mask;
  out_round_.cur_mask = cur_mask;
  DT(4, this->name() << " mem-req: line=0x" << std::hex << line << std::dec << ", coalesced=" << cur_mask.count() << ", blocks=" << out_mask.count() << " (#" << in_req.uuid << ")");
  perf_stats_.misses += (cur_mask.count() != in_req.mask.count());
}

#else
#line 317
#endif
const MemCoalescer::PerfStats& MemCoalescer::perf_stats() const {
  return perf_stats_;
}