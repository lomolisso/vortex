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

#include "dsmem_lsu_adapter.h"
#include "cta_residency_table.h"
#include "core.h"
#include "cluster.h"
#include "scheduler.h"
#include "debug.h"
#include <cassert>

using namespace vortex;

DsmemLsuAdapter::DsmemLsuAdapter(const SimContext& ctx, const char* name, Core* core)
  : SimObject<DsmemLsuAdapter>(ctx, name)
  , ReqIn(this)
  , RspOut(this)
  , ReqOut(this)
  , RspIn(this)
  , core_(core)
  , lg2_lanes_(log2ceil(VX_CFG_NUM_LSU_LANES))
  , lg2_warps_(log2ceil(VX_CFG_NUM_WARPS))
  , pending_mask_(VX_CFG_NUM_LSU_LANES)
{}

void DsmemLsuAdapter::on_reset() {
  pending_mask_.reset();
}

void DsmemLsuAdapter::on_tick() {
  // ── Responses: load data → the switch; store acks → drain accounting ─────
  if (!RspIn.empty() && !RspOut.full()) {
    auto& rsp = RspIn.peek();
    uint32_t lane = rsp.tag & ((1u << lg2_lanes_) - 1);
    uint32_t wid  = (rsp.tag >> lg2_lanes_) & ((1u << lg2_warps_) - 1);
    uint64_t tag  = rsp.tag >> (lg2_lanes_ + lg2_warps_);
    if (!rsp.data) {
      // A write ack (strsp): the remote store has committed in the peer's
      // LMEM banks — release one unit of the warp's drain obligation.
      core_->dsmem_store_dec(wid);
      DT(4, this->name() << " st-ack: wid=" << wid << ", lane=" << lane);
      RspIn.pop();
    } else {
      LsuRsp out_rsp(VX_CFG_NUM_LSU_LANES);
      out_rsp.mask.set(lane);
      out_rsp.tag  = tag;
      out_rsp.cid  = rsp.hart_id;
      out_rsp.uuid = rsp.uuid;
      out_rsp.data.at(lane) = rsp.data;
      RspOut.send(out_rsp, 1);
      DT(4, this->name() << " ld-rsp: " << out_rsp);
      RspIn.pop();
    }
  }

  // ── Requests: one remote lane per cycle onto the network egress port ─────
  if (!ReqIn.empty()) {
    auto& in_req = ReqIn.peek();
    if (pending_mask_.none()) {
      pending_mask_ = in_req.mask;
      assert(!pending_mask_.none());
    }

    auto& csrs = core_->scheduler().warp(in_req.wid).cta_csrs;
    uint32_t k = (csrs.cluster_size != 0) ? csrs.cluster_size : 1;

    for (uint32_t i = 0; i < VX_CFG_NUM_LSU_LANES; ++i) {
      if (!pending_mask_.test(i))
        continue;

      uint64_t addr = in_req.addrs.at(i);
      uint32_t region = dsmem_region(addr);
      assert(region != 0 && "local-shared lane routed onto the DSMEM path");
      uint32_t target_rank = (csrs.cluster_rank + region) % k;

      auto* slot = core_->cluster()->cta_table()->resolve(csrs.cluster_id, target_rank);
      // The kernel must rendezvous (cluster barrier) before touching a peer's
      // scratchpad — a missing entry means the peer CTA is not resident.
      assert(slot != nullptr && "DSMEM access to a non-resident cluster rank");

      MemReq out_req;
      out_req.op      = in_req.op;
      out_req.addr    = slot->lmem_base + (addr & ((1u << VX_CFG_LMEM_LOG_SIZE) - 1));
      out_req.data    = in_req.data.at(i);
      out_req.byteen  = in_req.byteen.at(i);
      out_req.tag     = (((in_req.tag << lg2_warps_) | in_req.wid) << lg2_lanes_) | i;
      out_req.hart_id = make_hart_id(in_req.cid, in_req.wid, in_req.tids.at(i));
      out_req.uuid    = in_req.uuid;
      out_req.flags.local        = 1;
      out_req.flags.dsmem_remote = 1;
      out_req.flags.dsmem_target = slot->core_id % VX_CFG_NUM_CORES;
      if (out_req.is_write()) {
        // Ask the peer LMEM for a store ack; the matching dsmem_store_inc
        // happened at LSU issue (the earliest post-retire point, closing the
        // window where an in-flight store could slip past a barrier arrive).
        out_req.flags.strsp = 1;
      }

      if (ReqOut.try_send(out_req, 1)) {
        DT(4, this->name() << " req" << i << ": " << out_req);
        pending_mask_.reset(i);
      }
      break; // one network flit per cycle
    }

    if (pending_mask_.none()) {
      ReqIn.pop();
    }
  }
}
