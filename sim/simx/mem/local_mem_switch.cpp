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

#include "local_mem_switch.h"

using namespace vortex;

LocalMemSwitch::LocalMemSwitch(
  const SimContext& ctx,
  const char* name,
  uint32_t delay
) : SimObject<LocalMemSwitch>(ctx, name)
  // Single-entry ingress: the switch accepts a new request only as its
  // current one drains, so upstream issue feels downstream cadence
  // immediately instead of running ahead into queue slack.
  , ReqIn(this, 1)
  , RspOut(this)
  , ReqOutLmem(this)
  , RspInLmem(this)
  , ReqOutDC(this)
  , RspInDC(this)
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  , ReqOutRemote(this)
  , RspInRemote(this)
#endif
  , delay_(delay)
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  , rsp_rr_last_(2)
#endif
{}

#ifdef VX_CFG_EXT_DLMEM_ENABLE
void LocalMemSwitch::on_reset() {
  // VX_rr_arbiter starts with its last grant at the top input.
  rsp_rr_last_ = 2;
}
#endif

void LocalMemSwitch::on_tick() {
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // One response per cycle, round-robin over {remote, global, local}: the RTL's ARBITER "R" merge.
  {
    SimChannel<LsuRsp>* rsp_in[3] = {&RspInRemote, &RspInDC, &RspInLmem};
    for (uint32_t k = 1; k <= 3; ++k) {
      uint32_t i = (rsp_rr_last_ + k) % 3;
      if (rsp_in[i]->empty())
        continue;
      auto& out_rsp = rsp_in[i]->peek();
      if (RspOut.try_send(out_rsp, 1)) {
        DT(4, this->name() << (i == 0 ? " remote" : (i == 1 ? " dc" : " lmem")) << "-rsp: " << out_rsp);
        rsp_in[i]->pop();
        rsp_rr_last_ = i;
      }
      break;
    }
  }
#else
  // process outgoing responses
  if (!RspInLmem.empty()) {
    auto& out_rsp = RspInLmem.peek();
    if (RspOut.try_send(out_rsp, 1)) {
      DT(4, this->name() << " lmem-rsp: " << out_rsp);
      RspInLmem.pop();
    }
  }
  if (!RspInDC.empty()) {
    auto& out_rsp = RspInDC.peek();
    if (RspOut.try_send(out_rsp, 1)) {
      DT(4, this->name() << " dc-rsp: " << out_rsp);
      RspInDC.pop();
    }
  }
#endif

  // process incoming requests
  if (!ReqIn.empty()) {
    auto& in_req = ReqIn.peek();

    LsuReq out_dc_req(in_req.mask.size());
    out_dc_req.tag   = in_req.tag;
    out_dc_req.cid   = in_req.cid;
    out_dc_req.uuid  = in_req.uuid;
    out_dc_req.wid   = in_req.wid;
    // Op + per-lane tids carry through. Adapter recovers hart_id via
    // make_hart_id(cid, wid, tids[i]) at the dcache boundary.
    out_dc_req.op    = in_req.op;
    out_dc_req.flags = in_req.flags;
    out_dc_req.tids  = in_req.tids;

    LsuReq out_lmem_req(in_req.mask.size());
    out_lmem_req.tag   = in_req.tag;
    out_lmem_req.cid   = in_req.cid;
    out_lmem_req.uuid  = in_req.uuid;
    out_lmem_req.wid   = in_req.wid;
    // Same fields as the dcache branch: the LMEM banks implement AMOs, so
    // amo_unsigned decides the MIN/MAX compare style, and the per-lane tids
    // are what give each lane a distinct hart_id for LR/SC reservations.
    out_lmem_req.op    = in_req.op;
    out_lmem_req.flags = in_req.flags;
    out_lmem_req.tids  = in_req.tids;

#ifdef VX_CFG_EXT_DLMEM_ENABLE
    LsuReq out_remote_req(in_req.mask.size());
    out_remote_req.tag    = in_req.tag;
    out_remote_req.cid    = in_req.cid;
    out_remote_req.uuid   = in_req.uuid;
    out_remote_req.wid    = in_req.wid;
    out_remote_req.op     = in_req.op;
    out_remote_req.flags  = in_req.flags;
    out_remote_req.tids   = in_req.tids;
    out_remote_req.cta_id = in_req.cta_id;
#endif

    for (uint32_t i = 0; i < in_req.mask.size(); ++i) {
      if (in_req.mask.test(i)) {
#ifdef VX_CFG_EXT_DLMEM_ENABLE
        // A remote lane keeps its window address, which must not be taken for
        // this core's own memory.
        if (in_req.dlmem_remote.test(i)) {
          out_remote_req.mask.set(i);
          out_remote_req.dlmem_remote.set(i);
          out_remote_req.addrs.at(i)      = in_req.addrs.at(i);
          out_remote_req.data.at(i)       = in_req.data.at(i);
          out_remote_req.byteen.at(i)     = in_req.byteen.at(i);
          out_remote_req.amo_cmp.at(i)    = in_req.amo_cmp.at(i);
          out_remote_req.dlmem_core.at(i) = in_req.dlmem_core.at(i);
          continue;
        }
#endif
        auto type = get_addr_type(in_req.addrs.at(i));
        // An AMO whose address is Shared is executed by the LMEM bank here.
        // The RTL has no such path -- VX_lmem_switch strips the AMO bits and
        // asserts -- so the two models diverge on this op until the hardware
        // path lands, and a Shared AMO cannot be run on FPGA.
        if (type == AddrType::Shared) {
          out_lmem_req.mask.set(i);
          out_lmem_req.addrs.at(i)  = in_req.addrs.at(i);
          out_lmem_req.data.at(i)   = in_req.data.at(i);
          out_lmem_req.byteen.at(i) = in_req.byteen.at(i);
#ifdef VX_CFG_EXT_DLMEM_ENABLE
          // A window CAS to a same-core peer executes in this core's local memory.
          out_lmem_req.amo_cmp.at(i) = in_req.amo_cmp.at(i);
#endif
        } else {
          out_dc_req.mask.set(i);
          out_dc_req.addrs.at(i)  = in_req.addrs.at(i);
          out_dc_req.data.at(i)   = in_req.data.at(i);
          out_dc_req.byteen.at(i) = in_req.byteen.at(i);
        }
      }
    }

    bool send_to_dc = !out_dc_req.mask.none();
    bool send_to_lmem = !out_lmem_req.mask.none();

    // check DC backpressure
    if (send_to_dc && ReqOutDC.full())
      return; // stall

    // check LMem backpressure
    if (send_to_lmem && ReqOutLmem.full())
      return; // stall

#ifdef VX_CFG_EXT_DLMEM_ENABLE
    bool send_to_remote = !out_remote_req.mask.none();
    if (send_to_remote && ReqOutRemote.full())
      return; // stall
#endif

    if (send_to_dc) {
      ReqOutDC.send(out_dc_req, delay_);
      DT(4, this->name() << " dc-req: " << out_dc_req);
    }

    if (send_to_lmem) {
      ReqOutLmem.send(out_lmem_req, delay_);
      DT(4, this->name() << " lmem-req: " << out_lmem_req);
    }

#ifdef VX_CFG_EXT_DLMEM_ENABLE
    if (send_to_remote) {
      ReqOutRemote.send(out_remote_req, delay_);
      DT(4, this->name() << " remote-req: " << out_remote_req);
    }
#endif
    ReqIn.pop();
  }

  // sleep when nothing is queued or in flight toward us; no state moves while idle.
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  if (ReqIn.size() == 0 && RspInLmem.size() == 0 && RspInDC.size() == 0 && RspInRemote.size() == 0) {
    this->tick_sleep();
  }
#else
  if (ReqIn.size() == 0 && RspInLmem.size() == 0 && RspInDC.size() == 0) {
    this->tick_sleep();
  }
#endif
}
