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

#include "ideal_transport.h"
#include "debug.h"

using namespace vortex;

namespace {
// Fixed delivery delay. A placeholder constant, not a modeling knob: real
// per-hop timing comes from the Omelet co-simulator (roadmap Task 4).
constexpr uint32_t kFixedDelay = 1;
}

IdealTransport::IdealTransport(const SimContext& ctx, const char* name, uint32_t num_cores)
  : SimObject<IdealTransport>(ctx, name)
  , ReqIn(num_cores, this)
  , RspOut(num_cores, this)
  , ReqOut(num_cores, this)
  , RspIn(num_cores, this)
  , num_cores_(num_cores)
  , lg2_cores_(log2ceil(num_cores))
  , req_rr_(num_cores, 0)
  , rsp_rr_(num_cores, 0)
{}

void IdealTransport::on_reset() {
  std::fill(req_rr_.begin(), req_rr_.end(), 0);
  std::fill(rsp_rr_.begin(), rsp_rr_.end(), 0);
}

void IdealTransport::on_tick() {
  uint32_t R = 1u << lg2_cores_;

  // Responses: LocalMem remote ports → requesting cores. The source core
  // index rides the low tag bits (appended on the request path below).
  for (uint32_t i = 0; i < num_cores_; ++i) {
    for (uint32_t k = 0; k < num_cores_; ++k) {
      uint32_t o = (rsp_rr_.at(i) + k) % num_cores_;
      auto& rsp_in = RspIn.at(o);
      if (rsp_in.empty())
        continue;
      uint32_t src = (lg2_cores_ != 0) ? (rsp_in.peek().tag & (R - 1)) : 0;
      if (src != i)
        continue;
      MemRsp out_rsp(rsp_in.peek());
      if (lg2_cores_ != 0) {
        out_rsp.tag >>= lg2_cores_;
      }
      if (RspOut.at(i).try_send(out_rsp, kFixedDelay)) {
        DT(4, this->name() << " rsp" << o << "→" << i << ": " << out_rsp);
        rsp_in.pop();
        rsp_rr_.at(i) = (o + 1) % num_cores_;
      }
      break; // one delivery per requester per cycle
    }
  }

  // Requests: core DSMEM egress → target core's LocalMem remote port.
  for (uint32_t o = 0; o < num_cores_; ++o) {
    for (uint32_t k = 0; k < num_cores_; ++k) {
      uint32_t i = (req_rr_.at(o) + k) % num_cores_;
      auto& req_in = ReqIn.at(i);
      if (req_in.empty())
        continue;
      if (req_in.peek().flags.dsmem_target != o)
        continue;
      MemReq out_req(req_in.peek());
      if (lg2_cores_ != 0) {
        out_req.tag = (out_req.tag << lg2_cores_) | i;
      }
      if (ReqOut.at(o).try_send(out_req, kFixedDelay)) {
        DT(4, this->name() << " req" << i << "→" << o << ": " << out_req);
        req_in.pop();
        req_rr_.at(o) = (i + 1) % num_cores_;
      }
      break; // one delivery per destination per cycle
    }
  }
}
