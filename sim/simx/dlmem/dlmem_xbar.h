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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "types.h"
#include "dlmem_switch.h"

namespace vortex {

// Output port (destination socket) of a packet to core `dst` in the crossbar.
uint32_t dlmem_xbar_route(uint32_t dst, uint32_t socket_size, uint32_t num_sockets);

// VX_dlmem_xbar, one port per socket. Each output feeds, in the same cycle,
// the sender-owned slice dlmem_connect() adds as its register, gated by
// credits the destination switch returns one cycle after each pop.
template <typename Pkt>
class DlmemXbar : public SimObject<DlmemXbar<Pkt>> {
public:
  using Ptr = std::shared_ptr<DlmemXbar<Pkt>>;

  std::vector<SimChannel<Pkt>> In;
  std::vector<SimChannel<Pkt>> Out;
  // Carries the index of the output whose downstream slot was freed.
  SimEventLink<uint32_t> CreditIn;
  std::vector<SimEventLink<uint32_t>> CreditOut;

  DlmemXbar(const SimContext& ctx,
            const char* name,
            uint32_t socket_size,
            uint32_t num_sockets)
    : SimObject<DlmemXbar<Pkt>>(ctx, name)
    , In(num_sockets, this)
    , Out(num_sockets, this)
    , CreditIn(this)
    , CreditOut(num_sockets, SimEventLink<uint32_t>(this))
    , socket_size_(socket_size)
    , num_sockets_(num_sockets)
    , arbiter_(num_sockets, num_sockets)
    , route_(num_sockets)
    , ready_(num_sockets)
    , grant_(num_sockets)
    , credits_(num_sockets, DLMEM_BUF_DEPTH)
    , collisions_(0) {
    __assert(num_sockets > 1, "a single-socket cluster has no crossbar");
    CreditIn.bind(this, &DlmemXbar<Pkt>::on_credit);
  }

  // Free slots behind output `port`; DLMEM_BUF_DEPTH once drained.
  uint32_t credits(uint32_t port) const {
    return credits_.at(port);
  }

  uint64_t collisions() const {
    return collisions_;
  }

protected:
  void on_reset() {
    arbiter_.reset();
    std::fill(credits_.begin(), credits_.end(), DLMEM_BUF_DEPTH);
    collisions_ = 0;
  }

  void on_tick() {
    uint32_t N = num_sockets_;
    bool busy = false;
    for (uint32_t i = 0; i < N; ++i) {
      auto& in = In.at(i);
      busy |= (in.size() != 0);
      route_[i] = in.empty() ? -1
                : int32_t(dlmem_xbar_route(in.peek().dst, socket_size_, N));
    }
    for (uint32_t o = 0; o < N; ++o) {
      ready_[o] = (credits_[o] != 0);
    }
    collisions_ += arbiter_.arbitrate(route_, ready_, grant_);
    for (uint32_t o = 0; o < N; ++o) {
      if (grant_[o] < 0) {
        continue;
      }
      uint32_t i = grant_[o];
      auto& in = In.at(i);
      Pkt pkt = in.peek();
      in.pop();
      CreditOut.at(i).send(0);
      --credits_[o];
      Out.at(o).send(pkt, 0);
      DT(4, this->name() << " port" << i << "->" << o << " dst=" << pkt.dst);
    }
    if (!busy) {
      this->tick_sleep();
    }
  }

private:
  void on_credit(const uint32_t& port) {
    __assert(credits_.at(port) < DLMEM_BUF_DEPTH, "crossbar credit overflow");
    ++credits_.at(port);
  }

  uint32_t socket_size_;
  uint32_t num_sockets_;
  DlmemArbiter arbiter_;
  std::vector<int32_t> route_;
  std::vector<bool> ready_;
  std::vector<int32_t> grant_;
  std::vector<uint32_t> credits_;
  uint64_t collisions_;

  friend class SimObject<DlmemXbar<Pkt>>;
};

// Each direction crosses the socket/cluster domain boundary through a slice
// owned by its sender. Created after both nodes, a slice ticks after the node
// feeding it, so the hop costs exactly the slice's one-cycle register.
template <typename Pkt>
void dlmem_connect(DlmemSocketSwitch<Pkt>* sw, DlmemXbar<Pkt>* xbar, uint32_t port) {
  __assert(sw->has_uplink() && sw->socket_id() == port, "switch/crossbar port mismatch");
  std::string up_name = sw->name() + "-up";
  std::string dn_name = xbar->name() + "-dn" + std::to_string(port);
  typename RegSlice<Pkt>::Ptr up;
  typename RegSlice<Pkt>::Ptr dn;
  {
    SimPlatform::DomainScope scope(sw);
    up = RegSlice<Pkt>::Create(up_name.c_str(), 1);
  }
  {
    SimPlatform::DomainScope scope(xbar);
    dn = RegSlice<Pkt>::Create(dn_name.c_str(), 1);
  }
  sw->UplinkOut.bind(&up->In);
  up->Out.bind(&xbar->In.at(port));
  xbar->Out.at(port).bind(&dn->In);
  dn->Out.bind(&sw->UplinkIn);
  xbar->CreditOut.at(port).bind(&sw->UplinkCreditIn);
  sw->UplinkCreditOut.bind(&xbar->CreditIn);
}

} // namespace vortex
