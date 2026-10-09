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
#include <vector>
#include "types.h"

namespace vortex {

// Slots of a registered output buffer (VX_elastic_buffer at OUT_BUF=3).
constexpr uint32_t DLMEM_BUF_DEPTH = 2;

// VX_stream_xbar's per-output round-robin (ARBITER "R"): the scan starts one
// past the last winner, and the pointer moves only when the output fires.
class DlmemArbiter {
public:
  DlmemArbiter(uint32_t num_inputs, uint32_t num_outputs);

  void reset();

  // route[i]: output the head of input i requests, -1 when input i is empty.
  // ready[o]: output o has a free slot this cycle.
  // grant[o]: winning input of output o, -1 when output o does not fire.
  // Returns this cycle's VX_stream_xbar collision count.
  uint32_t arbitrate(const std::vector<int32_t>& route,
                     const std::vector<bool>& ready,
                     std::vector<int32_t>& grant);

private:
  uint32_t num_inputs_;
  uint32_t num_outputs_;
  std::vector<uint32_t> rr_next_;
};

// Port map of one socket switch: core ports 0..S-1, then the uplink (port S)
// when the cluster has more than one socket.
class DlmemSwitchMap {
public:
  DlmemSwitchMap(uint32_t socket_id, uint32_t socket_size, uint32_t num_sockets);

  uint32_t num_ports() const {
    return socket_size_ + (this->has_uplink() ? 1 : 0);
  }

  bool has_uplink() const {
    return num_sockets_ > 1;
  }

  uint32_t uplink() const {
    return socket_size_;
  }

  // Output port for a packet to core `dst` arriving on port `input`.
  uint32_t route(uint32_t input, uint32_t dst) const;

  uint32_t socket_id() const {
    return socket_id_;
  }

private:
  uint32_t socket_id_;
  uint32_t socket_size_;
  uint32_t num_sockets_;
};

// VX_dlmem_socket_switch; Pkt carries its destination core in `dst`. Each
// output is a two-slot buffer with a registered ready, as OUT_BUF=3: a slot
// freed by a pop is reusable the next cycle. Core outputs register their send
// and check their endpoint, so the core side must pop after this tick. The
// uplink's register is the sender-owned slice dlmem_connect() adds; it is fed
// in the same cycle and gated by credits the crossbar returns.
template <typename Pkt>
class DlmemSocketSwitch : public SimObject<DlmemSocketSwitch<Pkt>> {
public:
  using Ptr = std::shared_ptr<DlmemSocketSwitch<Pkt>>;

  std::vector<SimChannel<Pkt>> CoreIn;
  std::vector<SimChannel<Pkt>> CoreOut;
  SimChannel<Pkt> UplinkIn;
  SimChannel<Pkt> UplinkOut;
  SimEventLink<uint32_t> UplinkCreditIn;
  SimEventLink<uint32_t> UplinkCreditOut;

  DlmemSocketSwitch(const SimContext& ctx,
                    const char* name,
                    uint32_t socket_id,
                    uint32_t socket_size,
                    uint32_t num_sockets)
    : SimObject<DlmemSocketSwitch<Pkt>>(ctx, name)
    , CoreIn(socket_size, this)
    , CoreOut(socket_size, this)
    , UplinkIn(this)
    , UplinkOut(this)
    , UplinkCreditIn(this)
    , UplinkCreditOut(this)
    , map_(socket_id, socket_size, num_sockets)
    , arbiter_(map_.num_ports(), map_.num_ports())
    , route_(map_.num_ports())
    , ready_(map_.num_ports())
    , grant_(map_.num_ports())
    , uplink_credits_(DLMEM_BUF_DEPTH)
    , collisions_(0) {
    UplinkCreditIn.bind(this, &DlmemSocketSwitch<Pkt>::on_uplink_credit);
  }

  uint32_t socket_id() const {
    return map_.socket_id();
  }

  bool has_uplink() const {
    return map_.has_uplink();
  }

  // Free uplink slots; DLMEM_BUF_DEPTH once the network has drained.
  uint32_t uplink_credits() const {
    return uplink_credits_;
  }

  uint64_t collisions() const {
    return collisions_;
  }

protected:
  void on_reset() {
    arbiter_.reset();
    uplink_credits_ = DLMEM_BUF_DEPTH;
    collisions_ = 0;
  }

  void on_tick() {
    uint32_t P = map_.num_ports();
    uint32_t S = CoreIn.size();
    bool busy = false;
    for (uint32_t i = 0; i < P; ++i) {
      auto& in = this->input(i);
      busy |= (in.size() != 0);
      route_[i] = in.empty() ? -1 : int32_t(map_.route(i, in.peek().dst));
    }
    for (uint32_t o = 0; o < P; ++o) {
      ready_[o] = (o < S) ? !CoreOut.at(o).full() : (uplink_credits_ != 0);
    }
    collisions_ += arbiter_.arbitrate(route_, ready_, grant_);
    for (uint32_t o = 0; o < P; ++o) {
      if (grant_[o] < 0) {
        continue;
      }
      uint32_t i = grant_[o];
      auto& in = this->input(i);
      Pkt pkt = in.peek();
      in.pop();
      if (i == S) {
        UplinkCreditOut.send(map_.socket_id());
      }
      if (o < S) {
        CoreOut.at(o).send(pkt, 1);
      } else {
        --uplink_credits_;
        UplinkOut.send(pkt, 0);
      }
      DT(4, this->name() << " port" << i << "->" << o << " dst=" << pkt.dst);
    }
    if (!busy) {
      this->tick_sleep();
    }
  }

private:
  SimChannel<Pkt>& input(uint32_t i) {
    return (i < CoreIn.size()) ? CoreIn.at(i) : UplinkIn;
  }

  void on_uplink_credit(const uint32_t&) {
    __assert(uplink_credits_ < DLMEM_BUF_DEPTH, "uplink credit overflow");
    ++uplink_credits_;
  }

  DlmemSwitchMap map_;
  DlmemArbiter arbiter_;
  std::vector<int32_t> route_;
  std::vector<bool> ready_;
  std::vector<int32_t> grant_;
  uint32_t uplink_credits_;
  uint64_t collisions_;

  friend class SimObject<DlmemSocketSwitch<Pkt>>;
};

} // namespace vortex
