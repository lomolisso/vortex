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

#include "dlmem_switch.h"

using namespace vortex;

DlmemArbiter::DlmemArbiter(uint32_t num_inputs, uint32_t num_outputs)
  : num_inputs_(num_inputs)
  , num_outputs_(num_outputs)
  , rr_next_(num_outputs, 0) {
  __assert(num_inputs != 0 && num_outputs != 0, "empty arbiter");
}

void DlmemArbiter::reset() {
  // VX_rr_arbiter resets its last grant to NUM_REQS-1, so input 0 wins first.
  std::fill(rr_next_.begin(), rr_next_.end(), 0);
}

uint32_t DlmemArbiter::arbitrate(const std::vector<int32_t>& route,
                                 const std::vector<bool>& ready,
                                 std::vector<int32_t>& grant) {
  uint32_t I = num_inputs_;
  uint32_t collisions = 0;
  for (uint32_t o = 0; o < num_outputs_; ++o) {
    grant[o] = -1;
    if (!ready[o]) {
      continue;
    }
    for (uint32_t k = 0; k < I; ++k) {
      uint32_t i = (rr_next_[o] + k) % I;
      if (route[i] == int32_t(o)) {
        grant[o] = i;
        break;
      }
    }
    if (grant[o] < 0) {
      continue;
    }
    uint32_t w = grant[o];
    rr_next_[o] = (w + 1) % I;
    // VX_stream_xbar counts input i when a later input j targets the same
    // output and either of the two fires.
    bool later = false;
    for (uint32_t i = 0; i < I; ++i) {
      if (route[i] != int32_t(o)) {
        continue;
      }
      if (i < w) {
        ++collisions;
      } else if (i > w) {
        later = true;
      }
    }
    collisions += later ? 1 : 0;
  }
  return collisions;
}

DlmemSwitchMap::DlmemSwitchMap(uint32_t socket_id, uint32_t socket_size, uint32_t num_sockets)
  : socket_id_(socket_id)
  , socket_size_(socket_size)
  , num_sockets_(num_sockets) {
  __assert(socket_size != 0 && num_sockets != 0, "empty DLMEM network");
  __assert(socket_id < num_sockets, "socket index out of range");
}

uint32_t DlmemSwitchMap::route(uint32_t input, uint32_t dst) const {
  __assert(dst < socket_size_ * num_sockets_, "destination core out of range");
  uint32_t socket = dst / socket_size_;
  uint32_t port = dst % socket_size_;
  if (input == this->uplink() && this->has_uplink()) {
    __assert(socket == socket_id_, "uplink packet for another socket");
    return port;
  }
  return (socket == socket_id_) ? port : this->uplink();
}
