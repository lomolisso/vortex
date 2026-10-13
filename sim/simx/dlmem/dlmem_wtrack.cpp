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

#include "dlmem_wtrack.h"

using namespace vortex;

DlmemWTrack::DlmemWTrack(const SimContext& ctx,
                         const char* name,
                         uint32_t num_inc,
                         uint32_t num_warps,
                         uint32_t max_count)
  : SimObject<DlmemWTrack>(ctx, name)
  , IncIn(num_inc, this)
  , DecIn(this)
  , DoneIn(this)
  , max_count_(max_count)
  , cnt_(num_warps, 0)
  , drained_(num_warps, true) {
  __assert(num_inc != 0 && num_warps != 0, "empty store tracker");
}

void DlmemWTrack::on_reset() {
  std::fill(cnt_.begin(), cnt_.end(), 0);
  std::fill(drained_.begin(), drained_.end(), true);
}

bool DlmemWTrack::empty() const {
  for (bool d : drained_) {
    if (!d) {
      return false;
    }
  }
  return true;
}

void DlmemWTrack::on_tick() {
  uint32_t NW = cnt_.size();
  // Increments and the decrement of one cycle land together, as cnt_n does.
  std::vector<uint64_t> inc(NW, 0);
  for (auto& in : IncIn) {
    if (in.empty()) {
      continue;
    }
    auto& w = in.peek();
    __assert(w.wid < NW, "store tracker: warp id out of range");
    inc[w.wid] += w.cnt;
    in.pop();
  }
  int32_t dec_wid = -1;
  if (!DecIn.empty()) {
    dec_wid = DecIn.peek();
    __assert(uint32_t(dec_wid) < NW, "store tracker: ack warp id out of range");
    DecIn.pop();
  }
  if (!DoneIn.empty()) {
    __assert(dec_wid < 0, "store tracker: an ack and a done in one cycle");
    dec_wid = DoneIn.peek();
    __assert(uint32_t(dec_wid) < NW, "store tracker: done warp id out of range");
    DoneIn.pop();
  }
  for (uint32_t w = 0; w < NW; ++w) {
    uint64_t n = cnt_[w] + inc[w];
    __assert(n <= max_count_, "store tracker: counter overflow");
    if (int32_t(w) == dec_wid) {
      __assert(n != 0, "store tracker: completion for a warp with no remote access");
      --n;
    }
    cnt_[w] = uint32_t(n);
    drained_[w] = (n == 0);
  }
  bool idle = (DecIn.size() == 0) && (DoneIn.size() == 0);
  for (auto& in : IncIn) {
    idle = idle && (in.size() == 0);
  }
  if (idle) {
    this->tick_sleep();
  }
}
