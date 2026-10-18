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
                         uint32_t max_count,
                         uint32_t num_commit)
  : SimObject<DlmemWTrack>(ctx, name)
  , IncIn(num_inc, this)
  , DecIn(this)
  , DoneIn(this)
  , CommitIn(num_commit, this)
  , max_count_(max_count)
  , cnt_(num_warps, 0)
  , drained_(num_warps, true) {
  __assert(num_inc != 0 && num_warps != 0, this->name() << ": empty tracker");
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
  // Increments and the decrements of one cycle land together, as cnt_n does.
  std::vector<uint64_t> inc(NW, 0);
  for (auto& in : IncIn) {
    if (in.empty()) {
      continue;
    }
    auto& w = in.peek();
    __assert(w.wid < NW, this->name() << ": warp id " << w.wid << " out of range");
    inc[w.wid] += w.cnt;
    in.pop();
  }
  std::vector<uint64_t> dec(NW, 0);
  bool acked = false;
  if (!DecIn.empty()) {
    uint32_t wid = DecIn.peek();
    __assert(wid < NW, this->name() << ": ack warp id " << wid << " out of range");
    ++dec[wid];
    acked = true;
    DecIn.pop();
  }
  if (!DoneIn.empty()) {
    __assert(!acked, this->name() << ": an ack and a done in one cycle");
    uint32_t wid = DoneIn.peek();
    __assert(wid < NW, this->name() << ": done warp id " << wid << " out of range");
    ++dec[wid];
    DoneIn.pop();
  }
  for (auto& in : CommitIn) {
    if (in.empty()) {
      continue;
    }
    uint32_t wid = in.peek();
    __assert(wid < NW, this->name() << ": commit warp id " << wid << " out of range");
    ++dec[wid];
    in.pop();
  }
  for (uint32_t w = 0; w < NW; ++w) {
    uint64_t n = cnt_[w] + inc[w];
    __assert(n <= max_count_, this->name() << ": counter overflow, wid=" << w << " count=" << n);
    __assert(dec[w] <= n, this->name() << ": completion for a warp with no access outstanding, wid=" << w
             << " count=" << n << " completions=" << dec[w]);
    n -= dec[w];
    cnt_[w] = uint32_t(n);
    drained_[w] = (n == 0);
  }
  bool idle = (DecIn.size() == 0) && (DoneIn.size() == 0);
  for (auto& in : IncIn) {
    idle = idle && (in.size() == 0);
  }
  for (auto& in : CommitIn) {
    idle = idle && (in.size() == 0);
  }
  if (idle) {
    this->tick_sleep();
  }
}
