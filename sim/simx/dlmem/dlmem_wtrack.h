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

// VX_dlmem_wtrack: per-warp count of remote stores not yet acknowledged.
// Increments come from each LSU block's translation, decrements from the
// egress acks; drained(w) is the registered cnt[w] == 0 of the cycle after.
class DlmemWTrack : public SimObject<DlmemWTrack> {
public:
  using Ptr = std::shared_ptr<DlmemWTrack>;

  std::vector<SimChannel<WtInc>> IncIn;
  SimChannel<uint32_t> DecIn;

  DlmemWTrack(const SimContext& ctx,
              const char* name,
              uint32_t num_inc,
              uint32_t num_warps,
              uint32_t max_count);

  bool drained(uint32_t wid) const {
    return drained_.at(wid);
  }

  bool empty() const;

  uint32_t count(uint32_t wid) const {
    return cnt_.at(wid);
  }

protected:
  void on_reset();
  void on_tick();

private:
  uint32_t max_count_;
  std::vector<uint32_t> cnt_;
  std::vector<bool> drained_;

  friend class SimObject<DlmemWTrack>;
};

} // namespace vortex
