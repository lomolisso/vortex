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
#include <deque>
#include <memory>
#include <vector>
#include "types.h"
#include "dlmem_ref.h"

namespace vortex {

// VX_dlmem_peer_tbl plus one VX_dlmem_xlate per LSU block: a single registered
// stage that rewrites the lanes of a request falling in the DLMEM window.
// As in the RTL, a request keeps the peer row current when it entered, since
// its slot may be re-admitted while a committed store waits here; the stride
// is read when it leaves.
class DlmemXlate : public SimObject<DlmemXlate> {
public:
  using Ptr = std::shared_ptr<DlmemXlate>;

  std::vector<SimChannel<LsuReq>> ReqIn;    // capacity 2: the RTL skid buffer
  std::vector<SimChannel<LsuReq>> ReqOut;
  SimChannel<dlmem_ref::RowWrite> PeerWrIn;
  std::vector<SimChannel<WtInc>>  WtIncOut; // remote plain stores, at the output fire

  DlmemXlate(const SimContext& ctx,
             const char* name,
             const dlmem_ref::Cfg& cfg,
             uint32_t core_id,
             uint32_t num_blocks,
             bool abort_on_oob = true,
             bool abort_on_remote = true);

  // No request held or in flight toward the stage.
  bool empty() const;

protected:
  void on_reset();
  void on_tick();

private:
  using RowPtr = std::shared_ptr<const dlmem_ref::Row>;

  uint32_t translate(LsuReq& req, const dlmem_ref::Row& row);

  dlmem_ref::Cfg cfg_;
  uint32_t my_core_;
  uint32_t num_blocks_;
  bool abort_on_oob_;
  bool abort_on_remote_;
  std::vector<RowPtr> rows_;
  std::vector<std::deque<RowPtr>> entry_rows_;  // per block, one per request in ReqIn
  uint32_t stride_;

  friend class SimObject<DlmemXlate>;
};

} // namespace vortex
