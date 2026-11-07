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

#pragma once

#include <cstdint>
#include <ostream>

#include "dxa_meta.h"

// Bulk async-groups are tracked per warp: DXA_BG_GROUPS committed groups not
// yet complete, and the open group's sequence number tagging each operation.

namespace vortex {

// One per DXA instruction of a warp that the groups see, at its send.
struct DxaBgEvt {
  enum Kind : uint32_t {
    OP        = 0,   // a bulk_group operation joins the open group
    COMMIT    = 1,
    WAIT      = 2,   // wait_group n
    WAIT_READ = 3,   // wait_group.read n
  };
  uint32_t wid  = 0;
  Kind     kind = OP;
  uint32_t n    = 0;   // the wait's N, 0 for OP and COMMIT
};

// A bulk_group operation's progress, from the socket's engine back to its
// core. One operation reports exactly one read_done and one write_done,
// possibly in the same event.
struct DxaBgCompl {
  uint32_t wid        = 0;
  uint32_t seq        = 0;
  bool     read_done  = false;   // its last local-memory source read returned
  bool     write_done = false;   // its writes are performed
};

// A warp's group state, sent to the DXA unit whenever it changes.
struct DxaBgState {
  uint32_t wid          = 0;
  uint32_t seq          = 0;       // the open group's sequence
  bool     stall_op     = false;   // the open group is full
  bool     stall_commit = false;   // DXA_BG_GROUPS committed groups pending
};

inline std::ostream& operator<<(std::ostream& os, const DxaBgEvt& e) {
  static const char* const kinds[] = {"op", "commit", "wait", "wait.read"};
  return os << "{wid=" << e.wid << ", " << kinds[e.kind & 3u] << ", n=" << e.n << "}";
}

inline std::ostream& operator<<(std::ostream& os, const DxaBgCompl& c) {
  return os << "{wid=" << c.wid << ", seq=" << c.seq << ", rd=" << c.read_done
            << ", wr=" << c.write_done << "}";
}

inline std::ostream& operator<<(std::ostream& os, const DxaBgState& s) {
  return os << "{wid=" << s.wid << ", seq=" << s.seq << ", stall_op=" << s.stall_op
            << ", stall_commit=" << s.stall_commit << "}";
}

} // namespace vortex
