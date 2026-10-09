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

// Golden model of DLMEM gang placement and aperture translation for SimX and the testbenches.
// Shapes come from Cfg, never VX_config.h, so one binary can sweep them.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace vortex {
namespace dlmem_ref {

struct Cfg {
  uint32_t ncc;          // cores of one hardware cluster (NUM_SOCKETS * SOCKET_SIZE)
  uint32_t nw;           // warps per core, also its CTA slots
  uint32_t nt;           // threads per warp
  uint32_t max_cluster;  // VX_CFG_DLMEM_MAX_CLUSTER
  uint32_t lmem_log;     // log2 of a core's local memory size
  uint32_t num_gangs;    // gang-id pool size
  uint32_t socket_size;
  uint32_t num_sockets;
  uint64_t dlmem_base;   // VX_MEM_DLMEM_BASE_ADDR
  uint64_t lmem_base;    // VX_MEM_LMEM_BASE_ADDR
};

struct Peer {
  uint32_t core;
  uint32_t slot;
};

struct Row {
  uint32_t k;
  std::vector<Peer> peers;  // peers.size() == k, indexed by rank
};

// The golden model checks its invariants in release builds too.
inline void check(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "dlmem_ref: %s\n", what);
    std::abort();
  }
}

// Same as the RTL `CLOG2: clog2(0) == clog2(1) == 0.
inline uint32_t clog2(uint64_t x) {
  uint32_t n = 0;
  while ((uint64_t(1) << n) < x) {
    ++n;
  }
  return n;
}

inline uint64_t low_mask(uint32_t bits) {
  return (bits >= 64) ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1);
}

inline uint32_t warps_of(uint32_t block_size, uint32_t nt) {
  return (block_size + nt - 1) / nt;
}

// Identical to VX_cta_dispatch's occupancy bound.
inline uint32_t usable_slots(const Cfg& cfg, uint32_t stride) {
  if (stride == 0)
    return cfg.nw;
  uint32_t u = 1;
  for (uint32_t m = 1; m <= cfg.nw; ++m) {
    if (uint64_t(m) * stride <= (uint64_t(1) << cfg.lmem_log))
      u = m;
  }
  return u;
}

// Field-wise, so it holds for any SOCKET_SIZE (KMU_DEST_LSB_*).
inline uint32_t kmu_dest(const Cfg& cfg, uint32_t cluster_id, uint32_t core) {
  return (((cluster_id << clog2(cfg.num_sockets)) | (core / cfg.socket_size)) << clog2(cfg.socket_size))
       | (core % cfg.socket_size);
}

///////////////////////////////////////////////////////////////////////////////
// Placement

struct Mirror {
  std::vector<uint64_t> free;  // free[c] bit s: slot s of core c is free
  std::vector<uint32_t> used;  // used[c]: warps reserved on core c
  uint32_t rr;
};

inline void reset(const Cfg& cfg, Mirror& m) {
  m.free.assign(cfg.ncc, low_mask(cfg.nw));
  m.used.assign(cfg.ncc, 0);
  m.rr = 0;
}

// All-or-nothing, spread first: a round gives each member a distinct core, and a new
// round starts only once every capable core holds one. On failure m is unchanged.
inline bool place(const Cfg& cfg, Mirror& m, uint32_t K, uint32_t W, uint32_t U, std::vector<Peer>& out) {
  check(K >= 1, "place: empty gang");
  Mirror s = m;
  std::vector<bool> visited(cfg.ncc, false);
  uint32_t start = m.rr;
  uint64_t umask = low_mask(U);
  out.clear();
  for (uint32_t r = 0; r < K; ++r) {
    std::vector<bool> capable(cfg.ncc, false);
    bool any_capable = false;
    for (uint32_t c = 0; c < cfg.ncc; ++c) {
      check(s.used[c] <= cfg.nw, "place: core over-reserved");
      capable[c] = ((s.free[c] & umask) != 0) && (cfg.nw - s.used[c] >= W);
      any_capable = any_capable || capable[c];
    }
    if (!any_capable) {
      out.clear();
      return false;
    }
    std::vector<bool> cand(cfg.ncc, false);
    bool any_cand = false;
    for (uint32_t c = 0; c < cfg.ncc; ++c) {
      cand[c] = capable[c] && !visited[c];
      any_cand = any_cand || cand[c];
    }
    if (!any_cand) {
      visited.assign(cfg.ncc, false);
      cand = capable;
    }
    uint32_t c = 0;
    for (uint32_t i = 0; i < cfg.ncc; ++i) {
      c = (start + i) % cfg.ncc;
      if (cand[c])
        break;
    }
    uint64_t avail = s.free[c] & umask;
    uint32_t sl = 0;
    while (((avail >> sl) & 1) == 0) {
      ++sl;
    }
    out.push_back({c, sl});
    s.free[c] &= ~(uint64_t(1) << sl);
    s.used[c] += W;
    visited[c] = true;
  }
  s.rr = (out[K - 1].core + 1) % cfg.ncc;
  m = s;
  return true;
}

inline void release(Mirror& m, uint32_t core, uint32_t slot, uint32_t W) {
  check(((m.free.at(core) >> slot) & 1) == 0, "release: slot already free");
  check(m.used.at(core) >= W, "release: warp count underflow");
  m.free[core] |= uint64_t(1) << slot;
  m.used[core] -= W;
}

struct GangIds {
  std::vector<bool>     live;
  std::vector<uint32_t> cnt;    // done events still expected
  std::vector<uint8_t>  epoch;
};

inline void reset(const Cfg& cfg, GangIds& g) {
  g.live.assign(cfg.num_gangs, false);
  g.cnt.assign(cfg.num_gangs, 0);
  g.epoch.assign(cfg.num_gangs, 0);
}

// Lowest free id, or -1 when the pool is exhausted.
inline int alloc(GangIds& g, uint32_t K) {
  for (size_t i = 0; i < g.live.size(); ++i) {
    if (!g.live[i]) {
      g.live[i] = true;
      g.cnt[i] = K;
      g.epoch[i] ^= 1;
      return int(i);
    }
  }
  return -1;
}

inline void done(GangIds& g, uint32_t id, uint32_t ep) {
  check(id < g.live.size() && g.live[id], "done: gang id not live");
  check(g.epoch[id] == ep, "done: stale epoch");
  check(g.cnt[id] > 0, "done: too many done events");
  if (--g.cnt[id] == 0)
    g.live[id] = false;
}

// A launch starts with nothing in flight, so rr and the epochs restart.
inline void new_launch(const Cfg& cfg, Mirror& m, GangIds& g) {
  for (uint32_t c = 0; c < cfg.ncc; ++c) {
    check(m.free.at(c) == low_mask(cfg.nw) && m.used.at(c) == 0, "new_launch: mirror not empty");
  }
  for (bool l : g.live) {
    check(!l, "new_launch: gang id still live");
  }
  m.rr = 0;
  g.epoch.assign(g.epoch.size(), 0);
}

// A same-core cluster's row: its members sit in consecutive slots of one core.
inline Row legacy_row(uint32_t my_core, uint32_t base_slot, uint32_t K, uint32_t max_cluster) {
  Row row;
  row.k = (K < max_cluster) ? K : max_cluster;
  for (uint32_t r = 0; r < row.k; ++r) {
    row.peers.push_back({my_core, base_slot + r});
  }
  return row;
}

///////////////////////////////////////////////////////////////////////////////
// Translation (byte addresses)

constexpr uint32_t DLMEM_OP_LD  = 0;
constexpr uint32_t DLMEM_OP_ST  = 1;
constexpr uint32_t DLMEM_OP_AMO = 2;

enum Kind { NONE, LOCAL, REMOTE, OOB };

struct Xlate {
  Kind     kind;
  uint32_t core;
  uint64_t lmem_byte;
  uint64_t out_addr;
};

// A peer-row write from the dispatcher to the translation table.
struct RowWrite {
  uint32_t slot;
  Row      row;
  uint32_t stride;
};

// The window decodes a power-of-two rank field: 2^(lmem_log + rank_bits) bytes.
inline uint32_t rank_bits(const Cfg& cfg) {
  return clog2(cfg.max_cluster);
}

// The encoding the egress decodes.
inline uint64_t remote_addr(const Cfg& cfg, uint32_t core, uint64_t lb) {
  return cfg.dlmem_base | (uint64_t(core) << cfg.lmem_log) | lb;
}

inline void decode_remote(const Cfg& cfg, uint64_t a, uint32_t* core, uint64_t* lb) {
  *core = uint32_t((a >> cfg.lmem_log) & low_mask(clog2(cfg.ncc)));
  *lb   = a & low_mask(cfg.lmem_log);
}

inline Xlate translate(const Cfg& cfg, uint64_t addr, const Row& row,
                       uint32_t my_core, uint32_t my_slot, uint32_t stride) {
  uint32_t wl = cfg.lmem_log + rank_bits(cfg);
  if ((addr >> wl) != (cfg.dlmem_base >> wl))
    return {NONE, 0, 0, addr};
  uint32_t rank = uint32_t((addr >> cfg.lmem_log) & low_mask(rank_bits(cfg)));
  uint64_t off  = addr & low_mask(cfg.lmem_log);
  // stride == 0 makes every window access out of bounds. The line offset is
  // kept so the lane's byteen and line-relative data still match its address;
  // DLMEM requires 64-byte memory blocks, so it stays inside the issuer's slot.
  if (rank >= row.k || off >= stride) {
    uint64_t lb = uint64_t(my_slot) * stride + (addr & 63);
    return {OOB, my_core, lb, cfg.lmem_base + lb};
  }
  const Peer& p = row.peers.at(rank);
  uint64_t lb = uint64_t(p.slot) * stride + off;
  if (p.core == my_core)
    return {LOCAL, p.core, lb, cfg.lmem_base + lb};
  return {REMOTE, p.core, lb, remote_addr(cfg, p.core, lb)};
}

} // namespace dlmem_ref
} // namespace vortex
