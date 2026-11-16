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

// In-GPC placement policies (VX_CFG_CLUSTER_PLACEMENT).
constexpr uint32_t PLACE_SPREAD = 0;  // round-robin from the core after the last placed one
constexpr uint32_t PLACE_H100   = 1;  // least-occupied core first, then the lowest in-GPC index

struct Cfg {
  uint32_t ncc;          // cores of one hardware cluster (NUM_SOCKETS * SOCKET_SIZE)
  uint32_t nw;           // warps per core
  uint32_t nt;           // threads per warp
  uint32_t max_cluster;  // VX_CFG_DLMEM_MAX_CLUSTER
  uint32_t lmem_log;     // log2 of a core's local memory size
  uint32_t num_gangs;    // gang-id pool size
  uint32_t socket_size;
  uint32_t num_sockets;
  uint64_t dlmem_base;   // VX_MEM_DLMEM_BASE_ADDR
  uint64_t lmem_base;    // VX_MEM_LMEM_BASE_ADDR
  // 0 in nslots / max_cluster_ctas is the pre-H100 rule: one CTA slot per warp, no cluster cap.
  uint32_t nslots = 0;            // CTA slots per core (VX_CFG_MAX_CTAS_PER_SM)
  uint32_t max_cluster_ctas = 0;  // cluster CTAs per core (VX_CFG_MAX_CTAS_PER_SM_CLUSTER)
  uint32_t num_gpcs = 1;          // hardware clusters of the chip (VX_CFG_NUM_CLUSTERS)
  uint32_t policy = PLACE_SPREAD; // VX_CFG_CLUSTER_PLACEMENT
  uint32_t gpc_start = 0;         // VX_CFG_CLUSTER_GPC_START
};

inline uint32_t slots(const Cfg& cfg) {
  return cfg.nslots ? cfg.nslots : cfg.nw;
}

inline uint32_t cluster_cap(const Cfg& cfg) {
  return cfg.max_cluster_ctas ? cfg.max_cluster_ctas : slots(cfg);
}

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
    return slots(cfg);
  uint32_t u = 1;
  for (uint32_t m = 1; m <= slots(cfg); ++m) {
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

// Placed CTAs on core c: every one of them is a cluster CTA.
inline uint32_t resident(const Cfg& cfg, const Mirror& m, uint32_t c) {
  return uint32_t(__builtin_popcountll(~m.free[c] & low_mask(slots(cfg))));
}

inline void reset(const Cfg& cfg, Mirror& m) {
  m.free.assign(cfg.ncc, low_mask(slots(cfg)));
  m.used.assign(cfg.ncc, 0);
  m.rr = 0;
}

// All-or-nothing, one member per core, as on the H100: a gang that finds fewer
// than K capable cores waits rather than doubling up. On failure m is unchanged.
inline bool place(const Cfg& cfg, Mirror& m, uint32_t K, uint32_t W, uint32_t U, std::vector<Peer>& out) {
  check(K >= 1, "place: empty gang");
  Mirror s = m;
  std::vector<bool> visited(cfg.ncc, false);
  uint32_t start = m.rr;
  uint64_t umask = low_mask(U);
  out.clear();
  for (uint32_t r = 0; r < K; ++r) {
    std::vector<bool> cand(cfg.ncc, false);
    bool any_cand = false;
    for (uint32_t c = 0; c < cfg.ncc; ++c) {
      check(s.used[c] <= cfg.nw, "place: core over-reserved");
      const bool capable = ((s.free[c] & umask) != 0) && (cfg.nw - s.used[c] >= W)
                        && (resident(cfg, s, c) < cluster_cap(cfg));
      cand[c] = capable && !visited[c];
      any_cand = any_cand || cand[c];
    }
    if (!any_cand) {
      out.clear();
      return false;
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

// The H100 policy: the K capable cores of least resident CTAs, ties to the
// lowest in-GPC index (the lowest free SM ids, a TPC at a time, on an idle
// GPC; 03/spec.md §4); rank r on the r-th, each on its lowest free usable
// slot. All or nothing; on failure m is unchanged. m.rr is not used.
inline bool place_h100(const Cfg& cfg, Mirror& m, uint32_t K, uint32_t W, uint32_t U, std::vector<Peer>& out) {
  check(K >= 1, "place_h100: empty gang");
  uint64_t umask = low_mask(U);
  std::vector<bool> taken(cfg.ncc, false);
  out.clear();
  for (uint32_t r = 0; r < K; ++r) {
    int best = -1;
    for (uint32_t c = 0; c < cfg.ncc; ++c) {
      check(m.used[c] <= cfg.nw, "place_h100: core over-reserved");
      const bool capable = !taken[c] && ((m.free[c] & umask) != 0) && (cfg.nw - m.used[c] >= W)
                        && (resident(cfg, m, c) < cluster_cap(cfg));
      if (capable && (best < 0 || resident(cfg, m, c) < resident(cfg, m, uint32_t(best))))
        best = int(c);
    }
    if (best < 0) {
      out.clear();
      return false;
    }
    taken[best] = true;
    uint64_t avail = m.free[best] & umask;
    uint32_t sl = 0;
    while (((avail >> sl) & 1) == 0) {
      ++sl;
    }
    out.push_back({uint32_t(best), sl});
  }
  for (const auto& p : out) {
    m.free[p.core] &= ~(uint64_t(1) << p.slot);
    m.used[p.core] += W;
  }
  return true;
}

// The configured in-GPC policy.
inline bool place_gpc(const Cfg& cfg, Mirror& m, uint32_t K, uint32_t W, uint32_t U, std::vector<Peer>& out) {
  return (cfg.policy == PLACE_H100) ? place_h100(cfg, m, K, W, U, out) : place(cfg, m, K, W, U, out);
}

inline void release(Mirror& m, uint32_t core, uint32_t slot, uint32_t W) {
  check(((m.free.at(core) >> slot) & 1) == 0, "release: slot already free");
  check(m.used.at(core) >= W, "release: warp count underflow");
  m.free[core] |= uint64_t(1) << slot;
  m.used[core] -= W;
}

struct GangIds {
  std::vector<bool>     live;
  std::vector<uint32_t> cnt;      // done events still expected
  std::vector<uint8_t>  epoch;
  std::vector<uint32_t> arrived;  // blocks arrived at the current cluster-barrier phase
  std::vector<bool>     rel_pend; // a cluster-barrier release not yet sent
};

inline void reset(const Cfg& cfg, GangIds& g) {
  g.live.assign(cfg.num_gangs, false);
  g.cnt.assign(cfg.num_gangs, 0);
  g.epoch.assign(cfg.num_gangs, 0);
  g.arrived.assign(cfg.num_gangs, 0);
  g.rel_pend.assign(cfg.num_gangs, false);
}

// Lowest free id, or -1 when the pool is exhausted. An id whose release is
// still pending is not free: the release must leave before the id is reused.
inline int alloc(GangIds& g, uint32_t K) {
  for (size_t i = 0; i < g.live.size(); ++i) {
    if (!g.live[i] && !g.rel_pend[i]) {
      g.live[i] = true;
      g.cnt[i] = K;
      g.epoch[i] ^= 1;
      g.arrived[i] = 0;
      return int(i);
    }
  }
  return -1;
}

inline void done(GangIds& g, uint32_t id, uint32_t ep) {
  check(id < g.live.size() && g.live[id], "done: gang id not live");
  check(g.epoch[id] == ep, "done: stale epoch");
  check(g.cnt[id] > 0, "done: too many done events");
  if (--g.cnt[id] == 0) {
    check(g.arrived[id] == 0, "gang ended inside a cluster barrier phase");
    g.live[id] = false;
  }
}

// One block of a K-block gang arrived; true when it completes the phase.
inline bool cbar_arrive(GangIds& g, uint32_t id, uint32_t ep, uint32_t K) {
  check(id < g.live.size() && g.live[id], "cbar_arrive: gang id not live");
  check(g.epoch[id] == ep, "cbar_arrive: stale epoch");
  check(!g.rel_pend[id], "cbar_arrive: arrival before the previous release left");
  check(g.arrived[id] < K, "cbar_arrive: too many arrivals");
  if (++g.arrived[id] == K) {
    g.arrived[id] = 0;
    g.rel_pend[id] = true;
    return true;
  }
  return false;
}

// The lowest gang with a pending release, now sent; -1 when none.
inline int cbar_pop(GangIds& g) {
  for (size_t i = 0; i < g.rel_pend.size(); ++i) {
    if (g.rel_pend[i]) {
      g.rel_pend[i] = false;
      return int(i);
    }
  }
  return -1;
}

// A launch starts with nothing in flight, so rr and the epochs restart.
inline void new_launch(const Cfg& cfg, Mirror& m, GangIds& g) {
  for (uint32_t c = 0; c < cfg.ncc; ++c) {
    check(m.free.at(c) == low_mask(slots(cfg)) && m.used.at(c) == 0, "new_launch: mirror not empty");
  }
  for (bool l : g.live) {
    check(!l, "new_launch: gang id still live");
  }
  for (bool r : g.rel_pend) {
    check(!r, "new_launch: cluster-barrier release still pending");
  }
  m.rr = 0;
  g.epoch.assign(g.epoch.size(), 0);
}

///////////////////////////////////////////////////////////////////////////////
// Chip-wide cluster dispatch: one in-order queue over the GPCs (the H4 fix).

struct Chip {
  std::vector<Mirror>  gpc;    // per GPC: its admission mirror
  std::vector<GangIds> gangs;  // per GPC: its gang table's ids
  uint32_t ptr;                // the GPC the next scan starts from
};

inline void reset(const Cfg& cfg, Chip& chip) {
  chip.gpc.resize(cfg.num_gpcs);
  chip.gangs.resize(cfg.num_gpcs);
  for (uint32_t g = 0; g < cfg.num_gpcs; ++g) {
    reset(cfg, chip.gpc[g]);
    reset(cfg, chip.gangs[g]);
  }
  chip.ptr = cfg.gpc_start;
}

// A cluster message: K CTAs of W warps on U usable slots. K = 1 takes no gang id.
struct Msg {
  uint32_t K;
  uint32_t W;
  uint32_t U;
};

struct Placed {
  uint32_t gpc;
  std::vector<Peer> peers;  // indexed by rank; Peer::core is the in-GPC index
  int gang;                 // -1 for K = 1
};

// Every kernel starts from CLUSTER_GPC_START, so an idle GPU places a launch
// shape the same way every time (03/spec.md §4).
inline void new_launch(const Cfg& cfg, Chip& chip) {
  for (uint32_t g = 0; g < cfg.num_gpcs; ++g) {
    new_launch(cfg, chip.gpc[g], chip.gangs[g]);
  }
  chip.ptr = cfg.gpc_start;
}

// The head message goes to the first GPC from the pointer that can place it
// now; the pointer moves past that GPC. False: no GPC can, the head waits and
// nothing behind it issues; chip is unchanged.
inline bool place_chip(const Cfg& cfg, Chip& chip, const Msg& msg, Placed& out) {
  check(cfg.num_gpcs >= 1 && chip.gpc.size() == cfg.num_gpcs, "place_chip: chip not reset");
  for (uint32_t i = 0; i < cfg.num_gpcs; ++i) {
    uint32_t g = (chip.ptr + i) % cfg.num_gpcs;
    if (msg.K > 1) {
      bool id_free = false;
      for (size_t j = 0; j < chip.gangs[g].live.size(); ++j) {
        id_free = id_free || (!chip.gangs[g].live[j] && !chip.gangs[g].rel_pend[j]);
      }
      if (!id_free)
        continue;
    }
    if (!place_gpc(cfg, chip.gpc[g], msg.K, msg.W, msg.U, out.peers))
      continue;
    out.gpc  = g;
    out.gang = (msg.K > 1) ? alloc(chip.gangs[g], msg.K) : -1;
    chip.ptr = (g + 1) % cfg.num_gpcs;
    return true;
  }
  out.peers.clear();
  return false;
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
