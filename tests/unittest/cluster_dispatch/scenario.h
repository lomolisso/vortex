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

// The cluster-dispatch scenario shared by the RTL bench (hw/unittest/cluster_dispatch,
// VX_cta_admit) and the SimX test (CtaAdmit): the same launches, the same output
// back-pressure and the same CTA lifetimes, so two models that agree cycle for
// cycle write the same CSV. Every input of a cycle depends only on earlier
// outputs. The placements are also replayed against dlmem_ref::place_chip.

#include <dlmem_ref.h>
#include <vx_hier.h>
#ifdef VX_CFG_H100_SM_ENABLE
#include <smem_carveout.h>
#else
#line 23
#endif
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace cluster_dispatch {

namespace ref = vortex::dlmem_ref;

// The build's shape, from the same VX_CFG_* macros both models are built with.
inline ref::Cfg build_cfg() {
  ref::Cfg cfg = {};
  cfg.ncc              = VX_CFG_NUM_CORES;
  cfg.nw               = VX_CFG_NUM_WARPS;
  cfg.nt               = VX_CFG_NUM_THREADS;
  cfg.max_cluster      = VX_CFG_DLMEM_MAX_CLUSTER;
  cfg.lmem_log         = VX_CFG_LMEM_LOG_SIZE;
  cfg.num_gangs        = VX_CFG_DLMEM_NUM_GANGS;
  cfg.socket_size      = VX_CFG_SOCKET_SIZE;
  cfg.num_sockets      = VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE;
  cfg.nslots           = VX_CFG_MAX_CTAS_PER_SM;
  cfg.max_cluster_ctas = VX_CFG_MAX_CTAS_PER_SM_CLUSTER;
  cfg.num_gpcs         = VX_CFG_NUM_CLUSTERS;
#ifdef VX_CFG_CLUSTER_PLACEMENT_H100
  cfg.policy           = ref::PLACE_H100;
#else
  cfg.policy           = ref::PLACE_SPREAD;
#endif
  cfg.gpc_start        = VX_CFG_CLUSTER_GPC_START;
  return cfg;
}

inline vortex::hier::Shape build_shape() {
  return vortex::hier::shape_of(VX_CFG_NUM_PARTITIONS, VX_CFG_NUM_CLUSTERS, VX_CFG_CPCS_PER_GPC,
                                VX_CFG_NUM_CORES, VX_CFG_SOCKET_SIZE);
}
#ifdef VX_CFG_H100_SM_ENABLE

// Slots a stride-0 launch opens at the KMU's reset carve-out S, as the admission computes them.
inline uint32_t reset_usable_slots(const ref::Cfg& cfg) {
  return smem_usable_slots(VX_CFG_SMEM_PER_SM_MAX, 0, ref::slots(cfg), smem_carve_cfg());
}
#else
#line 65
#endif

constexpr uint32_t KMU_POLICY_DEFAULT = 0;
constexpr uint32_t KMU_POLICY_SPREAD  = 1;

// One launch beat: rank `rank` of message `msg` of launch `launch`.
struct Beat {
  uint32_t launch;
  uint32_t msg;
  uint32_t rank;
  uint32_t k;
  uint32_t w;            // warps per CTA
  uint8_t  ctx;          // launch identity; a new value starts a kernel
  uint32_t policy;       // KMU_POLICY_*
  bool     explicit_cluster;
  bool     eop() const { return rank + 1 == k; }
};

// One GPC's CTA_DONE into the admission this cycle (after the done link).
struct Done {
  bool     valid = false;
  uint32_t core = 0;     // in-GPC index
  uint32_t slot = 0;
  uint32_t gang = 0;
  uint32_t epoch = 0;
};

struct In {
  const Beat*       head = nullptr;
  bool              out_ready = false;
  std::vector<Done> done;  // per GPC
};

struct Out {
  bool     busy = false;      // as of the start of the cycle
  bool     in_ready = false;
  bool     out_valid = false;
  bool     out_pin = false;
  uint32_t out_dest = 0;      // global SM index of a pinned beat
  uint32_t out_slot = 0;
  uint32_t out_gang = 0;
  uint32_t out_epoch = 0;
  uint32_t out_rank = 0;
  uint32_t out_ctx = 0;
};

// The model under test: evaluates one cycle with these inputs and advances the clock.
class Dut {
public:
  virtual ~Dut() = default;
  virtual void reset() = 0;
  virtual Out cycle(const In& in) = 0;
};

enum class Ready { Always, Pattern };

struct Launch {
  const char* name;
  uint32_t k;
  uint32_t w;
  uint32_t msgs;
  uint32_t policy;
  bool     explicit_cluster;
  Ready    ready;
  uint32_t life_base;   // CTA lifetime: base + hash % span cycles
  uint32_t life_span;
  bool     cluster() const { return policy == KMU_POLICY_SPREAD || explicit_cluster; }
};

inline uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}

// Clusters of K an idle chip holds at once (W warps each, every slot usable).
inline uint32_t first_wave(const ref::Cfg& cfg, uint32_t k, uint32_t w) {
  ref::Chip chip;
  ref::reset(cfg, chip);
  ref::Placed p;
  uint32_t n = 0;
#ifdef VX_CFG_H100_SM_ENABLE
  while (ref::place_chip(cfg, chip, {k, w, reset_usable_slots(cfg)}, p))
#else
#line 145
  while (ref::place_chip(cfg, chip, {k, w, ref::slots(cfg)}, p))
#line 145
#endif
    ++n;
  return n;
}

class Scenario {
public:
  Scenario(Dut& dut, uint32_t seed, FILE* csv) : dut_(dut), seed_(seed), csv_(csv), cfg_(build_cfg()) {}

  uint32_t errors = 0;

  // The launch list of the contract's T0 unit TB: a cluster stream of K = 2 run twice
  // back to back, a stream of the largest K with two-warp CTAs, a K = 1 cluster stream
  // and a plain stream, the last three with an always-ready output.
  std::vector<Launch> launches() const {
    std::vector<Launch> v;
    const uint32_t k2 = std::min<uint32_t>(2, cfg_.max_cluster);
    const uint32_t kmax = cfg_.max_cluster;
    const uint32_t w2 = std::min<uint32_t>(2, cfg_.nw);
    auto msgs = [&](uint32_t k, uint32_t w) {
      return std::min<uint32_t>(2 * first_wave(cfg_, k, w) + 3, 600);
    };
    v.push_back({"k2", k2, 1, msgs(k2, 1), KMU_POLICY_SPREAD, true, Ready::Pattern, 20, 400});
    v.push_back({"k2-again", k2, 1, msgs(k2, 1), KMU_POLICY_SPREAD, true, Ready::Pattern, 20, 400});
    v.push_back({"kmax", kmax, w2, msgs(kmax, w2), KMU_POLICY_SPREAD, true, Ready::Always, 10, 300});
#ifdef VX_CFG_CLUSTER_DISTINCT_SMS_ENABLE
    v.push_back({"k1", 1, 1, msgs(1, 1), KMU_POLICY_DEFAULT, true, Ready::Always, 10, 200});
#else
    v.push_back({"k1", 1, 1, msgs(1, 1), KMU_POLICY_SPREAD, true, Ready::Always, 10, 200});
#endif
    v.push_back({"plain", 1, 1, 64, KMU_POLICY_DEFAULT, false, Ready::Always, 0, 0});
    return v;
  }

  bool run() {
    const auto ls = this->launches();
    dut_.reset();
    std::fprintf(csv_, "# G=%u NCC=%u C=%u S2=%u NW=%u slots=%u cap=%u gangs=%u policy=%s start=%u\n",
                 cfg_.num_gpcs, cfg_.ncc, (uint32_t)VX_CFG_CPCS_PER_GPC, cfg_.socket_size, cfg_.nw,
                 ref::slots(cfg_), ref::cluster_cap(cfg_), cfg_.num_gangs,
                 cfg_.policy == ref::PLACE_H100 ? "H100" : "SPREAD", cfg_.gpc_start);
    for (uint32_t l = 0; l < ls.size() && errors == 0; ++l)
      this->run_launch(l, ls[l]);
    if (errors == 0)
      this->check_repeat(ls);
    return errors == 0;
  }

private:
  struct Live { uint32_t k, w, gang, epoch; };
  using Key = std::tuple<uint32_t, uint32_t, uint32_t>;  // GPC, in-GPC core, slot
  struct Pending { Key key; Live lv; };
  struct Due { uint64_t cycle; uint32_t core, slot; };
  struct Row { uint64_t rel; uint32_t msg, rank, gpc, sm, slot, gang, epoch; };

  void fail(const char* fmt, ...) {
    if (errors < 10) {
      va_list ap;
      va_start(ap, fmt);
      std::fprintf(stderr, "cluster_dispatch: cycle %llu: ", (unsigned long long)cycle_);
      std::vfprintf(stderr, fmt, ap);
      std::fprintf(stderr, "\n");
      va_end(ap);
    }
    ++errors;
  }

  uint32_t lifetime(const Launch& L, uint32_t msg, uint32_t rank) const {
    return L.life_base + (L.life_span ? mix(seed_ * 0x9e3779b9U ^ (msg << 8) ^ rank) % L.life_span : 0);
  }

  // Admission FSM: the start cycle, K place cycles, K emit cycles.
  static uint64_t place_pace(uint32_t k) { return 2 * uint64_t(k) + 1; }

  // the back-pressure pattern counts from the launch start, so identical launches see the same one
  bool ready_at(const Launch& L, uint64_t rel) const {
    return L.ready == Ready::Always || ((rel * 13 + 5) % 16) != 0;
  }

  void run_launch(uint32_t l, const Launch& L) {
    std::deque<Beat> beats;
    for (uint32_t m = 0; m < L.msgs; ++m)
      for (uint32_t r = 0; r < L.k; ++r)
        beats.push_back({l, m, r, L.k, L.w, uint8_t(l + 1), L.policy, L.explicit_cluster});
    const uint64_t start = cycle_;
    std::vector<std::deque<Due>> due(cfg_.num_gpcs);
    std::map<Key, Live> live;
    std::vector<Pending> pending;  // delivered dones the golden has not applied yet
    std::vector<ref::Peer> msg_peers;
    uint32_t msg_gpc = 0, next_msg = 0, next_rank = 0, last_first_out = 0;
    bool have_first_out = false;
    uint64_t last_first_cycle = 0, min_gap = ~0ull;
    ref::reset(cfg_, chip_);
    ref::new_launch(cfg_, chip_);
    rows_.emplace_back();
    uint32_t delivered = 0, outstanding = 0;
    uint64_t idle = 0;

    while (true) {
      In in;
      in.head = beats.empty() ? nullptr : &beats.front();
      in.out_ready = this->ready_at(L, cycle_ - start);
      in.done.assign(cfg_.num_gpcs, Done());
      for (uint32_t g = 0; g < cfg_.num_gpcs; ++g) {
        if (!due[g].empty() && due[g].front().cycle <= cycle_) {
          const Due d = due[g].front();
          due[g].pop_front();
          const Key key{g, d.core, d.slot};
          const Live lv = live.at(key);
          live.erase(key);
          in.done[g] = {true, d.core, d.slot, lv.gang, lv.epoch};
          pending.push_back({key, lv});
          std::fprintf(csv_, "D,%llu,%u,%u,%u\n", (unsigned long long)(cycle_ - start), g, d.core, d.slot);
          --outstanding;
        }
      }
      const Out o = dut_.cycle(in);
      const bool progress = (in.head && o.in_ready) || (o.out_valid && in.out_ready);
      if (in.head && o.in_ready)
        beats.pop_front();
      if (o.out_valid && in.out_ready) {
        if (o.out_ctx != uint32_t(l + 1))
          this->fail("%s: a beat of launch context %u, expected %u", L.name, o.out_ctx, l + 1);
        const uint32_t msg = next_msg, rank = next_rank;
        if (o.out_rank != rank)
          this->fail("%s: message %u: rank %u out, expected %u (in order, all or nothing)", L.name, msg, o.out_rank, rank);
        if (!L.cluster()) {
          if (o.out_pin)
            this->fail("%s: a plain CTA was pinned", L.name);
          std::fprintf(csv_, "U,%llu,%u,%u\n", (unsigned long long)(cycle_ - start), l, msg);
        } else {
          if (!o.out_pin)
            this->fail("%s: message %u rank %u left unpinned", L.name, msg, rank);
          const uint32_t gpc = o.out_dest / cfg_.ncc, core = o.out_dest % cfg_.ncc;
          if (rank == 0) {
            msg_gpc = gpc;
            msg_peers.clear();
            if (L.ready == Ready::Always && have_first_out) {
              const uint64_t gap = cycle_ - last_first_cycle;
              if (gap < place_pace(L.k))
                this->fail("%s: message %u issued %llu cycles after message %u, below the FSM's %llu", L.name, msg,
                           (unsigned long long)gap, last_first_out, (unsigned long long)place_pace(L.k));
              min_gap = std::min(min_gap, gap);
            }
            have_first_out = true;
            last_first_cycle = cycle_;
            last_first_out = msg;
          } else if (gpc != msg_gpc) {
            this->fail("%s: message %u rank %u in GPC %u, rank 0 in GPC %u", L.name, msg, rank, gpc, msg_gpc);
          }
          msg_peers.push_back({core, o.out_slot});
          if (!live.emplace(std::make_tuple(gpc, core, o.out_slot),
                            Live{L.k, L.w, o.out_gang, o.out_epoch}).second)
            this->fail("%s: SM %u slot %u placed twice", L.name, o.out_dest, o.out_slot);
          std::fprintf(csv_, "P,%llu,%u,%u,%u,%u,%u,%u,%u,%u\n", (unsigned long long)(cycle_ - start), l, msg,
                       rank, gpc, o.out_dest, o.out_slot, L.k > 1 ? o.out_gang : 0, L.k > 1 ? o.out_epoch : 0);
          rows_.back().push_back({cycle_ - start, msg, rank, gpc, o.out_dest, o.out_slot,
                                  L.k > 1 ? o.out_gang : 0, L.k > 1 ? o.out_epoch : 0});
          due[gpc].push_back({cycle_ + this->lifetime(L, msg, rank), core, o.out_slot});
          std::stable_sort(due[gpc].begin(), due[gpc].end(),
                           [](const Due& a, const Due& b) { return a.cycle < b.cycle; });
          ++outstanding;
          if (rank + 1 == L.k)
            this->golden(L, msg, msg_gpc, msg_peers, {o.out_gang, o.out_epoch}, pending);
        }
        ++delivered;
        if (++next_rank == L.k) {
          next_rank = 0;
          ++next_msg;
        }
      }
      ++cycle_;
      idle = progress ? 0 : idle + 1;
      if (idle > 100000) {
        this->fail("%s: no progress for 100000 cycles (%u of %u beats out)", L.name, delivered, L.msgs * L.k);
        return;
      }
      if (errors)
        return;
      if (beats.empty() && delivered == L.msgs * L.k && outstanding == 0 && !o.busy)
        break;
    }
    if (L.cluster() && L.ready == Ready::Always && min_gap != ~0ull) {
      // nothing but the FSM paces back-to-back placements
      if (min_gap != place_pace(L.k))
        this->fail("%s: closest placements %llu cycles apart, the FSM's pace is %llu", L.name,
                   (unsigned long long)min_gap, (unsigned long long)place_pace(L.k));
      if (L.k == 1)
        std::printf("H100REF metric=cluster_place_interval_cycles h100=%.1f vortex=%llu\n",
                    VX_CFG_H100_CLUSTER_DISPATCH_PS * 1e-6 * VX_CFG_H100_SM_CLOCK_MHZ, (unsigned long long)min_gap);
    }
    // Drained: every done reached the mirror, so the next launch meets an idle chip.
    for (const auto& d : pending)
      apply_done(chip_, d);
  }

  static void apply_done(ref::Chip& chip, const Pending& d) {
    const uint32_t g = std::get<0>(d.key);
    ref::release(chip.gpc[g], std::get<1>(d.key), std::get<2>(d.key), d.lv.w);
    if (d.lv.k > 1)
      ref::done(chip.gangs[g], d.lv.gang, d.lv.epoch);
  }

  // The message must be what place_chip gives for the mirror with some prefix of
  // the dones delivered since the previous placement (the admission sees a done a
  // fixed number of cycles after its delivery). The slots come from the mirror as
  // the placement started and the gang id from the ids free as it commits, so a
  // gang's id may reflect a longer prefix than its slots.
  void golden(const Launch& L, uint32_t msg, uint32_t gpc, const std::vector<ref::Peer>& got,
              const std::vector<uint32_t>& gang_epoch, std::vector<Pending>& pending) {
    for (size_t j = 0; j <= pending.size(); ++j) {
      ref::Chip trial = chip_;
      for (size_t i = 0; i < j; ++i)
        apply_done(trial, pending[i]);
      ref::Placed p;
#ifdef VX_CFG_H100_SM_ENABLE
      if (!ref::place_chip(cfg_, trial, {L.k, L.w, reset_usable_slots(cfg_)}, p))
#else
#line 349
      if (!ref::place_chip(cfg_, trial, {L.k, L.w, ref::slots(cfg_)}, p))
#line 349
#endif
        continue;
      bool same = (p.gpc == gpc) && (p.peers.size() == got.size());
      for (size_t r = 0; same && r < got.size(); ++r)
        same = (p.peers[r].core == got[r].core) && (p.peers[r].slot == got[r].slot);
      if (!same)
        continue;
      if (L.k == 1) {
        pending.erase(pending.begin(), pending.begin() + j);
        chip_ = trial;
        return;
      }
      for (size_t j2 = j; j2 <= pending.size(); ++j2) {
        ref::Chip commit = chip_;
        for (size_t i = 0; i < j2; ++i)
          apply_done(commit, pending[i]);
        for (const auto& pr : got) {
          commit.gpc[gpc].free[pr.core] &= ~(uint64_t(1) << pr.slot);
          commit.gpc[gpc].used[pr.core] += L.w;
        }
        commit.gpc[gpc].rr = trial.gpc[gpc].rr;
        commit.ptr = trial.ptr;
        const int id = ref::alloc(commit.gangs[gpc], L.k);
        if (id == int(gang_epoch[0]) && gang_epoch[1] == commit.gangs[gpc].epoch[id]) {
          pending.erase(pending.begin(), pending.begin() + j2);
          chip_ = commit;
          return;
        }
      }
    }
    std::string s;
    for (const auto& pr : got)
      s += " " + std::to_string(pr.core) + ":" + std::to_string(pr.slot);
    this->fail("%s: message %u in GPC %u on%s, gang %u epoch %u: no mirror state since the previous placement"
               " gives that (dlmem_ref::place_chip)", L.name, msg, gpc, s.c_str(), gang_epoch[0], gang_epoch[1]);
    const ref::Mirror& m = chip_.gpc.at(gpc);
    for (uint32_t c = 0; c < cfg_.ncc; ++c)
      std::fprintf(stderr, "  golden GPC %u core %u: free slots 0x%llx, warps %u\n", gpc, c,
                   (unsigned long long)m.free[c], m.used[c]);
    for (const auto& d : pending)
      std::fprintf(stderr, "  done not applied yet: GPC %u core %u slot %u (gang %u)\n", std::get<0>(d.key),
                   std::get<1>(d.key), std::get<2>(d.key), d.lv.gang);
  }

  // Two identical launches back to back place identically, cycle for cycle.
  void check_repeat(const std::vector<Launch>& ls) {
    if (ls.size() < 2 || rows_.size() < 2)
      return;
    const auto& a = rows_[0];
    const auto& b = rows_[1];
    if (a.size() != b.size()) {
      this->fail("%s placed %zu CTAs, %s %zu", ls[0].name, a.size(), ls[1].name, b.size());
      return;
    }
    for (size_t i = 0; i < a.size(); ++i) {
      if (std::tie(a[i].rel, a[i].msg, a[i].rank, a[i].gpc, a[i].sm, a[i].slot, a[i].gang, a[i].epoch)
          != std::tie(b[i].rel, b[i].msg, b[i].rank, b[i].gpc, b[i].sm, b[i].slot, b[i].gang, b[i].epoch)) {
        this->fail("%s and %s differ at CTA %zu: message %u rank %u on SM %u slot %u at +%llu vs message %u rank"
                   " %u on SM %u slot %u at +%llu", ls[0].name, ls[1].name, i, a[i].msg, a[i].rank, a[i].sm,
                   a[i].slot, (unsigned long long)a[i].rel, b[i].msg, b[i].rank, b[i].sm, b[i].slot,
                   (unsigned long long)b[i].rel);
        return;
      }
    }
  }

  Dut&      dut_;
  uint32_t  seed_;
  FILE*     csv_;
  ref::Cfg  cfg_;
  ref::Chip chip_;
  uint64_t  cycle_ = 0;
  std::vector<std::vector<Row>> rows_;
};

// The RTL pkg's position functions against vx_hier.h, SM by SM: `pos(sm)` returns
// {partition, gpc, cpc, tpc_slot, parity, slot_dist} as the RTL computes them.
template <typename F>
uint32_t check_positions(F pos) {
  const vortex::hier::Shape s = build_shape();
  uint32_t errors = 0;
  for (uint32_t sm = 0; sm < s.num_sms(); ++sm) {
    const vortex::hier::Pos p = vortex::hier::decode(s, sm);
    const std::vector<uint32_t> want = {p.partition, p.gpc, p.cpc, p.tpc_slot, p.parity,
                                        vortex::hier::slot_dist(s, sm)};
    const std::vector<uint32_t> got = pos(sm);
    if (got != want) {
      if (errors < 10)
        std::fprintf(stderr, "cluster_dispatch: SM %u: RTL position (%u, %u, %u, %u, %u, dist %u), vx_hier"
                     " (%u, %u, %u, %u, %u, dist %u)\n", sm, got[0], got[1], got[2], got[3], got[4], got[5],
                     want[0], want[1], want[2], want[3], want[4], want[5]);
      ++errors;
    }
  }
  return errors;
}

}  // namespace cluster_dispatch
