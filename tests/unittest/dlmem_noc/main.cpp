// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// DLMEM network (socket switches + cluster crossbar) driven from host context
// the way the RTL testbench drives VX_dlmem_noc_top: an injection is offered
// before a cycle's tick and fires when the switch takes it; an ejection port
// pops after the tick the head that was visible before it.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "dlmem/dlmem_switch.h"
#include "dlmem/dlmem_xbar.h"

using namespace vortex;

namespace {

struct NocPkt {
  uint32_t dst;
  std::vector<uint64_t> data;
};

struct Shape {
  uint32_t sockets;
  uint32_t size;
};

enum class Mode { Uniform, Hotspot, Perm, NextSocket, Fair, Reset, Latency };

const char* mode_name(Mode m) {
  switch (m) {
  case Mode::Uniform:    return "uniform";
  case Mode::Hotspot:    return "hotspot";
  case Mode::Perm:       return "perm";
  case Mode::NextSocket: return "nextsock";
  case Mode::Fair:       return "fair";
  case Mode::Reset:      return "reset";
  case Mode::Latency:    return "latency";
  }
  return "?";
}

// One domain for everything, one domain per socket run serially, and one
// domain per socket on worker threads: all three must match bit for bit.
enum class Variant { Flat, Domains, Threads };

const char* variant_name(Variant v) {
  switch (v) {
  case Variant::Flat:    return "flat";
  case Variant::Domains: return "domains";
  case Variant::Threads: return "threads";
  }
  return "?";
}

struct LatRec {
  uint32_t src;
  uint32_t dst;
  uint32_t seq;
  uint64_t inject;
  uint64_t deliver;
  bool operator==(const LatRec& o) const {
    return std::tie(src, dst, seq, inject, deliver)
        == std::tie(o.src, o.dst, o.seq, o.inject, o.deliver);
  }
};

struct Result {
  uint64_t digest = 0xcbf29ce484222325ull;
  uint64_t cycles = 0;
  uint64_t packets = 0;
  uint64_t max_wait = 0;
  uint64_t collisions = 0;
  std::vector<LatRec> lat;

  void mix(uint64_t v) {
    for (int i = 0; i < 8; ++i) {
      digest ^= (v >> (8 * i)) & 0xff;
      digest *= 0x100000001b3ull;
    }
  }
};

struct Config {
  Shape shape;
  Mode mode;
  uint32_t dataw;
  uint32_t seed;
  uint32_t packets;
};

[[noreturn]] void fail(const Config& cfg, const std::string& msg) {
  std::cerr << "FAILED: shape " << cfg.shape.sockets << "x" << cfg.shape.size
            << " mode " << mode_name(cfg.mode) << " dataw " << cfg.dataw
            << ": " << msg << std::endl;
  std::exit(1);
}

uint64_t splitmix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// The RTL testbench's payload: {random bits, seq[31:8], src[7:0]}, DATAW wide.
constexpr uint32_t kSrcBits = 8;
constexpr uint32_t kSeqBits = 24;
constexpr uint32_t kMaxCores = 1u << kSrcBits;

std::vector<uint64_t> make_payload(uint32_t dataw, uint32_t seed, uint32_t src, uint32_t seq) {
  uint32_t words = (dataw + 63) / 64;
  std::vector<uint64_t> data(words);
  for (uint32_t k = 0; k < words; ++k) {
    data[k] = splitmix((uint64_t(seed) << 40) ^ (uint64_t(src) << 32) ^ (uint64_t(seq) << 4) ^ k);
  }
  data[0] = (data[0] & ~0xffffffffull) | (uint64_t(seq) << kSrcBits) | src;
  if (dataw % 64) {
    data[words - 1] &= (1ull << (dataw % 64)) - 1;
  }
  return data;
}

constexpr uint64_t kWatchdog = 2000;
constexpr uint64_t kMaxCycles = 2000000;

// The fixed latency schedule of hw/unittest/dlmem_noc/main.cpp, per source in
// offer order as (dst, release cycle); cycle 0 is the first cycle after reset.
//   0. reset:       one packet per source to core 0, then one to core NCC-1,
//                   released at 0, or at 2 when the destination is in its own
//                   socket, so that local and remote packets meet at the
//                   destination's switch: the first grants after reset are
//                   contended ones.
//   1. isolated:    T1 = 8*NCC + 64; packet k in [0, NCC*NCC) goes from core
//                   k / NCC to core k % NCC, released at T1 + 8*k.
//   2. permutation: from T2 = T1 + 8*NCC*NCC + 64, 16 packets per source s,
//                   the j-th to core (s + j + 1) % NCC.
//   3. hot spot:    from T3 = T2 + 64*NCC + 64, 8 packets per source to core 0,
//                   then 8 to core NCC-1.
std::vector<std::vector<std::pair<uint32_t, uint64_t>>> latency_schedule(uint32_t ncc, uint32_t socket_size) {
  std::vector<std::vector<std::pair<uint32_t, uint64_t>>> plan(ncc);
  uint64_t t1 = 8ull * ncc + 64;
  uint64_t t2 = t1 + 8ull * ncc * ncc + 64;
  uint64_t t3 = t2 + 64ull * ncc + 64;
  auto t0 = [socket_size](uint32_t src, uint32_t dst) -> uint64_t {
    return (src / socket_size == dst / socket_size) ? 2 : 0;
  };
  for (uint32_t s = 0; s < ncc; ++s) {
    plan[s].emplace_back(0, t0(s, 0));
    plan[s].emplace_back(ncc - 1, t0(s, ncc - 1));
  }
  for (uint32_t k = 0; k < ncc * ncc; ++k) {
    plan[k / ncc].emplace_back(k % ncc, t1 + 8ull * k);
  }
  for (uint32_t s = 0; s < ncc; ++s) {
    for (uint32_t j = 0; j < 16; ++j) {
      plan[s].emplace_back((s + j + 1) % ncc, t2);
    }
  }
  for (uint32_t s = 0; s < ncc; ++s) {
    for (uint32_t j = 0; j < 8; ++j) {
      plan[s].emplace_back(0, t3);
    }
    for (uint32_t j = 0; j < 8; ++j) {
      plan[s].emplace_back(ncc - 1, t3);
    }
  }
  return plan;
}

struct Source {
  std::mt19937 rng;
  uint32_t sent = 0;
  uint32_t total = 0;
  uint64_t eligible = 0;
  bool offered = false;
  uint64_t offer_cycle = 0;
  NocPkt pkt;
  uint32_t pkt_seq = 0;
  std::vector<std::pair<uint32_t, uint64_t>> plan;
  std::vector<uint64_t> inject_cycle;
};

struct Sink {
  std::mt19937 rng;
  int kind = 0;     // 0: always ready, 1: stalled, 2: random
  uint32_t left = 0;
  bool ready(bool random_ready) {
    if (!random_ready) {
      return true;
    }
    if (left == 0) {
      uint32_t pick = rng() % 4;
      kind = (pick == 0) ? 1 : (pick == 1) ? 2 : 0;
      left = (kind == 1) ? 20 + rng() % 130 : 20 + rng() % 280;
    }
    --left;
    return (kind == 0) || (kind == 2 && (rng() & 1));
  }
};

Result run_case(const Config& cfg, Variant variant, uint32_t workers) {
  using Switch = DlmemSocketSwitch<NocPkt>;
  using Xbar = DlmemXbar<NocPkt>;

  const uint32_t NS = cfg.shape.sockets;
  const uint32_t S = cfg.shape.size;
  const uint32_t NCC = NS * S;
  auto& platform = SimPlatform::instance();
  platform.set_num_workers(variant == Variant::Threads ? workers : 1);

  Result res;
  {
    std::vector<Switch::Ptr> switches;
    Xbar::Ptr xbar;
    for (uint32_t s = 0; s < NS; ++s) {
      uint32_t domain = (variant == Variant::Flat) ? 0 : platform.alloc_domain();
      SimPlatform::DomainScope scope(domain);
      std::string name = "dlmem-switch" + std::to_string(s);
      switches.push_back(Switch::Create(name.c_str(), s, S, NS));
    }
    if (NS > 1) {
      xbar = Xbar::Create("dlmem-xbar", S, NS);
      for (uint32_t s = 0; s < NS; ++s) {
        dlmem_connect(switches[s].get(), xbar.get(), s);
      }
    }
    platform.reset();

    auto core_in = [&](uint32_t c) -> SimChannel<NocPkt>& {
      return switches.at(c / S)->CoreIn.at(c % S);
    };
    auto core_out = [&](uint32_t c) -> SimChannel<NocPkt>& {
      return switches.at(c / S)->CoreOut.at(c % S);
    };

    std::vector<uint32_t> perm(NCC);
    for (uint32_t i = 0; i < NCC; ++i) {
      perm[i] = (cfg.mode == Mode::NextSocket) ? (i + S) % NCC : i;
    }
    std::mt19937 top_rng(cfg.seed * 7919u + NCC);
    if (cfg.mode == Mode::Perm) {
      std::shuffle(perm.begin(), perm.end(), top_rng);
    }
    uint32_t hot = top_rng() % NCC;
    // A core of socket 1 when there is one, so that the crossbar arbitrates too.
    uint32_t target = (NS > 1) ? S : 0;
    int32_t first_local = -1;
    int32_t first_remote = -1;

    std::vector<Source> srcs(NCC);
    std::vector<Sink> sinks(NCC);
    auto schedule = latency_schedule(NCC, S);
    for (uint32_t c = 0; c < NCC; ++c) {
      auto& src = srcs[c];
      src.rng.seed(cfg.seed * 1000003u + c);
      sinks[c].rng.seed(cfg.seed * 999983u + 7 * c + 1);
      if (cfg.mode == Mode::Latency) {
        src.plan = schedule[c];
        src.total = src.plan.size();
        src.eligible = src.plan.front().second;
      } else {
        src.total = (cfg.mode == Mode::Reset) ? 1 : cfg.packets;
      }
    }
    // Per (src,dst): per-source indices still to be delivered, in order.
    std::vector<std::deque<uint32_t>> expect(NCC * NCC);

    const bool random_traffic = (cfg.mode == Mode::Uniform
                              || cfg.mode == Mode::Hotspot
                              || cfg.mode == Mode::Perm
                              || cfg.mode == Mode::NextSocket);
    uint64_t injected = 0;
    uint64_t delivered = 0;
    uint64_t total = uint64_t(NCC) * srcs[0].total;
    uint64_t last_progress = 0;
    std::vector<bool> had(NCC);
    std::vector<bool> rdy(NCC);

    uint64_t cycle = 0;
    for (; delivered < total; ++cycle) {
      if (cycle > kMaxCycles) {
        fail(cfg, "cycle limit reached");
      }
      // Offer each source's next packet, one at a time, as a held valid.
      for (uint32_t c = 0; c < NCC; ++c) {
        auto& src = srcs[c];
        if (src.offered || src.sent == src.total || cycle < src.eligible) {
          continue;
        }
        uint32_t dst;
        switch (cfg.mode) {
        case Mode::Uniform:    dst = src.rng() % NCC; break;
        case Mode::Hotspot:
        case Mode::Fair:       dst = hot; break;
        case Mode::Perm:
        case Mode::NextSocket: dst = perm[c]; break;
        case Mode::Reset:      dst = target; break;
        default:               dst = src.plan.at(src.sent).first; break;
        }
        src.pkt.dst = dst;
        src.pkt_seq = src.sent;
        src.pkt.data = make_payload(cfg.dataw, cfg.seed, c, src.sent);
        if (core_in(c).size() != 0) {
          fail(cfg, "injection port still holds a packet");
        }
        core_in(c).send(src.pkt, 0);
        src.offered = true;
        src.offer_cycle = cycle;
      }
      for (uint32_t c = 0; c < NCC; ++c) {
        had[c] = !core_out(c).empty();
        rdy[c] = sinks[c].ready(random_traffic);
      }

      platform.tick();

      for (uint32_t c = 0; c < NCC; ++c) {
        auto& src = srcs[c];
        if (!src.offered || core_in(c).size() != 0) {
          continue;
        }
        uint32_t dst = src.pkt.dst;
        uint32_t pair = c * NCC + dst;
        expect[pair].push_back(src.pkt_seq);
        src.inject_cycle.push_back(cycle);
        res.max_wait = std::max(res.max_wait, cycle - src.offer_cycle);
        res.mix(cycle); res.mix(1); res.mix(c); res.mix(src.pkt_seq);
        src.offered = false;
        ++src.sent;
        ++injected;
        last_progress = cycle;
        if (cfg.mode == Mode::Latency) {
          src.eligible = (src.sent < src.total) ? std::max(cycle + 1, src.plan[src.sent].second) : 0;
        } else if (cfg.mode == Mode::Fair) {
          src.eligible = cycle + 1;
        } else {
          uint32_t pick = src.rng() % 10;
          uint32_t gap = (pick < 5) ? 0 : (pick < 9) ? 1 + src.rng() % 8 : 20 + src.rng() % 40;
          src.eligible = cycle + 1 + gap;
        }
      }
      for (uint32_t c = 0; c < NCC; ++c) {
        if (!had[c] || !rdy[c]) {
          continue;
        }
        auto& out = core_out(c);
        NocPkt pkt = out.peek();
        out.pop();
        if (pkt.dst != c) {
          fail(cfg, "packet for core " + std::to_string(pkt.dst) + " ejected at core " + std::to_string(c));
        }
        if (pkt.data.empty()) {
          fail(cfg, "empty payload");
        }
        uint32_t src = pkt.data[0] & (kMaxCores - 1);
        uint32_t seq = (pkt.data[0] >> kSrcBits) & ((1u << kSeqBits) - 1);
        if (src >= NCC || pkt.data != make_payload(cfg.dataw, cfg.seed, src, seq)) {
          fail(cfg, "corrupted payload at core " + std::to_string(c));
        }
        auto& q = expect[src * NCC + c];
        if (q.empty() || q.front() != seq) {
          std::ostringstream ss;
          ss << "out-of-order or duplicate packet " << src << "->" << c << " seq " << seq
             << " (expected " << (q.empty() ? std::string("none") : std::to_string(q.front())) << ")";
          fail(cfg, ss.str());
        }
        q.pop_front();
        if (cfg.mode == Mode::Reset && c == target) {
          int32_t& first = (src / S == target / S) ? first_local : first_remote;
          first = (first < 0) ? int32_t(src) : first;
        }
        if (cfg.mode == Mode::Latency) {
          res.lat.push_back({src, c, seq, srcs[src].inject_cycle.at(seq), cycle});
        }
        res.mix(cycle); res.mix(2); res.mix(c); res.mix(seq);
        ++delivered;
        last_progress = cycle;
      }
      if (injected > delivered || std::any_of(srcs.begin(), srcs.end(), [](const Source& s) { return s.offered; })) {
        if (cycle - last_progress > kWatchdog) {
          fail(cfg, "no progress for " + std::to_string(kWatchdog) + " cycles ("
                    + std::to_string(injected - delivered) + " packets in flight)");
        }
      }
    }
    // Drain the credit returns, then every buffer must be free again.
    for (uint32_t i = 0; i < 8 && !platform.idle(); ++i) {
      platform.tick();
    }
    if (!platform.idle()) {
      fail(cfg, "network not idle after the last delivery");
    }
    for (auto& sw : switches) {
      if (sw->has_uplink() && sw->uplink_credits() != DLMEM_BUF_DEPTH) {
        fail(cfg, "uplink credits not restored");
      }
      res.collisions += sw->collisions();
    }
    if (xbar) {
      for (uint32_t s = 0; s < NS; ++s) {
        if (xbar->credits(s) != DLMEM_BUF_DEPTH) {
          fail(cfg, "crossbar credits not restored");
        }
      }
      res.collisions += xbar->collisions();
    }
    for (auto& q : expect) {
      if (!q.empty()) {
        fail(cfg, "packet lost");
      }
    }
    // VX_rr_arbiter resets to NUM_REQS-1, so every first contention after
    // reset goes to the lowest input: the target's first local core, and at
    // the crossbar socket 0, whose uplink in turn granted core 0 first.
    if (cfg.mode == Mode::Reset) {
      if (first_local != int32_t(target / S * S)) {
        fail(cfg, "first local packet after reset from core " + std::to_string(first_local));
      }
      if (NS > 1 && first_remote != 0) {
        fail(cfg, "first remote packet after reset from core " + std::to_string(first_remote));
      }
    }
    res.cycles = cycle;
    res.packets = delivered;
  }
  platform.finalize();
  std::sort(res.lat.begin(), res.lat.end(), [](const LatRec& a, const LatRec& b) {
    return std::tie(a.src, a.seq) < std::tie(b.src, b.seq);
  });
  return res;
}

// The RTL testbench's starvation bound for an all-to-one hot spot under
// always-ready sinks: every arbiter on a packet's path is round-robin.
uint64_t fair_bound(const Shape& shape) {
  return 2ull * shape.size * shape.sockets * (shape.size + 1) + 32;
}

bool parse_shape(const std::string& s, Shape* out) {
  auto x = s.find('x');
  if (x == std::string::npos) {
    return false;
  }
  out->sockets = std::atoi(s.substr(0, x).c_str());
  out->size = std::atoi(s.substr(x + 1).c_str());
  return out->sockets != 0 && out->size != 0;
}

void usage() {
  std::cout << "usage: dlmem_noc_test [--shape=NSxS|all] [--mode=uniform|hotspot|perm|nextsock|fair|reset|latency|all]\n"
               "                      [--dataw=N[,N]] [--seed=N] [--packets=N] [--workers=N]\n"
               "                      [--csv=FILE]\n";
}

} // namespace

int main(int argc, char** argv) {
  std::vector<Shape> shapes = {{1, 1}, {1, 2}, {1, 4}, {2, 1}, {2, 2}, {4, 1}, {4, 2}};
  std::vector<Mode> modes = {Mode::Uniform, Mode::Hotspot, Mode::Perm, Mode::NextSocket,
                             Mode::Fair, Mode::Reset, Mode::Latency};
  std::vector<uint32_t> datawidths = {48, 160};
  uint32_t seed = 1;
  uint32_t packets = 0;
  uint32_t workers = (SIMX_NUM_WORKERS > 1) ? SIMX_NUM_WORKERS : 4;
  std::string csv;

  for (int a = 1; a < argc; ++a) {
    std::string arg = argv[a];
    auto val = [&](const char* key) -> const char* {
      size_t n = std::strlen(key);
      return (arg.compare(0, n, key) == 0) ? arg.c_str() + n : nullptr;
    };
    if (auto v = val("--shape=")) {
      if (std::string(v) != "all") {
        Shape s;
        if (!parse_shape(v, &s)) {
          usage();
          return 2;
        }
        shapes = {s};
      }
    } else if (auto v = val("--mode=")) {
      std::string m = v;
      if (m == "uniform") { modes = {Mode::Uniform}; }
      else if (m == "hotspot") { modes = {Mode::Hotspot}; }
      else if (m == "perm") { modes = {Mode::Perm}; }
      else if (m == "nextsock") { modes = {Mode::NextSocket}; }
      else if (m == "fair") { modes = {Mode::Fair}; }
      else if (m == "reset") { modes = {Mode::Reset}; }
      else if (m == "latency") { modes = {Mode::Latency}; }
      else if (m != "all") { usage(); return 2; }
    } else if (auto v = val("--dataw=")) {
      datawidths.clear();
      std::stringstream ss(v);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        datawidths.push_back(std::atoi(tok.c_str()));
      }
    } else if (auto v = val("--seed=")) {
      seed = std::atoi(v);
    } else if (auto v = val("--packets=")) {
      packets = std::atoi(v);
    } else if (auto v = val("--workers=")) {
      workers = std::max(2, std::atoi(v));
    } else if (auto v = val("--csv=")) {
      csv = v;
    } else {
      usage();
      return 2;
    }
  }
  for (auto w : datawidths) {
    if (w < kSrcBits + kSeqBits) {
      std::cerr << "DATAW must hold the {seq, src} header" << std::endl;
      return 2;
    }
  }
  if (!csv.empty() && (shapes.size() != 1 || datawidths.size() != 1)) {
    std::cerr << "--csv needs a single --shape and --dataw" << std::endl;
    return 2;
  }

  for (auto& shape : shapes) {
    for (auto mode : modes) {
      for (auto dataw : datawidths) {
        uint32_t n = packets;
        if (n == 0) {
          n = (mode == Mode::Hotspot || mode == Mode::Fair) ? 200 : 1000;
        }
        Config cfg{shape, mode, dataw, seed, n};
        Result ref;
        for (auto variant : {Variant::Flat, Variant::Domains, Variant::Threads}) {
          Result r = run_case(cfg, variant, workers);
          if (variant == Variant::Flat) {
            ref = r;
            continue;
          }
          if (r.digest != ref.digest || r.cycles != ref.cycles || !(r.lat == ref.lat)
           || r.collisions != ref.collisions) {
            fail(cfg, std::string("variant '") + variant_name(variant) + "' differs from 'flat'");
          }
        }
        if (mode == Mode::Fair && ref.max_wait > fair_bound(shape)) {
          fail(cfg, "a source waited " + std::to_string(ref.max_wait)
                    + " cycles under round-robin (bound " + std::to_string(fair_bound(shape)) + ")");
        }
        std::cout << "shape " << shape.sockets << "x" << shape.size
                  << " mode " << mode_name(mode) << " dataw " << dataw
                  << ": packets=" << ref.packets << " cycles=" << ref.cycles
                  << " max_wait=" << ref.max_wait << " collisions=" << ref.collisions
                  << " digest=" << std::hex << ref.digest << std::dec
                  << " (flat=domains=threads x" << workers << ")" << std::endl;
        if (mode == Mode::Latency && !csv.empty()) {
          std::ofstream ofs(csv);
          ofs << "src,dst,seq,inject_cycle,deliver_cycle\n";
          for (auto& l : ref.lat) {
            ofs << l.src << "," << l.dst << "," << l.seq << "," << l.inject << "," << l.deliver << "\n";
          }
          std::cout << "wrote " << csv << std::endl;
        }
      }
    }
  }
  std::cout << "PASSED" << std::endl;
  return 0;
}
