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

// DLMEM aperture translation (DlmemXlate) driven from host context: a request
// or peer row sent before a cycle's tick lands after it, and an output port
// pops after the tick the head that was visible before it. A cycle-exact model
// of the one-stage pipe and of the peer table checks every output against
// dlmem_ref::translate.

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#include "dlmem/dlmem_xlate.h"
#include "dlmem/dlmem_wtrack.h"
#include "vectors.h"
#ifdef VX_CFG_EXT_MBAR_ENABLE
#include "mbar_ref.h"
#endif

using namespace vortex;
namespace V = dlmem_xlate_vec;

namespace {

static_assert(uint32_t(dlmem_ref::NONE) == V::K_NONE && uint32_t(dlmem_ref::LOCAL) == V::K_LOCAL
           && uint32_t(dlmem_ref::REMOTE) == V::K_REMOTE && uint32_t(dlmem_ref::OOB) == V::K_OOB,
              "vector kinds must follow dlmem_ref::Kind");

[[noreturn]] void fail(const std::string& ctx, const std::string& msg) {
  std::cerr << "FAILED: " << ctx << ": " << msg << std::endl;
  std::exit(1);
}

std::string hex(uint64_t v) {
  std::ostringstream ss;
  ss << "0x" << std::hex << v;
  return ss.str();
}

dlmem_ref::Cfg make_cfg(uint32_t ncc, uint32_t nw, uint32_t max_cluster, uint32_t lmem_log) {
  dlmem_ref::Cfg cfg{};
  cfg.ncc = ncc;
  cfg.nw = nw;
  cfg.nt = 4;
  cfg.max_cluster = max_cluster;
  cfg.lmem_log = lmem_log;
  cfg.num_gangs = std::max(2u, ncc * nw / 2);
  cfg.socket_size = ncc;
  cfg.num_sockets = 1;
  cfg.dlmem_base = VX_MEM_DLMEM_BASE_ADDR;
  cfg.lmem_base = VX_MEM_LMEM_BASE_ADDR;
  return cfg;
}

dlmem_ref::Cfg shape_cfg(const V::Shape& s) {
  return make_cfg(s.ncc, s.nw, s.max_cluster, s.lmem_log);
}

const char* kind_name(uint32_t k) {
  switch (k) {
  case dlmem_ref::NONE:   return "NONE";
  case dlmem_ref::LOCAL:  return "LOCAL";
  case dlmem_ref::REMOTE: return "REMOTE";
  case dlmem_ref::OOB:    return "OOB";
  }
  return "?";
}

// The properties the RTL and the endpoints rely on (see dlmem_ref.h).
void check_props(const std::string& ctx, const dlmem_ref::Cfg& cfg, uint64_t addr,
                 uint32_t my_core, const dlmem_ref::Xlate& x) {
  uint64_t lmem_end = cfg.lmem_base + (uint64_t(1) << cfg.lmem_log);
  if ((x.out_addr & 63) != (addr & 63)) {
    fail(ctx, "low six bits changed: " + hex(addr) + " -> " + hex(x.out_addr));
  }
  switch (x.kind) {
  case dlmem_ref::NONE:
    if (x.out_addr != addr) {
      fail(ctx, "NONE changed the address");
    }
    break;
  case dlmem_ref::LOCAL:
  case dlmem_ref::OOB:
    if (x.core != my_core) {
      fail(ctx, std::string(kind_name(x.kind)) + " lane targets core " + std::to_string(x.core));
    }
    if (x.out_addr < cfg.lmem_base || x.out_addr >= lmem_end || x.out_addr - cfg.lmem_base != x.lmem_byte) {
      fail(ctx, std::string(kind_name(x.kind)) + " lane leaves the core's SRAM: " + hex(x.out_addr));
    }
    break;
  case dlmem_ref::REMOTE: {
    if (x.core == my_core) {
      fail(ctx, "REMOTE lane targets its own core");
    }
    if (x.out_addr < cfg.dlmem_base || (x.out_addr >= cfg.lmem_base && x.out_addr < lmem_end)) {
      fail(ctx, "REMOTE lane outside the DLMEM aperture: " + hex(x.out_addr));
    }
    uint32_t core;
    uint64_t lb;
    dlmem_ref::decode_remote(cfg, x.out_addr, &core, &lb);
    if (core != x.core || lb != x.lmem_byte || lb >= (uint64_t(1) << cfg.lmem_log)) {
      fail(ctx, "remote encoding " + hex(x.out_addr) + " does not decode to core "
                + std::to_string(x.core) + " byte " + hex(x.lmem_byte));
    }
    break;
  }
  }
}

std::shared_ptr<mem_block_t> copy_block(const std::shared_ptr<mem_block_t>& b) {
  // Plain heap blocks: the worker threads release them under any SIMX_MT.
  return b ? std::make_shared<mem_block_t>(*b) : nullptr;
}

// What the stage must emit for `in`, under its CTA's row and the stride.
LsuReq expected_out(const std::string& ctx, const dlmem_ref::Cfg& cfg, uint32_t my_core,
                    const dlmem_ref::Row& row, uint32_t stride,
                    const LsuReq& in, uint32_t* remote, uint64_t* kinds) {
  LsuReq e = in;
  *remote = 0;
  for (uint32_t i = 0; i < in.mask.size(); ++i) {
    e.data.at(i) = copy_block(in.data.at(i));
    e.dlmem_remote.reset(i);
    e.dlmem_oob.reset(i);
    e.dlmem_core.at(i) = 0;
    if (!in.mask.test(i) || in.flags.fence) {
      continue;
    }
    auto x = dlmem_ref::translate(cfg, in.addrs.at(i), row, my_core, in.cta_id, stride);
    check_props(ctx, cfg, in.addrs.at(i), my_core, x);
    ++kinds[x.kind];
    if (x.kind == dlmem_ref::NONE) {
      continue;
    }
    e.addrs.at(i) = x.out_addr;
    e.dlmem_core.at(i) = x.core;
    if (x.kind == dlmem_ref::REMOTE) {
      e.dlmem_remote.set(i);
      ++*remote;
    }
    if (x.kind == dlmem_ref::OOB) {
      e.dlmem_oob.set(i);
    }
  }
  return e;
}

void compare_req(const std::string& ctx, const LsuReq& got, const LsuReq& exp) {
  auto bad = [&](const std::string& what) {
    fail(ctx, "request #" + std::to_string(exp.uuid) + ": " + what);
  };
  if (got.uuid != exp.uuid) {
    bad("out of order: got #" + std::to_string(got.uuid));
  }
  if (got.op != exp.op || got.flags != exp.flags || got.tag != exp.tag || got.cid != exp.cid
   || got.wid != exp.wid || got.cta_id != exp.cta_id) {
    bad("header fields changed");
  }
  if (!(got.mask == exp.mask)) {
    bad("mask changed");
  }
  uint32_t n = exp.mask.size();
  if (got.addrs.size() != n || got.data.size() != n || got.byteen.size() != n || got.amo_cmp.size() != n
   || got.tids.size() != n || got.dlmem_remote.size() != n || got.dlmem_oob.size() != n
   || got.dlmem_core.size() != n) {
    bad("lane count changed");
  }
  for (uint32_t i = 0; i < n; ++i) {
    std::string lane = " on lane " + std::to_string(i);
    if (got.addrs[i] != exp.addrs[i]) {
      bad("address " + hex(got.addrs[i]) + ", expected " + hex(exp.addrs[i]) + lane);
    }
    if (got.dlmem_remote.test(i) != exp.dlmem_remote.test(i) || got.dlmem_oob.test(i) != exp.dlmem_oob.test(i)
     || got.dlmem_core[i] != exp.dlmem_core[i]) {
      bad("remote/oob/core sideband " + std::to_string(got.dlmem_remote.test(i)) + "/"
          + std::to_string(got.dlmem_oob.test(i)) + "/" + std::to_string(got.dlmem_core[i]) + ", expected "
          + std::to_string(exp.dlmem_remote.test(i)) + "/" + std::to_string(exp.dlmem_oob.test(i)) + "/"
          + std::to_string(exp.dlmem_core[i]) + lane);
    }
    if (got.byteen[i] != exp.byteen[i] || got.amo_cmp[i] != exp.amo_cmp[i] || got.tids[i] != exp.tids[i]) {
      bad("byteen/amo_cmp/tid changed" + lane);
    }
    if (bool(got.data[i]) != bool(exp.data[i]) || (got.data[i] && *got.data[i] != *exp.data[i])) {
      bad("store data changed" + lane);
    }
  }
}

// Lanes of a translated request that access this core's local memory, whatever
// the op: the switch's local leg. A fence token's lanes carry no access.
uint32_t local_access_lanes(const LsuReq& out) {
  if (out.flags.fence) {
    return 0;
  }
  uint32_t n = 0;
  for (uint32_t i = 0; i < out.mask.size(); ++i) {
    n += out.mask.test(i) && !out.dlmem_remote.test(i) && get_addr_type(out.addrs.at(i)) == AddrType::Shared;
  }
  return n;
}

///////////////////////////////////////////////////////////////////////////////
// Cycle-exact model of the stage and its peer table

struct InEntry {
  LsuReq req;
  uint64_t avail;  // first tick the stage can take it
  bool entered = false;
  dlmem_ref::Row row;  // its CTA's row as it entered the stage
};

struct OutEntry {
  LsuReq req;
  uint64_t tick;   // tick the stage sent it
  uint64_t sent;   // host cycle the request was sent
};

struct BlockModel {
  std::deque<InEntry> in;
  std::deque<OutEntry> out;
  std::deque<WtInc> wt;  // expected this tick
  std::deque<WtInc> lacc; // expected this tick
};

struct TableModel {
  std::vector<dlmem_ref::Row> rows;
  uint32_t stride = 0;
  std::deque<std::pair<uint64_t, dlmem_ref::RowWrite>> pending;  // (tick it lands, write)

  void reset(uint32_t nw) {
    rows.assign(nw, dlmem_ref::Row{0, {}});
    stride = 0;
    pending.clear();
  }
  void apply(uint64_t tick) {
    while (!pending.empty() && pending.front().first <= tick) {
      auto& w = pending.front().second;
      rows.at(w.slot) = w.row;
      stride = w.stride;
      pending.pop_front();
    }
  }
};

// Host-side driver of one DlmemXlate and its model.
struct Harness {
  std::string ctx;
  dlmem_ref::Cfg cfg;
  uint32_t my_core;
  uint32_t num_blocks;
  DlmemXlate::Ptr dut;
  TableModel tm;
  std::vector<BlockModel> blocks;
  uint64_t outputs = 0;
  uint64_t fences = 0;
  uint64_t wt_events = 0;
  uint64_t lacc_events = 0;
  uint64_t lacc_lanes = 0;
  uint64_t kinds[4] = {0, 0, 0, 0};  // active lanes per dlmem_ref::Kind

  Harness(const std::string& name, const dlmem_ref::Cfg& c, uint32_t core_id, uint32_t nb,
          bool aborts = false)
    : ctx(name), cfg(c), my_core(core_id % c.ncc), num_blocks(nb), blocks(nb) {
    dut = DlmemXlate::Create(name.c_str(), cfg, core_id, nb, aborts, aborts);
    tm.reset(cfg.nw);
    if (dut->ReqIn.size() != nb || dut->ReqOut.size() != nb || dut->WtIncOut.size() != nb
     || dut->LaccIncOut.size() != nb) {
      fail(ctx, "one request/output/count channel per block expected");
    }
    for (uint32_t b = 0; b < nb; ++b) {
      if (dut->ReqIn.at(b).capacity() != 2) {
        fail(ctx, "ReqIn capacity " + std::to_string(dut->ReqIn.at(b).capacity()) + ", expected 2");
      }
    }
  }

  // Before tick `cycle`.
  void write_row(uint64_t cycle, const dlmem_ref::RowWrite& w) {
    if (dut->PeerWrIn.full()) {
      fail(ctx, "peer-row port full");
    }
    dut->PeerWrIn.send(w, 1);
    tm.pending.push_back({cycle + 1, w});
  }

  bool in_full(uint32_t b) {
    auto& ch = dut->ReqIn.at(b);
    uint32_t occ = blocks[b].in.size();
    if (ch.size() != occ || ch.full() != (occ >= 2)) {
      fail(ctx, "block " + std::to_string(b) + " input holds " + std::to_string(ch.size())
                + " (full=" + std::to_string(ch.full()) + "), model " + std::to_string(occ));
    }
    return ch.full();
  }

  // Before tick `cycle`; the caller checked in_full().
  void send_req(uint64_t cycle, uint32_t b, const LsuReq& req) {
    dut->ReqIn.at(b).send(req, 1);
    blocks[b].in.push_back({req, cycle + 1, false, dlmem_ref::Row{}});
    if (dut->empty()) {
      fail(ctx, "empty() with request #" + std::to_string(req.uuid) + " in flight to the stage");
    }
  }

  // The core's drain and busy terms: empty() is exactly "no request held or arriving".
  void check_empty(uint64_t cycle) {
    bool model_empty = true;
    for (auto& bm : blocks) {
      model_empty = model_empty && bm.in.empty();
    }
    if (dut->empty() != model_empty) {
      fail(ctx, "empty()=" + std::to_string(dut->empty()) + ", model " + std::to_string(model_empty)
                + " (cycle " + std::to_string(cycle) + ")");
    }
  }

  // The model's half of tick `cycle`.
  void model_tick(uint64_t cycle) {
    if (cycle != SimPlatform::instance().cycles()) {
      fail(ctx, "harness out of step with the platform");
    }
    tm.apply(cycle);
    for (uint32_t b = 0; b < num_blocks; ++b) {
      auto& bm = blocks[b];
      bm.wt.clear();
      bm.lacc.clear();
      // A request sent this cycle enters after this tick, under the rows written so far.
      for (auto& in : bm.in) {
        if (in.entered) {
          continue;
        }
        if (in.avail != cycle + 1) {
          fail(ctx, "the model skipped the entry of request #" + std::to_string(in.req.uuid));
        }
        in.row = tm.rows.at(in.req.cta_id);
        in.entered = true;
      }
      if (bm.in.empty() || bm.in.front().avail > cycle || bm.out.size() >= 2) {
        continue;
      }
      auto& ie = bm.in.front();
      uint32_t remote = 0;
      LsuReq e = expected_out(ctx, cfg, my_core, ie.row, tm.stride, ie.req, &remote, kinds);
      if (remote != 0) {
        bm.wt.push_back(WtInc{ie.req.wid, remote});
      }
      uint32_t local = local_access_lanes(e);
      if (local != 0) {
        bm.lacc.push_back(WtInc{ie.req.wid, local});
      }
      // Marked for its banks exactly when its local lanes were counted.
      e.flags.lacc = (local != 0);
      bm.out.push_back({std::move(e), cycle, ie.avail - 1});
      bm.in.pop_front();
    }
  }

  // After tick `cycle`: outputs match the model; pop one where ready.
  void check_outputs(uint64_t cycle, const std::vector<bool>& ready, std::vector<OutEntry>* popped) {
    this->check_empty(cycle);
    for (uint32_t b = 0; b < num_blocks; ++b) {
      auto& bm = blocks[b];
      auto& out = dut->ReqOut.at(b);
      std::string bctx = " (block " + std::to_string(b) + ", cycle " + std::to_string(cycle) + ")";
      if (out.size() != bm.out.size()) {
        fail(ctx, "output holds " + std::to_string(out.size()) + " requests, model "
                  + std::to_string(bm.out.size()) + bctx);
      }
      auto& wt = dut->WtIncOut.at(b);
      if (wt.size() != bm.wt.size()) {
        fail(ctx, "remote-access count events " + std::to_string(wt.size()) + ", expected "
                  + std::to_string(bm.wt.size()) + bctx);
      }
      for (auto& w : bm.wt) {
        auto& g = wt.peek();
        if (g.wid != w.wid || g.cnt != w.cnt) {
          fail(ctx, "remote-access count {wid=" + std::to_string(g.wid) + ", cnt=" + std::to_string(g.cnt)
                    + "}, expected {" + std::to_string(w.wid) + ", " + std::to_string(w.cnt) + "}" + bctx);
        }
        wt.pop();
        ++wt_events;
      }
      bm.wt.clear();
      auto& lacc = dut->LaccIncOut.at(b);
      if (lacc.size() != bm.lacc.size()) {
        fail(ctx, "local-access count events " + std::to_string(lacc.size()) + ", expected "
                  + std::to_string(bm.lacc.size()) + bctx);
      }
      for (auto& w : bm.lacc) {
        auto& g = lacc.peek();
        if (g.wid != w.wid || g.cnt != w.cnt) {
          fail(ctx, "local-access count {wid=" + std::to_string(g.wid) + ", cnt=" + std::to_string(g.cnt)
                    + "}, expected {" + std::to_string(w.wid) + ", " + std::to_string(w.cnt) + "}" + bctx);
        }
        lacc.pop();
        ++lacc_events;
        lacc_lanes += w.cnt;
      }
      bm.lacc.clear();
      if (!ready.at(b) || out.empty()) {
        continue;
      }
      compare_req(ctx + bctx, out.peek(), bm.out.front().req);
      fences += bm.out.front().req.flags.fence;
      out.pop();
      if (popped) {
        popped->push_back(bm.out.front());
      }
      bm.out.pop_front();
      ++outputs;
    }
  }

  bool drained() const {
    for (auto& bm : blocks) {
      if (!bm.in.empty() || !bm.out.empty()) {
        return false;
      }
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Traffic

uint64_t splitmix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

struct Rng {
  std::mt19937_64 gen;
  explicit Rng(uint64_t seed) : gen(splitmix(seed)) {}
  uint64_t operator()() { return gen(); }
  uint64_t below(uint64_t n) { return (n == 0) ? 0 : gen() % n; }
  bool chance(uint32_t pct) { return below(100) < pct; }
};

// Strides of at most SRAM/NW, so every slot of every row stays inside the SRAM.
uint32_t random_stride(const dlmem_ref::Cfg& cfg, Rng& rng) {
  uint64_t max_stride = (uint64_t(1) << cfg.lmem_log) / cfg.nw;
  uint32_t pick = rng.below(10);
  if (pick == 0) {
    return 0;
  }
  if (pick == 1) {
    return uint32_t(max_stride);
  }
  return uint32_t(64 * (1 + rng.below(max_stride / 64)));
}

dlmem_ref::Row random_row(const dlmem_ref::Cfg& cfg, uint32_t my_core, Rng& rng) {
  dlmem_ref::Row row;
  row.k = uint32_t(rng.below(cfg.max_cluster + 1));
  for (uint32_t r = 0; r < row.k; ++r) {
    uint32_t core = rng.chance(30) ? my_core : uint32_t(rng.below(cfg.ncc));
    row.peers.push_back({core, uint32_t(rng.below(cfg.nw))});
  }
  return row;
}

const MemOp kAmoOps[] = {MemOp::AMO_LR, MemOp::AMO_SC, MemOp::AMO_SWAP, MemOp::AMO_ADD,
                         MemOp::AMO_AND, MemOp::AMO_OR, MemOp::AMO_XOR, MemOp::AMO_MIN,
                         MemOp::AMO_MAX, MemOp::AMO_CAS};

// A lane address: in-window lanes aimed at the current row, out-of-bounds
// window lanes, window edges, and pass-through addresses.
uint64_t random_addr(const dlmem_ref::Cfg& cfg, const dlmem_ref::Row& row, uint32_t stride,
                     uint32_t size, Rng& rng) {
  uint32_t rb = dlmem_ref::rank_bits(cfg);
  uint64_t sram = uint64_t(1) << cfg.lmem_log;
  uint64_t win = uint64_t(1) << (cfg.lmem_log + rb);
  uint64_t db = cfg.dlmem_base;
  uint64_t addr;
  uint32_t pick = rng.below(100);
  if (pick < 40) {
    uint64_t rank = rng.below(std::max<uint32_t>(row.k, 1));
    addr = db + rank * sram + rng.below(std::max<uint32_t>(stride, 1));
  } else if (pick < 50) {
    uint64_t ranks = uint64_t(1) << rb;
    uint64_t rank = (row.k < ranks) ? row.k + rng.below(ranks - row.k) : rng.below(ranks);
    uint64_t off = (stride < sram && rng.chance(50)) ? stride + rng.below(sram - stride) : rng.below(sram);
    addr = db + rank * sram + off;
  } else if (pick < 60) {
    uint64_t rank = rng.below(uint64_t(1) << rb);
    switch (rng.below(4)) {
    case 0:  addr = db; break;
    case 1:  addr = db + win - size; break;
    case 2:  addr = db + rank * sram + (stride >= size ? stride - size : 0); break;
    default: addr = db + rank * sram + stride; break;
    }
  } else {
    switch (rng.below(9)) {
    case 0:  addr = cfg.lmem_base + rng.below(sram); break;
    case 1:  addr = VX_MEM_IO_BASE_ADDR + rng.below(0x1000); break;
    case 2:  addr = 0x10000 + rng.below(uint64_t(1) << 32); break;
    case 3:  addr = db - size; break;
    case 4:  addr = db - 8; break;
    case 5:  addr = db + win; break;
    case 6:  addr = db + win + 8; break;
    case 7:  addr = db + win + rng.below(uint64_t(1) << 30); break;
    default: addr = dlmem_ref::remote_addr(cfg, uint32_t(rng.below(cfg.ncc)), rng.below(sram)); break;
    }
  }
  return addr & ~uint64_t(size - 1);
}

LsuReq random_req(const dlmem_ref::Cfg& cfg, const TableModel& tm, uint32_t num_lanes,
                  uint64_t uuid, Rng& rng) {
  LsuReq req(num_lanes);
  uint32_t pick = rng.below(10);
  req.op = (pick < 4) ? MemOp::LD : (pick < 8) ? MemOp::ST : kAmoOps[rng.below(sizeof(kAmoOps) / sizeof(kAmoOps[0]))];
  // A fence token, given window addresses here, must leave with every lane untouched.
  bool fence = rng.chance(5);
  if (fence) {
    req.op = MemOp::FLUSH;
    req.flags.fence = 1;
  }
  req.flags.amo_unsigned = memop_is_atomic(req.op) && rng.chance(50);
  req.wid = uint32_t(rng.below(cfg.nw));
  req.cta_id = uint32_t(rng.below(cfg.nw));
  req.tag = uint32_t(rng());
  req.cid = uint32_t(rng.below(64));
  req.uuid = uuid;
  const auto& row = tm.rows.at(req.cta_id);
  for (uint32_t i = 0; i < num_lanes; ++i) {
    uint32_t size = 1u << rng.below(4);
    uint64_t addr = random_addr(cfg, row, tm.stride, size, rng);
    req.addrs[i] = addr;
    req.tids[i] = uint32_t(rng.below(cfg.nt * 4));
    req.amo_cmp[i] = rng();
    // The stage owns the sideband: whatever arrives in it must be overwritten.
    req.dlmem_remote.set(i, rng.chance(50));
    req.dlmem_oob.set(i, rng.chance(50));
    req.dlmem_core[i] = uint32_t(rng());
    // Masked lanes keep a window address, which must pass untouched.
    if (rng.chance(15)) {
      continue;
    }
    req.mask.set(i);
    if (req.op != MemOp::LD && !fence) {
      auto block = std::make_shared<mem_block_t>();
      for (auto& byte : *block) {
        byte = uint8_t(rng());
      }
      req.data[i] = block;
      req.byteen[i] = ((uint64_t(1) << size) - 1) << (addr & (VX_CFG_MEM_BLOCK_SIZE - 1));
    }
  }
  return req;
}

// Output-ready phases: always ready, stalled, or ready half the time.
struct Sink {
  Rng rng;
  uint32_t kind = 0;
  uint32_t left = 0;
  explicit Sink(uint64_t seed) : rng(seed) {}
  bool ready() {
    if (left == 0) {
      uint32_t pick = rng.below(4);
      kind = (pick == 0) ? 1 : (pick == 1) ? 2 : 0;
      left = (kind == 1) ? 20 + rng.below(130) : 20 + rng.below(280);
    }
    --left;
    return kind == 0 || (kind == 2 && rng.chance(50));
  }
};

// Offer phases: back to back, sparse, or idle.
struct Source {
  Rng rng;
  uint32_t kind = 0;
  uint32_t left = 0;
  explicit Source(uint64_t seed) : rng(seed) {}
  bool offer() {
    if (left == 0) {
      kind = uint32_t(rng.below(3));
      left = 10 + rng.below(200);
    }
    --left;
    return kind == 0 || (kind == 1 && rng.chance(30));
  }
};

constexpr uint64_t kWatchdog = 2000;

///////////////////////////////////////////////////////////////////////////////
// Directed: the shared vector table

void run_vectors() {
  auto& platform = SimPlatform::instance();
  if (uint64_t(VX_MEM_DLMEM_BASE_ADDR) != V::kDlmemBase || uint64_t(VX_MEM_LMEM_BASE_ADDR) != V::kLmemBase) {
    fail("vectors", "the memory map moved: DLMEM base " + hex(VX_MEM_DLMEM_BASE_ADDR) + ", LMEM base "
                    + hex(VX_MEM_LMEM_BASE_ADDR) + "; recompute vectors.h");
  }
  uint32_t per_kind[4] = {0, 0, 0, 0};
  for (uint32_t n = 0; n < V::kNumVectors; ++n) {
    const auto& v = V::kVectors[n];
    const auto& s = V::kShapes[v.shape];
    std::string ctx = std::string("vector ") + v.name;
    auto cfg = shape_cfg(s);
    dlmem_ref::Row row;
    row.k = v.k;
    for (uint32_t r = 0; r < v.k; ++r) {
      row.peers.push_back({v.peers[r].core, v.peers[r].slot});
    }

    // The hand-computed value against the golden model.
    auto x = dlmem_ref::translate(cfg, v.addr, row, s.my_core, v.my_slot, v.stride);
    if (uint32_t(x.kind) != v.kind || x.core != v.core || x.out_addr != v.out_addr) {
      fail(ctx, std::string("dlmem_ref gives ") + kind_name(x.kind) + " core " + std::to_string(x.core) + " "
                + hex(x.out_addr) + ", the table " + kind_name(v.kind) + " core " + std::to_string(v.core)
                + " " + hex(v.out_addr));
    }
    check_props(ctx, cfg, v.addr, s.my_core, x);
    ++per_kind[v.kind];

    // The stage: lanes 0 and 2 carry the case, lane 1 is masked, lane 3 is global.
    // A hart id from another cluster checks my_core = core_id % ncc.
    {
      platform.set_num_workers(1);
      uint32_t b = n % 2;
      Harness h(ctx, cfg, s.my_core + 2 * s.ncc, 2);
      platform.reset();
      h.write_row(0, {v.my_slot, row, v.stride});
      std::vector<bool> ready = {true, true};
      const MemOp ops[] = {MemOp::ST, MemOp::LD, MemOp::AMO_ADD};
      uint64_t cycle = 0;
      h.model_tick(cycle);
      platform.tick();
      h.check_outputs(cycle, ready, nullptr);
      for (uint32_t k = 0; k < 3; ++k) {
        LsuReq req(4);
        req.op = ops[k];
        req.wid = 3;
        req.cta_id = v.my_slot;
        req.uuid = n * 8 + k;
        req.tag = 0x5a + k;
        for (uint32_t i = 0; i < 4; ++i) {
          req.addrs[i] = (i == 3) ? 0x10040 : v.addr;
          req.tids[i] = i;
          if (i != 1) {
            req.mask.set(i);
          }
          if (req.op != MemOp::LD && i != 1) {
            req.data[i] = std::make_shared<mem_block_t>();
            (*req.data[i])[req.addrs[i] & 63] = uint8_t(0xa0 + i);
            req.byteen[i] = uint64_t(1) << (req.addrs[i] & 63);
          }
        }
        ++cycle;
        if (h.in_full(b)) {
          fail(ctx, "input full under an always-ready output");
        }
        h.send_req(cycle, b, req);
        h.model_tick(cycle);
        platform.tick();
        h.check_outputs(cycle, ready, nullptr);
      }
      for (uint32_t k = 0; k < 4; ++k) {
        ++cycle;
        h.model_tick(cycle);
        platform.tick();
        h.check_outputs(cycle, ready, nullptr);
      }
      // Every remote or local request counts, whatever its op.
      uint32_t lanes = (v.kind == V::K_REMOTE) ? 2 : 0;
      bool local = (v.kind != V::K_REMOTE) && get_addr_type(v.out_addr) == AddrType::Shared;
      if (h.outputs != 3 || h.wt_events != (lanes ? 3u : 0u) || h.lacc_events != (local ? 3u : 0u)
       || h.lacc_lanes != (local ? 6u : 0u) || !h.drained()) {
        fail(ctx, "stage emitted " + std::to_string(h.outputs) + " requests, "
                  + std::to_string(h.wt_events) + " remote and " + std::to_string(h.lacc_events)
                  + " local-access count events");
      }
    }
    platform.finalize();
  }
  std::cout << "vectors: " << V::kNumVectors << " cases (NONE " << per_kind[0] << ", LOCAL " << per_kind[1]
            << ", REMOTE " << per_kind[2] << ", OOB " << per_kind[3] << ") match dlmem_ref and the stage"
            << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
// Directed: latency, throughput, back-pressure, table visibility, reset

LsuReq window_req(uint64_t uuid, uint32_t cta, uint64_t addr, MemOp op = MemOp::LD) {
  LsuReq req(2);
  req.op = op;
  req.cta_id = cta;
  req.wid = cta;
  req.uuid = uuid;
  req.mask.set(0);
  req.mask.set(1);
  req.addrs[0] = addr;
  req.addrs[1] = addr + 8;
  return req;
}

void run_timing() {
  auto& platform = SimPlatform::instance();
  platform.set_num_workers(1);
  auto cfg = make_cfg(4, 8, 4, 16);
  const uint64_t db = cfg.dlmem_base;
  const uint64_t sram = uint64_t(1) << cfg.lmem_log;
  std::string ctx = "timing";
  Harness h(ctx, cfg, 1, 2);
  platform.reset();

  std::vector<bool> both = {true, true};
  std::vector<bool> stall0 = {false, true};
  std::vector<OutEntry> popped;
  uint64_t cycle = 0;
  uint64_t uuid = 1;
  auto step = [&](const std::vector<bool>& ready) {
    h.model_tick(cycle);
    platform.tick();
    h.check_outputs(cycle, ready, &popped);
    ++cycle;
  };

  // Reset leaves every row empty and the stride zero: a window lane is out of
  // bounds and lands at the base of local memory.
  {
    h.send_req(cycle, 0, window_req(uuid++, 3, db + 0x40));
    step(both);
    step(both);
    if (popped.size() != 1 || popped[0].req.addrs[0] != cfg.lmem_base || !popped[0].req.dlmem_oob.test(0)) {
      fail(ctx, "a window lane after reset is not out of bounds at the local-memory base");
    }
    popped.clear();
  }

  // Rows for every slot: rank 0 is self, rank 1 is core 2.
  for (uint32_t s = 0; s < cfg.nw; ++s) {
    dlmem_ref::Row row{2, {{1, s}, {2, (s + 1) % cfg.nw}}};
    h.write_row(cycle, {s, row, 0x1000});
    step(both);
  }
  step(both);

  // One request per cycle per block: each leaves exactly one tick after the
  // tick it lands, so a block streams at one request per cycle.
  for (uint32_t k = 0; k < 64; ++k) {
    for (uint32_t b = 0; b < 2; ++b) {
      if (h.in_full(b)) {
        fail(ctx, "stream: input full under an always-ready output");
      }
      h.send_req(cycle, b, window_req(uuid++, k % cfg.nw, db + (k % 2) * sram + 8 * k, MemOp::ST));
    }
    step(both);
  }
  step(both);
  if (popped.size() != 128) {
    fail(ctx, "stream: " + std::to_string(popped.size()) + " of 128 requests left");
  }
  for (auto& e : popped) {
    if (e.tick != e.sent + 1) {
      fail(ctx, "stream: request #" + std::to_string(e.req.uuid) + " sent at " + std::to_string(e.sent)
                + " left at tick " + std::to_string(e.tick));
    }
  }
  popped.clear();

  // Block 0 stalled: its stage holds exactly its two buffered requests and
  // refuses the next, while block 1 keeps streaming.
  {
    uint32_t accepted = 0;
    for (uint32_t k = 0; k < 12; ++k) {
      if (!h.in_full(0)) {
        h.send_req(cycle, 0, window_req(uuid++, 1, db + 0x80 + 8 * k));
        ++accepted;
      }
      if (h.in_full(1)) {
        fail(ctx, "back-pressure leaked into block 1");
      }
      h.send_req(cycle, 1, window_req(uuid++, 2, db + 0x100 + 8 * k));
      step(stall0);
    }
    uint32_t held = h.dut->ReqIn.at(0).size();
    uint32_t parked = h.dut->ReqOut.at(0).size();
    if (held != 2 || parked != h.dut->ReqOut.at(0).capacity() || accepted != held + parked
     || !h.dut->ReqIn.at(0).full()) {
      fail(ctx, "stalled block took " + std::to_string(accepted) + " requests: " + std::to_string(held)
                + " in the stage, " + std::to_string(parked) + " at its output");
    }
    // Released, it drains in order at one request per cycle.
    popped.clear();
    for (uint32_t k = 0; k < 6; ++k) {
      step(both);
    }
    if (!h.drained()) {
      fail(ctx, "stalled block did not drain");
    }
  }
  popped.clear();

  // A request takes the row current when it enters the stage, as the RTL: one
  // sent in the cycle of a row write still sees the old row, one sent a cycle
  // later the new row.
  {
    dlmem_ref::Row old_row{2, {{1, 4}, {2, 7}}};
    dlmem_ref::Row new_row{2, {{1, 4}, {3, 5}}};
    h.write_row(cycle, {4, old_row, 0x1000});
    step(both);
    step(both);
    uint64_t addr = db + sram + 0x40;  // rank 1
    h.send_req(cycle, 0, window_req(uuid++, 4, addr));
    step(both);
    h.write_row(cycle, {4, new_row, 0x1000});
    h.send_req(cycle, 0, window_req(uuid++, 4, addr));
    h.send_req(cycle, 1, window_req(uuid++, 4, addr));
    step(both);
    h.send_req(cycle, 0, window_req(uuid++, 4, addr));
    h.send_req(cycle, 1, window_req(uuid++, 4, addr));
    step(both);
    step(both);
    if (popped.size() != 5) {
      fail(ctx, "visibility: " + std::to_string(popped.size()) + " of 5 requests left");
    }
    uint64_t old_out = dlmem_ref::remote_addr(cfg, 2, 7 * 0x1000 + 0x40);
    uint64_t new_out = dlmem_ref::remote_addr(cfg, 3, 5 * 0x1000 + 0x40);
    for (uint32_t k = 0; k < 3; ++k) {
      if (popped[k].req.addrs[0] != old_out || popped[k].req.dlmem_core[0] != 2) {
        fail(ctx, "visibility: request #" + std::to_string(popped[k].req.uuid)
                  + " took a row written in or after the cycle it was sent");
      }
    }
    for (uint32_t k = 3; k < 5; ++k) {
      if (popped[k].req.addrs[0] != new_out || popped[k].req.dlmem_core[0] != 3) {
        fail(ctx, "visibility: request #" + std::to_string(popped[k].req.uuid)
                  + " missed a row written the cycle before it was sent");
      }
    }
  }
  popped.clear();

  // A request waiting in the stage keeps the row it entered with when its
  // slot's row is rewritten before it leaves: the slot can be re-admitted
  // while a committed store still waits here.
  {
    dlmem_ref::Row old_row{2, {{1, 6}, {2, 3}}};
    dlmem_ref::Row new_row{2, {{1, 6}, {0, 1}}};
    h.write_row(cycle, {6, old_row, 0x1000});
    step(both);
    step(both);
    uint64_t addr = db + sram + 0x80;  // rank 1
    // Two requests fill block 0's stalled output, so the third waits in the stage.
    for (uint32_t k = 0; k < 2; ++k) {
      h.send_req(cycle, 0, window_req(uuid++, 2, db + 0x40 + 8 * k));
      step(stall0);
    }
    uint64_t held = uuid;
    h.send_req(cycle, 0, window_req(uuid++, 6, addr, MemOp::ST));
    step(stall0);
    h.write_row(cycle, {6, new_row, 0x1000});
    for (uint32_t k = 0; k < 4; ++k) {
      step(stall0);
    }
    if (h.dut->ReqIn.at(0).size() != 1 || !h.dut->ReqOut.at(0).full()) {
      fail(ctx, "held: the request is not waiting in the stage");
    }
    uint64_t late = uuid;
    h.send_req(cycle, 1, window_req(uuid++, 6, addr, MemOp::ST));
    for (uint32_t k = 0; k < 4; ++k) {
      step(both);
    }
    uint64_t old_out = dlmem_ref::remote_addr(cfg, 2, 3 * 0x1000 + 0x80);
    uint64_t new_out = dlmem_ref::remote_addr(cfg, 0, 1 * 0x1000 + 0x80);
    uint32_t seen = 0;
    for (auto& e : popped) {
      if (e.req.uuid == held) {
        ++seen;
        if (e.req.addrs[0] != old_out || e.req.dlmem_core[0] != 2) {
          fail(ctx, "held: a waiting request took a row written after it entered");
        }
      } else if (e.req.uuid == late) {
        ++seen;
        if (e.req.addrs[0] != new_out || e.req.dlmem_core[0] != 0) {
          fail(ctx, "held: a request entering after the rewrite missed the new row");
        }
      }
    }
    if (seen != 2 || !h.drained()) {
      fail(ctx, "held: " + std::to_string(seen) + " of the 2 checked requests left");
    }
  }

  for (uint32_t k = 0; k < 4; ++k) {
    step(both);
  }
  if (!h.drained() || !platform.idle()) {
    fail(ctx, "not idle at the end");
  }
  platform.finalize();
  std::cout << "timing: +1 cycle, 1 request/cycle/block, back-pressure at 2, row taken on entry, reset: ok"
            << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
// Random traffic over several cores, run flat, in domains, and on threads

enum class Variant { Flat, Domains, Threads };

const char* variant_name(Variant v) {
  switch (v) {
  case Variant::Flat:    return "flat";
  case Variant::Domains: return "domains";
  case Variant::Threads: return "threads";
  }
  return "?";
}

struct RandomShape {
  uint32_t shape;       // index into V::kShapes
  uint32_t blocks;
  uint32_t lanes;
};

struct RandomResult {
  uint64_t digest = 0xcbf29ce484222325ull;
  uint64_t cycles = 0;
  uint64_t requests = 0;
  uint64_t fences = 0;
  uint64_t wt_events = 0;
  uint64_t lacc_events = 0;
  uint64_t lacc_lanes = 0;
  uint64_t kinds[4] = {0, 0, 0, 0};

  void mix(uint64_t v) {
    for (int i = 0; i < 8; ++i) {
      digest ^= (v >> (8 * i)) & 0xff;
      digest *= 0x100000001b3ull;
    }
  }
};

constexpr uint32_t kCores = 3;

RandomResult run_random(const RandomShape& rs, Variant variant, uint32_t seed, uint32_t reqs,
                        uint32_t workers) {
  auto& platform = SimPlatform::instance();
  platform.set_num_workers(variant == Variant::Threads ? workers : 1);
  const auto& s = V::kShapes[rs.shape];
  auto cfg = shape_cfg(s);
  std::string base = std::string("random ") + s.name + " " + variant_name(variant);

  RandomResult res;
  {
    std::vector<std::unique_ptr<Harness>> hs;
    std::vector<Rng> gen;
    std::vector<std::vector<Source>> srcs(kCores);
    std::vector<std::vector<Sink>> sinks(kCores);
    std::vector<std::vector<uint32_t>> sent(kCores, std::vector<uint32_t>(rs.blocks, 0));
    for (uint32_t c = 0; c < kCores; ++c) {
      uint32_t domain = (variant == Variant::Flat) ? 0 : platform.alloc_domain();
      SimPlatform::DomainScope scope(domain);
      uint32_t core_id = c + s.ncc * c;
      hs.emplace_back(new Harness(base + " core" + std::to_string(c), cfg, core_id, rs.blocks));
      gen.emplace_back(uint64_t(seed) * 7919 + c * 104729 + rs.shape);
      for (uint32_t b = 0; b < rs.blocks; ++b) {
        srcs[c].emplace_back(uint64_t(seed) * 1000003 + rs.shape * 7907 + c * 131 + b);
        sinks[c].emplace_back(uint64_t(seed) * 999983 + rs.shape * 7919 + c * 137 + b + 1);
      }
    }
    platform.reset();

    uint64_t total = uint64_t(kCores) * rs.blocks * reqs;
    uint64_t done = 0;
    uint64_t last_progress = 0;
    uint64_t uuid = 1;
    std::vector<bool> ready(rs.blocks);
    uint64_t cycle = 0;
    for (; done < total; ++cycle) {
      for (uint32_t c = 0; c < kCores; ++c) {
        auto& h = *hs[c];
        auto& rng = gen[c];
        // Rows for every slot first, then a rewrite now and then.
        if (cycle < cfg.nw || rng.chance(4)) {
          uint32_t slot = (cycle < cfg.nw) ? uint32_t(cycle) : uint32_t(rng.below(cfg.nw));
          h.write_row(cycle, {slot, random_row(cfg, h.my_core, rng), random_stride(cfg, rng)});
        }
        for (uint32_t b = 0; b < rs.blocks; ++b) {
          bool full = h.in_full(b);
          if (sent[c][b] == reqs || !srcs[c][b].offer() || full) {
            continue;
          }
          h.send_req(cycle, b, random_req(cfg, h.tm, rs.lanes, uuid++, rng));
          ++sent[c][b];
          last_progress = cycle;
        }
        h.model_tick(cycle);
      }
      platform.tick();
      for (uint32_t c = 0; c < kCores; ++c) {
        auto& h = *hs[c];
        for (uint32_t b = 0; b < rs.blocks; ++b) {
          ready[b] = sinks[c][b].ready();
        }
        std::vector<OutEntry> popped;
        h.check_outputs(cycle, ready, &popped);
        for (auto& e : popped) {
          res.mix(cycle); res.mix(c); res.mix(e.req.uuid);
          for (uint32_t i = 0; i < e.req.mask.size(); ++i) {
            res.mix(e.req.addrs[i]);
          }
          ++done;
          last_progress = cycle;
        }
      }
      if (cycle - last_progress > kWatchdog) {
        fail(base, "no progress for " + std::to_string(kWatchdog) + " cycles");
      }
    }
    for (uint32_t i = 0; i < 4 && !platform.idle(); ++i) {
      platform.tick();
    }
    for (auto& h : hs) {
      if (!h->drained()) {
        fail(base, "requests left in the model");
      }
      for (uint32_t b = 0; b < rs.blocks; ++b) {
        if (!h->dut->ReqIn.at(b).empty() || !h->dut->ReqOut.at(b).empty() || !h->dut->WtIncOut.at(b).empty()
         || !h->dut->LaccIncOut.at(b).empty()) {
          fail(base, "stage not empty at the end");
        }
      }
      res.wt_events += h->wt_events;
      res.lacc_events += h->lacc_events;
      res.lacc_lanes += h->lacc_lanes;
      res.requests += h->outputs;
      res.fences += h->fences;
      for (uint32_t k = 0; k < 4; ++k) {
        res.kinds[k] += h->kinds[k];
      }
    }
    if (!platform.idle()) {
      fail(base, "platform not idle at the end");
    }
    res.cycles = cycle;
  }
  platform.finalize();
  return res;
}

void run_random_all(uint32_t seed, uint32_t reqs, uint32_t workers) {
  // The RTL testbench's shapes, plus NCC 16 > MAX 8.
  const RandomShape shapes[] = {{0, 1, 4}, {1, 1, 4}, {2, 2, 4}, {3, 1, 8}, {4, 2, 2}, {5, 2, 8}};
  for (auto& rs : shapes) {
    RandomResult ref;
    for (auto variant : {Variant::Flat, Variant::Domains, Variant::Threads}) {
      RandomResult r = run_random(rs, variant, seed, reqs, workers);
      if (variant == Variant::Flat) {
        ref = r;
        continue;
      }
      if (r.digest != ref.digest || r.cycles != ref.cycles || r.wt_events != ref.wt_events
       || r.lacc_events != ref.lacc_events || r.lacc_lanes != ref.lacc_lanes) {
        fail(std::string("random ") + V::kShapes[rs.shape].name,
             std::string("variant '") + variant_name(variant) + "' differs from 'flat'");
      }
    }
    // A one-core cluster has no remote peer.
    bool need_remote = V::kShapes[rs.shape].ncc > 1;
    if (ref.kinds[dlmem_ref::NONE] == 0 || ref.kinds[dlmem_ref::LOCAL] == 0 || ref.kinds[dlmem_ref::OOB] == 0
     || ref.fences == 0 || ref.lacc_events == 0
     || (need_remote && (ref.kinds[dlmem_ref::REMOTE] == 0 || ref.wt_events == 0))) {
      fail(std::string("random ") + V::kShapes[rs.shape].name, "traffic missed a lane kind");
    }
    std::cout << "random " << V::kShapes[rs.shape].name << " blocks=" << rs.blocks << " lanes=" << rs.lanes
              << ": requests=" << ref.requests << " cycles=" << ref.cycles
              << " local=" << ref.kinds[1] << " remote=" << ref.kinds[2] << " oob=" << ref.kinds[3]
              << " pass=" << ref.kinds[0] << " fences=" << ref.fences << " wt_inc=" << ref.wt_events
              << " lacc_inc=" << ref.lacc_events << "/" << ref.lacc_lanes
              << " digest=" << std::hex << ref.digest << std::dec
              << " (flat=domains=threads x" << workers << ")" << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Fences: untranslated, in order, through a stage that aborts on window lanes

void run_fence() {
  auto& platform = SimPlatform::instance();
  platform.set_num_workers(1);
  auto cfg = make_cfg(2, 4, 2, 14);
  const uint64_t db = cfg.dlmem_base;
  const uint64_t sram = uint64_t(1) << cfg.lmem_log;
  std::string ctx = "fence";
  Harness h(ctx, cfg, 0, 1, /*aborts*/ true);
  platform.reset();
  std::vector<bool> ready = {true};
  std::vector<OutEntry> popped;
  uint64_t cycle = 0;
  auto step = [&]() {
    h.model_tick(cycle);
    platform.tick();
    h.check_outputs(cycle, ready, &popped);
    ++cycle;
  };
  // Slot 2: rank 0 is self, rank 1 is core 1. Slot 3 keeps the empty reset row.
  h.write_row(cycle, {2, dlmem_ref::Row{2, {{0, 2}, {1, 3}}}, 0x1000});
  step();
  // Fence lanes on a remote rank and on the empty row's window would abort if translated.
  auto fence = [&](uint64_t uuid, uint32_t cta) {
    LsuReq req = window_req(uuid, cta, db + sram + 0x40, MemOp::FLUSH);
    req.flags.fence = 1;
    req.dlmem_remote.set(0);
    req.dlmem_oob.set(1);
    req.dlmem_core[1] = 1;
    return req;
  };
  h.send_req(cycle, 0, window_req(1, 2, db + 0x40));
  step();
  h.send_req(cycle, 0, fence(2, 2));
  step();
  h.send_req(cycle, 0, fence(3, 3));
  step();
  h.send_req(cycle, 0, window_req(4, 2, db + 0x80, MemOp::ST));
  step();
  for (uint32_t k = 0; k < 4; ++k) {
    step();
  }
  if (popped.size() != 4 || h.fences != 2 || h.wt_events != 0 || !h.drained()) {
    fail(ctx, std::to_string(popped.size()) + " of 4 requests left, " + std::to_string(h.fences) + " fences");
  }
  for (uint32_t k = 0; k < 4; ++k) {
    auto& e = popped[k];
    if (e.req.uuid != k + 1 || e.tick != e.sent + 1) {
      fail(ctx, "request #" + std::to_string(e.req.uuid) + " sent at " + std::to_string(e.sent)
                + " left at tick " + std::to_string(e.tick) + " in position " + std::to_string(k));
    }
  }
  for (uint32_t k : {1u, 2u}) {
    auto& r = popped[k].req;
    if (r.addrs[0] != db + sram + 0x40 || r.addrs[1] != db + sram + 0x48 || r.dlmem_remote.any()
     || r.dlmem_oob.any() || r.dlmem_core[0] != 0 || r.dlmem_core[1] != 0) {
      fail(ctx, "fence #" + std::to_string(r.uuid) + " was translated");
    }
  }
  if (popped[0].req.addrs[0] != cfg.lmem_base + 2 * 0x1000 + 0x40) {
    fail(ctx, "the load ahead of the fences was not translated");
  }
  platform.finalize();
  std::cout << "fence: untranslated, in order, +1 cycle, sideband cleared: ok" << std::endl;
}

// The fence hold against the real access tracker, created after the stage as
// in the core: a fence right behind its warp's remote store, load or AMO waits
// for the tracker's registered drain, holds the block behind it, and leaves the
// cycle after the last completion is counted; another warp's fence on another
// block does not wait.
void run_fence_hold(MemOp op) {
  auto& platform = SimPlatform::instance();
  platform.set_num_workers(1);
  auto cfg = make_cfg(2, 4, 2, 14);
  const uint64_t db = cfg.dlmem_base;
  const uint64_t sram = uint64_t(1) << cfg.lmem_log;
  std::string ctx = std::string("fence-hold-") + (op == MemOp::ST ? "st" : (op == MemOp::LD ? "ld" : "amo"));
  auto dut = DlmemXlate::Create("fence-hold", cfg, 0, 2, true, false);
  auto wt = DlmemWTrack::Create("fence-hold-wtrack", 2, cfg.nw, 255);
  auto lacc = DlmemWTrack::Create("fence-hold-lacc", 2, cfg.nw, 255, 4);
  for (uint32_t b = 0; b < 2; ++b) {
    dut->WtIncOut.at(b).bind(&wt->IncIn.at(b));
    dut->LaccIncOut.at(b).bind(&lacc->IncIn.at(b));
  }
  dut->set_drained_query([w = wt.get(), l = lacc.get()](uint32_t wid) { return w->drained(wid) && l->drained(wid); });
  platform.reset();

  uint64_t cycle = 0;
  std::vector<int64_t> left(8, -1);  // tick each uuid left the stage
  int64_t zero_tick = -1;            // tick the tracker's count of warp 2 returned to zero
  auto step = [&]() {
    platform.tick();
    for (uint32_t b = 0; b < 2; ++b) {
      auto& out = dut->ReqOut.at(b);
      while (!out.empty()) {
        left.at(out.peek().uuid) = cycle;
        out.pop();
      }
    }
    if (zero_tick < 0 && left[1] >= 0 && wt->count(2) == 0) {
      zero_tick = cycle;
    }
    ++cycle;
  };
  auto fence = [&](uint64_t uuid, uint32_t wid) {
    LsuReq req = window_req(uuid, wid, 0, MemOp::FLUSH);
    req.flags.fence = 1;
    return req;
  };

  // Slot 2: rank 1 lives on core 1, so both lanes of the access are remote.
  dut->PeerWrIn.send({2, dlmem_ref::Row{2, {{0, 2}, {1, 3}}}, 0x1000}, 1);
  step();
  dut->ReqIn.at(0).send(window_req(1, 2, db + sram + 0x40, op), 1);
  step();
  dut->ReqIn.at(0).send(fence(2, 2), 1);
  dut->ReqIn.at(1).send(fence(3, 1), 1);
  step();
  if (left[1] != 2 || wt->count(2) != 2 || wt->drained(2)) {
    fail(ctx, "the remote access left at tick " + std::to_string(left[1]) + " counting "
              + std::to_string(wt->count(2)) + " (drained=" + std::to_string(wt->drained(2)) + ")");
  }
  dut->ReqIn.at(0).send(window_req(4, 2, db + 0x80), 1);
  for (uint32_t k = 0; k < 6; ++k) {
    step();
  }
  auto check_held = [&]() {
    if (left[2] >= 0 || left[4] >= 0) {
      fail(ctx, "the fence of warp 2 left at tick " + std::to_string(left[2]) + " and the load behind it at "
                + std::to_string(left[4]) + " with " + std::to_string(wt->count(2)) + " accesses unperformed");
    }
  };
  check_held();
  if (left[3] != 3) {
    fail(ctx, "the fence of warp 1 left at tick " + std::to_string(left[3]) + ", expected 3");
  }
  // The egress reports one completion per cycle: a store's ack, a load's or AMO's done.
  for (uint32_t k = 0; k < 2; ++k) {
    (op == MemOp::ST ? wt->DecIn : wt->DoneIn).send(2, 1);
    step();
    check_held();
  }
  for (uint32_t k = 0; k < 4; ++k) {
    step();
  }
  if (zero_tick < 0 || left[2] != zero_tick + 1 || left[4] != zero_tick + 2) {
    fail(ctx, "the last completion was counted at tick " + std::to_string(zero_tick) + ", the fence left at "
              + std::to_string(left[2]) + " and the load at " + std::to_string(left[4]));
  }
  // The load behind the fence is a local access: its two lanes commit.
  if (lacc->count(2) != 2) {
    fail(ctx, "the local load counted " + std::to_string(lacc->count(2)) + " lanes, expected 2");
  }
  lacc->CommitIn.at(0).send(2, 1);
  lacc->CommitIn.at(1).send(2, 1);
  for (uint32_t k = 0; k < 3; ++k) {
    platform.tick();
  }
  if (!dut->empty() || !wt->empty() || !lacc->empty() || !platform.idle()) {
    fail(ctx, "stage or tracker not empty at the end");
  }
  platform.finalize();
  std::cout << ctx << ": held behind its warp's remote accesses until the registered drain (+1 cycle): ok"
            << std::endl;
}

const char* op_name(MemOp op) {
  switch (op) {
  case MemOp::ST:      return "st";
  case MemOp::LD:      return "ld";
  case MemOp::AMO_ADD: return "amo";
  case MemOp::AMO_LR:  return "lr";
  case MemOp::AMO_SC:  return "sc";
#ifdef VX_CFG_EXT_MBAR_ENABLE
  case MemOp::MBAR_ARRIVE:    return "mbar_arrive";
  case MemOp::MBAR_ARRIVE_TX: return "mbar_arrive_tx";
  case MemOp::MBAR_EXPECT_TX:   return "mbar_expect_tx";
  case MemOp::MBAR_COMPLETE_TX: return "mbar_complete_tx";
  case MemOp::MBAR_TEST_WAIT: return "mbar_test_wait";
  case MemOp::MBAR_TRY_WAIT:  return "mbar_try_wait";
#endif
  default:             return "op";
  }
}

// The fence hold on local accesses, with both trackers as the core has them:
// the stage counts every local lane of a load, store or atomic (the warp's own
// rank and plain local memory) when it leaves (delay 0, so the count is visible
// after that tick), and the fence waits for both trackers. A bank commit pulse
// sent at tick T with delay 1 is counted at T+1 and the fence leaves at T+2.
void run_fence_hold_local(MemOp op) {
  auto& platform = SimPlatform::instance();
  platform.set_num_workers(1);
  constexpr uint32_t kBanks = 4;
  auto cfg = make_cfg(2, 4, 2, 14);
  const uint64_t db = cfg.dlmem_base;
  const uint64_t sram = uint64_t(1) << cfg.lmem_log;
  std::string ctx = std::string("fence-hold-local-") + op_name(op);
  auto dut = DlmemXlate::Create("fence-hold-local", cfg, 0, 2, true, false);
  auto wt = DlmemWTrack::Create("fence-hold-local-wtrack", 2, cfg.nw, 255);
  auto lacc = DlmemWTrack::Create("fence-hold-local-lacc", 2, cfg.nw, 255, kBanks);
  if (lacc->CommitIn.size() != kBanks || wt->CommitIn.size() != 0) {
    fail(ctx, "one commit port per bank expected");
  }
  for (uint32_t b = 0; b < 2; ++b) {
    dut->WtIncOut.at(b).bind(&wt->IncIn.at(b));
    dut->LaccIncOut.at(b).bind(&lacc->IncIn.at(b));
  }
  dut->set_drained_query([w = wt.get(), l = lacc.get()](uint32_t wid) { return w->drained(wid) && l->drained(wid); });
  platform.reset();

  uint64_t cycle = 0;
  std::vector<int64_t> left(16, -1);  // tick each uuid left the stage
  auto step = [&](uint32_t n = 1) {
    for (uint32_t k = 0; k < n; ++k) {
      platform.tick();
      for (uint32_t b = 0; b < 2; ++b) {
        auto& out = dut->ReqOut.at(b);
        while (!out.empty()) {
          left.at(out.peek().uuid) = cycle;
          out.pop();
        }
      }
      ++cycle;
    }
  };
  auto fence = [&](uint64_t uuid, uint32_t wid) {
    LsuReq req = window_req(uuid, wid, 0, MemOp::FLUSH);
    req.flags.fence = 1;
    return req;
  };
  auto check_count = [&](uint32_t want, const char* when) {
    if (lacc->count(2) != want || lacc->drained(2) != (want == 0)) {
      fail(ctx, std::string(when) + ": local-access count " + std::to_string(lacc->count(2)) + " (drained="
                + std::to_string(lacc->drained(2)) + "), expected " + std::to_string(want));
    }
  };
  auto commit = [&](std::initializer_list<uint32_t> banks) {
    for (uint32_t bank : banks) {
      lacc->CommitIn.at(bank).send(2, 1);
    }
  };
  auto check_held = [&](uint64_t fence_id, uint64_t load_id) {
    if (left[fence_id] >= 0 || left[load_id] >= 0) {
      fail(ctx, "the fence of warp 2 left at tick " + std::to_string(left[fence_id]) + " and the load behind it at "
                + std::to_string(left[load_id]) + " with " + std::to_string(lacc->count(2))
                + " local and " + std::to_string(wt->count(2)) + " remote lanes outstanding");
    }
  };

  // Slot 2: rank 0 is this core (slot 2), rank 1 lives on core 1.
  dut->PeerWrIn.send({2, dlmem_ref::Row{2, {{0, 2}, {1, 3}}}, 0x1000}, 1);
  step();
  // Both lanes on the warp's own rank; then one plain local lane and one
  // global lane (an mbarrier op has no global lane: it would abort).
  dut->ReqIn.at(0).send(window_req(1, 2, db + 0x40, op), 1);
  step();
  {
    LsuReq req = window_req(2, 2, cfg.lmem_base + 0x200, op);
    req.addrs[1] = 0x10040;
#ifdef VX_CFG_EXT_MBAR_ENABLE
    if (memop_is_mbar(op)) {
      req.mask.reset(1);
    }
#endif
    dut->ReqIn.at(0).send(req, 1);
  }
  step();
  if (left[1] != 2) {
    fail(ctx, "the self-rank access left at tick " + std::to_string(left[1]) + ", expected 2");
  }
  check_count(2, "after the self-rank access left");
  dut->ReqIn.at(0).send(fence(3, 2), 1);
  dut->ReqIn.at(1).send(fence(4, 1), 1);
  step();
  if (left[2] != 3) {
    fail(ctx, "the plain local access left at tick " + std::to_string(left[2]) + ", expected 3");
  }
  check_count(3, "after the plain local access left");
  dut->ReqIn.at(0).send(window_req(5, 2, db + 0x80), 1);
  step(6);
  if (left[4] != 4) {
    fail(ctx, "the fence of warp 1 left at tick " + std::to_string(left[4]) + ", expected 4");
  }
  if (wt->count(2) != 0) {
    fail(ctx, "local accesses counted as remote");
  }
  check_held(3, 5);
  // Two banks commit in one cycle: both pulses land in one tracker tick.
  commit({0, 3});
  step();
  check_held(3, 5);
  step();
  check_count(1, "after two commits in one cycle");
  check_held(3, 5);
  // The last commit pulse, sent at tick T.
  uint64_t T = cycle;
  commit({1});
  step();
  check_count(1, "in the tick of the last commit");
  check_held(3, 5);
  step();
  check_count(0, "the tick after the last commit");
  check_held(3, 5);
  step(3);
  if (int64_t(T) + 2 != left[3] || int64_t(T) + 3 != left[5]) {
    fail(ctx, "last commit at tick " + std::to_string(T) + ", the fence left at " + std::to_string(left[3])
              + " and the load at " + std::to_string(left[5]) + ", expected T+2 and T+3");
  }
  // The load behind the fence counts its own lanes.
  check_count(2, "after the load behind the fence left");
  commit({0, 1});
  step(3);
  check_count(0, "after the load's commits");

  // Remote and local lanes outstanding together: the fence waits for both trackers.
  dut->ReqIn.at(0).send(window_req(6, 2, db + sram + 0x40, MemOp::ST), 1);
  step();
  dut->ReqIn.at(0).send(window_req(7, 2, db + 0x100, op), 1);
  step();
  dut->ReqIn.at(0).send(fence(8, 2), 1);
  step();
  dut->ReqIn.at(0).send(window_req(9, 2, db + 0xc0), 1);
  step(4);
  if (left[6] < 0 || left[7] < 0 || wt->count(2) != 2) {
    fail(ctx, "mixed: the accesses did not leave or the remote count is " + std::to_string(wt->count(2)));
  }
  check_count(2, "mixed: after the local access left");
  check_held(8, 9);
  commit({0, 1});
  step(4);
  check_count(0, "mixed: after the local commits");
  check_held(8, 9);
  int64_t ack = -1;
  for (uint32_t k = 0; k < 2; ++k) {
    ack = cycle;
    wt->DecIn.send(2, 1);
    step();
    check_held(8, 9);
  }
  step(4);
  if (left[8] != ack + 2 || left[9] != ack + 3) {
    fail(ctx, "mixed: last remote ack sent at " + std::to_string(ack) + ", the fence left at "
              + std::to_string(left[8]) + " and the load at " + std::to_string(left[9]));
  }
  commit({2, 3});
  step(3);
  if (!dut->empty() || !wt->empty() || !lacc->empty() || !platform.idle()) {
    fail(ctx, "stage or trackers not empty at the end");
  }
  platform.finalize();
  std::cout << ctx << ": held behind its warp's local accesses until the commit (pulse +2 cycles)"
            << " and behind both trackers together: ok" << std::endl;
}

// The release hold, with both trackers as the core has them: a release op
// (an AMO with .rl, or an mbarrier arrive) right behind its warp's uncommitted
// local access, then behind its unacknowledged remote store, waits as a fence
// does and holds the load behind it; it leaves in the cycle the registered
// drain rises. A plain AMO in the same spot leaves right behind the access,
// and a drained warp's release op on another block is not held.
void run_rl_hold(MemOp first, MemOp rel) {
  auto& platform = SimPlatform::instance();
  platform.set_num_workers(1);
  constexpr uint32_t kBanks = 4;
  auto cfg = make_cfg(2, 4, 2, 14);
  const uint64_t db = cfg.dlmem_base;
  const uint64_t sram = uint64_t(1) << cfg.lmem_log;
  const uint64_t amo_addr = cfg.lmem_base + 0x300;
  std::string ctx = std::string("rl-hold-") + op_name(first) + "-" + op_name(rel);
  auto dut = DlmemXlate::Create("rl-hold", cfg, 0, 2, true, false);
  auto wt = DlmemWTrack::Create("rl-hold-wtrack", 2, cfg.nw, 255);
  auto lacc = DlmemWTrack::Create("rl-hold-lacc", 2, cfg.nw, 255, kBanks);
  for (uint32_t b = 0; b < 2; ++b) {
    dut->WtIncOut.at(b).bind(&wt->IncIn.at(b));
    dut->LaccIncOut.at(b).bind(&lacc->IncIn.at(b));
  }
  dut->set_drained_query([w = wt.get(), l = lacc.get()](uint32_t wid) { return w->drained(wid) && l->drained(wid); });
  platform.reset();

  uint64_t cycle = 0;
  std::vector<int64_t> left(32, -1);  // tick each uuid left the stage
  auto step = [&](uint32_t n = 1) {
    for (uint32_t k = 0; k < n; ++k) {
      platform.tick();
      for (uint32_t b = 0; b < 2; ++b) {
        auto& out = dut->ReqOut.at(b);
        while (!out.empty()) {
          auto& r = out.peek();
          if (r.amo_rl != (r.uuid == 3 || r.uuid == 4 || r.uuid == 9)) {
            fail(ctx, "request #" + std::to_string(r.uuid) + " left with amo_rl=" + std::to_string(r.amo_rl));
          }
          left.at(r.uuid) = cycle;
          out.pop();
        }
      }
      ++cycle;
    }
  };
  auto amo = [&](uint64_t uuid, uint32_t wid, bool rl) {
    LsuReq req = window_req(uuid, wid, amo_addr, rl ? rel : MemOp::AMO_ADD);
    req.addrs[1] = amo_addr + 8;
    req.amo_rl = rl;
    return req;
  };
  auto check_held = [&](const char* what, uint64_t amo_id, uint64_t load_id) {
    if (left[amo_id] >= 0 || left[load_id] >= 0) {
      fail(ctx, std::string(what) + ": the release op left at tick " + std::to_string(left[amo_id])
                + " and the load behind it at " + std::to_string(left[load_id]) + " with "
                + std::to_string(lacc->count(2)) + " local and " + std::to_string(wt->count(2))
                + " remote lanes outstanding");
    }
  };
  // Commits one lane per bank for warp `wid`, then lets them land.
  auto commit_all = [&](uint32_t wid, uint32_t lanes) {
    for (uint32_t k = 0; k < lanes; ++k) {
      lacc->CommitIn.at(k % kBanks).send(wid, 1 + k / kBanks);
    }
    step(3 + lanes / kBanks);
  };

  // Slot 2: rank 0 is this core (slot 2), rank 1 lives on core 1.
  dut->PeerWrIn.send({2, dlmem_ref::Row{2, {{0, 2}, {1, 3}}}, 0x1000}, 1);
  step();

  // A plain local-memory access, then warp 2's release op and a load behind it
  // on block 0, while drained warp 1 sends a release op on block 1.
  dut->ReqIn.at(0).send(window_req(1, 2, cfg.lmem_base + 0x200, first), 1);
  step();
  dut->ReqIn.at(0).send(amo(3, 2, true), 1);
  dut->ReqIn.at(1).send(amo(4, 1, true), 1);
  uint64_t sent = cycle;
  step();
  if (left[1] != 2 || lacc->count(2) != 2) {
    fail(ctx, "local: the access left at tick " + std::to_string(left[1]) + " counting " + std::to_string(lacc->count(2)));
  }
  dut->ReqIn.at(0).send(window_req(5, 2, db + 0x80), 1);
  step(6);
  check_held("local", 3, 5);
  if (left[4] != int64_t(sent) + 1 || lacc->count(1) != 2) {
    fail(ctx, "the drained warp's release op left at tick " + std::to_string(left[4]) + ", expected "
              + std::to_string(sent + 1) + ", counting " + std::to_string(lacc->count(1)));
  }
  // Both commit pulses sent at tick T: the tracker counts them at T+1, the op leaves at T+2.
  uint64_t T = cycle;
  lacc->CommitIn.at(0).send(2, 1);
  lacc->CommitIn.at(1).send(2, 1);
  step();
  check_held("local, in the tick of the commits", 3, 5);
  step();
  check_held("local, the tick after the commits", 3, 5);
  step(3);
  if (int64_t(T) + 2 != left[3] || int64_t(T) + 3 != left[5]) {
    fail(ctx, "local: commits at tick " + std::to_string(T) + ", the release op left at "
              + std::to_string(left[3]) + " and the load at " + std::to_string(left[5]) + ", expected T+2 and T+3");
  }
  // The release op and the load behind it are local accesses too.
  if (lacc->count(2) != 4) {
    fail(ctx, "local: " + std::to_string(lacc->count(2)) + " lanes counted after the release op and the load, expected 4");
  }
  commit_all(2, 4);
  commit_all(1, 2);

  // A remote store, then the release op and a load behind it.
  dut->ReqIn.at(0).send(window_req(6, 2, db + sram + 0x40, MemOp::ST), 1);
  step();
  dut->ReqIn.at(0).send(amo(9, 2, true), 1);
  step();
  if (left[6] < 0 || wt->count(2) != 2) {
    fail(ctx, "remote: the store did not leave or the remote count is " + std::to_string(wt->count(2)));
  }
  dut->ReqIn.at(0).send(window_req(10, 2, db + 0xc0), 1);
  step(6);
  check_held("remote", 9, 10);
  int64_t ack = -1;
  for (uint32_t k = 0; k < 2; ++k) {
    ack = cycle;
    wt->DecIn.send(2, 1);
    step();
    check_held("remote, during the acks", 9, 10);
  }
  step(4);
  if (left[9] != ack + 2 || left[10] != ack + 3) {
    fail(ctx, "remote: last ack sent at " + std::to_string(ack) + ", the release op left at "
              + std::to_string(left[9]) + " and the load at " + std::to_string(left[10]) + ", expected +2 and +3");
  }
  commit_all(2, 4);

  // A plain AMO right behind an uncommitted local access is not held.
  dut->ReqIn.at(0).send(window_req(11, 2, cfg.lmem_base + 0x240, first), 1);
  step();
  dut->ReqIn.at(0).send(amo(12, 2, false), 1);
  step(2);
  if (left[11] < 0 || left[12] != left[11] + 1 || lacc->count(2) != 4) {
    fail(ctx, "plain: the access left at tick " + std::to_string(left[11]) + " and the plain AMO at "
              + std::to_string(left[12]) + " with " + std::to_string(lacc->count(2)) + " local lanes outstanding");
  }
  commit_all(2, 4);
  if (!dut->empty() || !wt->empty() || !lacc->empty() || !platform.idle()) {
    fail(ctx, "stage or trackers not empty at the end");
  }
  platform.finalize();
  std::cout << ctx << ": a release op waits behind its warp's local accesses and remote accesses like a fence;"
            << " a plain AMO and a drained warp's release op do not: ok" << std::endl;
}

#ifdef VX_CFG_EXT_MBAR_ENABLE
// Arrive and tx ops on rank 1 (core 1): each leaves the next tick as a remote
// AMO, flagged remote, at core 1's address of the word, op and rl unchanged,
// counted by the remote tracker only. Behind the warp's unacknowledged remote
// store, a release arrive (or arrive_tx) waits for the drain like a fence; a
// relaxed one and an expect_tx leave right behind the store.
void run_mbar_remote() {
  auto& platform = SimPlatform::instance();
  platform.set_num_workers(1);
  constexpr uint32_t kBanks = 4;
  auto cfg = make_cfg(2, 4, 2, 14);
  const uint64_t db = cfg.dlmem_base;
  const uint64_t sram = uint64_t(1) << cfg.lmem_log;
  const uint64_t stride = 0x1000;
  const uint64_t rank1 = db + sram + 0x300;
  const std::string ctx = "mbar-remote";
  auto dut = DlmemXlate::Create("mbar-remote", cfg, 0, 1, true, false);
  auto wt = DlmemWTrack::Create("mbar-remote-wtrack", 1, cfg.nw, 255);
  auto lacc = DlmemWTrack::Create("mbar-remote-lacc", 1, cfg.nw, 255, kBanks);
  dut->WtIncOut.at(0).bind(&wt->IncIn.at(0));
  dut->LaccIncOut.at(0).bind(&lacc->IncIn.at(0));
  dut->set_drained_query([w = wt.get(), l = lacc.get()](uint32_t wid) { return w->drained(wid) && l->drained(wid); });
  platform.reset();

  uint64_t cycle = 0;
  std::vector<int64_t> left(64, -1);
  std::vector<LsuReq> out_reqs(64, LsuReq(2));
  auto step = [&](uint32_t n = 1) {
    for (uint32_t k = 0; k < n; ++k) {
      platform.tick();
      auto& out = dut->ReqOut.at(0);
      while (!out.empty()) {
        left.at(out.peek().uuid) = cycle;
        out_reqs.at(out.peek().uuid) = out.peek();
        out.pop();
      }
      ++cycle;
    }
  };
  auto mbar = [&](uint64_t uuid, MemOp op, bool rl) {
    LsuReq req = window_req(uuid, 2, rank1, op);
    req.amo_rl = rl;
    return req;
  };
  auto ack_all = [&]() {
    for (uint32_t k = wt->count(2); k != 0; --k) {
      wt->DecIn.send(2, 1);
      step();
    }
    step(2);
  };

  // Slot 2: rank 0 is this core (slot 2), rank 1 is slot 3 of core 1.
  dut->PeerWrIn.send({2, dlmem_ref::Row{2, {{0, 2}, {1, 3}}}, uint32_t(stride)}, 1);
  step();

  struct OpCase {
    MemOp op;
    bool rl;
  };
  const OpCase ops[] = {
    {MemOp::MBAR_ARRIVE, true}, {MemOp::MBAR_ARRIVE, false},
    {MemOp::MBAR_ARRIVE_TX, true}, {MemOp::MBAR_ARRIVE_TX, false},
    {MemOp::MBAR_EXPECT_TX, false}, {MemOp::MBAR_COMPLETE_TX, false},
  };
  uint64_t uuid = 1;
  for (auto& c : ops) {
    const std::string what = std::string(c.rl ? "release " : "relaxed ") + op_name(c.op);
    const uint64_t id = uuid++;
    const uint64_t sent = cycle;
    dut->ReqIn.at(0).send(mbar(id, c.op, c.rl), 1);
    step(3);
    const LsuReq& r = out_reqs.at(id);
    if (left[id] != int64_t(sent) + 1 || r.op != c.op || r.amo_rl != c.rl) {
      fail(ctx, what + ": left at tick " + std::to_string(left[id]) + " (expected " + std::to_string(sent + 1)
                + ") as " + op_name(r.op) + " rl=" + std::to_string(r.amo_rl));
    }
    for (uint32_t i = 0; i < 2; ++i) {
      const uint64_t want = dlmem_ref::remote_addr(cfg, 1, 3 * stride + 0x300 + 8 * i);
      if (!r.dlmem_remote.test(i) || r.dlmem_oob.test(i) || r.dlmem_core.at(i) != 1 || r.addrs.at(i) != want) {
        fail(ctx, what + ": lane " + std::to_string(i) + " left remote=" + std::to_string(r.dlmem_remote.test(i))
                  + " core=" + std::to_string(r.dlmem_core.at(i)) + " addr=" + hex(r.addrs.at(i))
                  + ", expected a remote lane to core 1 at " + hex(want));
      }
    }
    if (wt->count(2) != 2 || lacc->count(2) != 0 || r.flags.lacc) {
      fail(ctx, what + ": " + std::to_string(wt->count(2)) + " remote and " + std::to_string(lacc->count(2))
                + " local lanes counted, expected 2 and 0");
    }
    ack_all();
  }

  // Behind an unacknowledged remote store.
  const OpCase behind[] = {
    {MemOp::MBAR_ARRIVE, true}, {MemOp::MBAR_ARRIVE_TX, true},
    {MemOp::MBAR_ARRIVE, false}, {MemOp::MBAR_ARRIVE_TX, false}, {MemOp::MBAR_EXPECT_TX, false},
  };
  for (auto& c : behind) {
    const std::string what = std::string(c.rl ? "release " : "relaxed ") + op_name(c.op) + " behind a remote store";
    const uint64_t st = uuid++, id = uuid++;
    dut->ReqIn.at(0).send(window_req(st, 2, db + sram + 0x40, MemOp::ST), 1);
    step();
    dut->ReqIn.at(0).send(mbar(id, c.op, c.rl), 1);
    step(6);
    if (left[st] < 0) {
      fail(ctx, what + ": the store did not leave");
    }
    if (!c.rl) {
      if (left[id] != left[st] + 1) {
        fail(ctx, what + ": left at tick " + std::to_string(left[id]) + ", expected right behind the store at "
                  + std::to_string(left[st] + 1));
      }
      ack_all();
      continue;
    }
    if (left[id] >= 0) {
      fail(ctx, what + ": left at tick " + std::to_string(left[id]) + " with " + std::to_string(wt->count(2))
                + " remote lanes outstanding");
    }
    int64_t ack = -1;
    for (uint32_t k = 0; k < 2; ++k) {
      ack = cycle;
      wt->DecIn.send(2, 1);
      step();
    }
    step(4);
    if (left[id] != ack + 2) {
      fail(ctx, what + ": last ack sent at " + std::to_string(ack) + ", left at " + std::to_string(left[id])
                + ", expected +2");
    }
    ack_all();
  }
  if (!dut->empty() || !wt->empty() || !lacc->empty() || !platform.idle()) {
    fail(ctx, "stage or trackers not empty at the end");
  }
  platform.finalize();
  std::cout << ctx << ": arrive, arrive_tx, expect_tx and complete_tx leave as remote AMOs to core 1, op unchanged,"
            << " counted as remote accesses; behind a remote store a release arrive waits for the drain, a relaxed"
            << " one does not: ok" << std::endl;
}
#endif

///////////////////////////////////////////////////////////////////////////////
// Out-of-bounds and remote aborts, each in a child process

std::string run_child(const std::function<void()>& body, const std::string& ctx, bool expect_abort) {
  int fds[2];
  if (pipe(fds) != 0) {
    fail(ctx, "pipe failed");
  }
  std::cout.flush();
  std::cerr.flush();
  pid_t pid = fork();
  if (pid < 0) {
    fail(ctx, "fork failed");
  }
  if (pid == 0) {
    close(fds[0]);
    dup2(fds[1], STDERR_FILENO);
    body();
    _exit(0);
  }
  close(fds[1]);
  std::string err;
  char buf[256];
  ssize_t n;
  while ((n = read(fds[0], buf, sizeof(buf))) > 0) {
    err.append(buf, n);
  }
  close(fds[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  bool aborted = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
  bool clean = WIFEXITED(status) && WEXITSTATUS(status) == 0;
  if (expect_abort ? !aborted : !clean) {
    fail(ctx, std::string(expect_abort ? "the simulation did not abort" : "the simulation did not finish")
              + ": " + err);
  }
  return err;
}

// One in-bounds request, then `bad` on block 0 of core 0, under slot 2's row.
void abort_body(const char* name, const dlmem_ref::Row& row, const LsuReq& bad, bool abort_on_remote,
                uint32_t want_wt) {
  auto& platform = SimPlatform::instance();
  platform.set_num_workers(1);
  auto cfg = make_cfg(2, 4, 2, 14);
  auto dut = abort_on_remote ? DlmemXlate::Create(name, cfg, 0, 1)
                             : DlmemXlate::Create(name, cfg, 0, 1, true, false);
  platform.reset();
  dut->PeerWrIn.send({2, row, 0x1000}, 1);
  platform.tick();
  dut->ReqIn.at(0).send(window_req(7, 2, cfg.dlmem_base + 0x40), 1);  // rank 0: self
  platform.tick();
  dut->ReqIn.at(0).send(bad, 1);
  uint32_t wt = 0;
  for (int i = 0; i < 4; ++i) {
    platform.tick();
    while (!dut->ReqOut.at(0).empty()) {
      dut->ReqOut.at(0).pop();
    }
    while (!dut->WtIncOut.at(0).empty()) {
      auto& w = dut->WtIncOut.at(0).peek();
      if (w.wid != bad.wid) {
        _exit(3);
      }
      wt += w.cnt;
      dut->WtIncOut.at(0).pop();
    }
  }
  _exit(wt == want_wt ? 0 : 4);
}

void run_abort_case(const char* name, const dlmem_ref::Row& row, const LsuReq& bad, bool abort_on_remote,
                    bool expect_abort, const std::string& expect, uint32_t want_wt);

void run_abort() {
  auto cfg = make_cfg(2, 4, 2, 14);
  const uint64_t db = cfg.dlmem_base;
  const uint64_t sram = uint64_t(1) << cfg.lmem_log;
  struct Case {
    const char* name;
    dlmem_ref::Row row;
    uint64_t addr;
    bool abort_on_remote;
    bool expect_abort;
    std::string expect;
  };
  // Rank 1 of a one-member row is out of bounds; of a two-member row whose
  // rank 1 lives on core 1, it is remote.
  uint64_t oob = db + sram + 5;
  uint64_t rem = db + sram + 0x40;
  const Case cases[] = {
    {"xlate-oob", dlmem_ref::Row{1, {{0, 2}}}, oob, true, true,
     "DLMEM access out of bounds: wid=2 cta=2 lane=0 addr=" + hex(oob & ~uint64_t(LSU_WORD_SIZE - 1)) + " (#8)"},
    {"xlate-remote", dlmem_ref::Row{2, {{0, 2}, {1, 3}}}, rem, true, true,
     "remote DLMEM access is not supported yet: wid=2 cta=2 lane=0 addr=" + hex(rem) + " core=1 (#8)"},
    {"xlate-remote-count", dlmem_ref::Row{2, {{0, 2}, {1, 3}}}, rem, false, false, ""},
  };
  for (auto& c : cases) {
    LsuReq bad = window_req(8, 2, c.addr, MemOp::ST);
    run_abort_case(c.name, c.row, bad, c.abort_on_remote, c.expect_abort, c.expect, 2);
  }
#ifdef VX_CFG_EXT_MBAR_ENABLE
  // An mbarrier op aborts on a global lane, and a wait on a remote lane,
  // whatever the stage's remote setting; arrive and tx ops on remote lanes
  // are remote accesses like any AMO, and local lanes pass.
  const dlmem_ref::Row two{2, {{0, 2}, {1, 3}}};
  struct MbarCase {
    const char* name;
    MemOp op;
    uint64_t addr0, addr1;
    bool abort_on_remote;
    uint32_t want_wt;
    std::string expect;
  };
  const MbarCase mcases[] = {
    {"mbar-global", MemOp::MBAR_ARRIVE, 0x10040, 0x10048, false, 0,
     std::string(mbar_ref::MSG_GLOBAL) + ": wid=2 cta=2 lane=0 addr=0x10040 (#8)"},
    {"mbar-global-lane1", MemOp::MBAR_ARRIVE, db + 0x40, 0x10048, false, 0,
     std::string(mbar_ref::MSG_GLOBAL) + ": wid=2 cta=2 lane=1 addr=0x10048 (#8)"},
    {"mbar-remote", MemOp::MBAR_ARRIVE, rem, rem + 8, false, 2, ""},
    {"mbar-remote-lane1", MemOp::MBAR_COMPLETE_TX, cfg.lmem_base + 0x40, rem + 8, false, 1, ""},
    {"mbar-remote-unsupported", MemOp::MBAR_ARRIVE_TX, rem, rem + 8, true, 0,
     "remote DLMEM access is not supported yet: wid=2 cta=2 lane=0 addr=" + hex(rem) + " core=1 (#8)"},
    {"mbar-remote-wait", MemOp::MBAR_TEST_WAIT, rem, rem + 8, false, 0,
     std::string(mbar_ref::MSG_REMOTE) + ": wid=2 cta=2 lane=0 addr=" + hex(rem) + " core=1 (#8)"},
    {"mbar-remote-trywait", MemOp::MBAR_TRY_WAIT, rem, rem + 8, false, 0,
     std::string(mbar_ref::MSG_REMOTE) + ": wid=2 cta=2 lane=0 addr=" + hex(rem) + " core=1 (#8)"},
    {"mbar-remote-wait-lane1", MemOp::MBAR_TEST_WAIT, cfg.lmem_base + 0x40, rem + 8, true, 0,
     std::string(mbar_ref::MSG_REMOTE) + ": wid=2 cta=2 lane=1 addr=" + hex(rem + 8) + " core=1 (#8)"},
    {"mbar-local", MemOp::MBAR_ARRIVE, db + 0x40, cfg.lmem_base + 0x48, true, 0, ""},
  };
  for (auto& c : mcases) {
    LsuReq bad = window_req(8, 2, c.addr0, c.op);
    bad.addrs[1] = c.addr1;
    bad.amo_rl = (c.op == MemOp::MBAR_ARRIVE || c.op == MemOp::MBAR_ARRIVE_TX);
    run_abort_case(c.name, two, bad, c.abort_on_remote, !c.expect.empty(), c.expect, c.want_wt);
  }
#endif
}

void run_abort_case(const char* name, const dlmem_ref::Row& row, const LsuReq& bad, bool abort_on_remote,
                    bool expect_abort, const std::string& expect, uint32_t want_wt) {
  std::string err = run_child([&]() { abort_body(name, row, bad, abort_on_remote, want_wt); },
                              std::string("abort ") + name, expect_abort);
  if (!expect_abort) {
    std::cout << "abort: " << name << " passes (" << want_wt << " remote lanes counted)" << std::endl;
    return;
  }
  if (err.find(expect) == std::string::npos) {
    fail(std::string("abort ") + name, "unexpected abort message: " + err);
  }
  std::cout << "abort: " << name << " aborts with: " << err.substr(err.find(' ') + 1);
}

void usage() {
  std::cout << "usage: dlmem_xlate_test [--mode=vectors|timing|fence|rl|mbar-remote|random|abort|all] [--seed=N]\n"
               "                        [--reqs=N] [--workers=N]\n";
}

} // namespace

int main(int argc, char** argv) {
  std::string mode = "all";
  uint32_t seed = 1;
  uint32_t reqs = 1500;
  uint32_t workers = (SIMX_NUM_WORKERS > 1) ? SIMX_NUM_WORKERS : 4;
  for (int a = 1; a < argc; ++a) {
    std::string arg = argv[a];
    auto val = [&](const char* key) -> const char* {
      size_t n = std::strlen(key);
      return (arg.compare(0, n, key) == 0) ? arg.c_str() + n : nullptr;
    };
    if (auto v = val("--mode=")) {
      mode = v;
    } else if (auto v = val("--seed=")) {
      seed = std::atoi(v);
    } else if (auto v = val("--reqs=")) {
      reqs = std::max(1, std::atoi(v));
    } else if (auto v = val("--workers=")) {
      workers = std::max(2, std::atoi(v));
    } else {
      usage();
      return 2;
    }
  }
  if (mode != "all" && mode != "vectors" && mode != "timing" && mode != "fence" && mode != "rl"
   && mode != "mbar-remote" && mode != "random" && mode != "abort") {
    usage();
    return 2;
  }
  // Fork before any run has started worker threads.
  if (mode == "all" || mode == "abort") {
    run_abort();
  }
  if (mode == "all" || mode == "vectors") {
    run_vectors();
  }
  if (mode == "all" || mode == "timing") {
    run_timing();
  }
  if (mode == "all" || mode == "fence") {
    run_fence();
    for (MemOp op : {MemOp::ST, MemOp::LD, MemOp::AMO_ADD}) {
      run_fence_hold(op);
    }
    for (MemOp op : {MemOp::ST, MemOp::LD, MemOp::AMO_ADD, MemOp::AMO_LR, MemOp::AMO_SC}) {
      run_fence_hold_local(op);
    }
#ifdef VX_CFG_EXT_MBAR_ENABLE
    for (MemOp op : {MemOp::MBAR_ARRIVE, MemOp::MBAR_TEST_WAIT}) {
      run_fence_hold_local(op);
    }
#endif
  }
  if (mode == "all" || mode == "rl") {
    for (MemOp op : {MemOp::ST, MemOp::LD, MemOp::AMO_ADD, MemOp::AMO_LR}) {
      run_rl_hold(op, MemOp::AMO_ADD);
    }
#ifdef VX_CFG_EXT_MBAR_ENABLE
    for (MemOp op : {MemOp::ST, MemOp::LD, MemOp::MBAR_TEST_WAIT, MemOp::MBAR_ARRIVE}) {
      run_rl_hold(op, MemOp::MBAR_ARRIVE);
    }
    run_rl_hold(MemOp::AMO_ADD, MemOp::MBAR_ARRIVE_TX);
#endif
  }
#ifdef VX_CFG_EXT_MBAR_ENABLE
  if (mode == "all" || mode == "mbar-remote") {
    run_mbar_remote();
  }
#endif
  if (mode == "all" || mode == "random") {
    run_random_all(seed, reqs, workers);
  }
  std::cout << "PASSED" << std::endl;
  return 0;
}
