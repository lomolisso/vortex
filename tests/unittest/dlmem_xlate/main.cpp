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

// Lanes of a translated request that are plain stores into this core's local
// memory: the switch's local leg, restricted to stores (an AMO counts nothing).
uint32_t local_store_lanes(const LsuReq& out) {
  if (out.op != MemOp::ST || out.flags.fence) {
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
  std::deque<WtInc> lst; // expected this tick
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
  uint64_t lst_events = 0;
  uint64_t lst_lanes = 0;
  uint64_t kinds[4] = {0, 0, 0, 0};  // active lanes per dlmem_ref::Kind

  Harness(const std::string& name, const dlmem_ref::Cfg& c, uint32_t core_id, uint32_t nb,
          bool aborts = false)
    : ctx(name), cfg(c), my_core(core_id % c.ncc), num_blocks(nb), blocks(nb) {
    dut = DlmemXlate::Create(name.c_str(), cfg, core_id, nb, aborts, aborts);
    tm.reset(cfg.nw);
    if (dut->ReqIn.size() != nb || dut->ReqOut.size() != nb || dut->WtIncOut.size() != nb
     || dut->LstIncOut.size() != nb) {
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
      bm.lst.clear();
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
      uint32_t local = local_store_lanes(e);
      if (local != 0) {
        bm.lst.push_back(WtInc{ie.req.wid, local});
      }
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
      auto& lst = dut->LstIncOut.at(b);
      if (lst.size() != bm.lst.size()) {
        fail(ctx, "local-store count events " + std::to_string(lst.size()) + ", expected "
                  + std::to_string(bm.lst.size()) + bctx);
      }
      for (auto& w : bm.lst) {
        auto& g = lst.peek();
        if (g.wid != w.wid || g.cnt != w.cnt) {
          fail(ctx, "local-store count {wid=" + std::to_string(g.wid) + ", cnt=" + std::to_string(g.cnt)
                    + "}, expected {" + std::to_string(w.wid) + ", " + std::to_string(w.cnt) + "}" + bctx);
        }
        lst.pop();
        ++lst_events;
        lst_lanes += w.cnt;
      }
      bm.lst.clear();
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
      // Every remote request counts, whatever its op; a local lane only as a store.
      uint32_t lanes = (v.kind == V::K_REMOTE) ? 2 : 0;
      bool local = (v.kind != V::K_REMOTE) && get_addr_type(v.out_addr) == AddrType::Shared;
      if (h.outputs != 3 || h.wt_events != (lanes ? 3u : 0u) || h.lst_events != (local ? 1u : 0u)
       || h.lst_lanes != (local ? 2u : 0u) || !h.drained()) {
        fail(ctx, "stage emitted " + std::to_string(h.outputs) + " requests, "
                  + std::to_string(h.wt_events) + " remote and " + std::to_string(h.lst_events)
                  + " local-store count events");
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
  uint64_t lst_events = 0;
  uint64_t lst_lanes = 0;
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
         || !h->dut->LstIncOut.at(b).empty()) {
          fail(base, "stage not empty at the end");
        }
      }
      res.wt_events += h->wt_events;
      res.lst_events += h->lst_events;
      res.lst_lanes += h->lst_lanes;
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
       || r.lst_events != ref.lst_events || r.lst_lanes != ref.lst_lanes) {
        fail(std::string("random ") + V::kShapes[rs.shape].name,
             std::string("variant '") + variant_name(variant) + "' differs from 'flat'");
      }
    }
    // A one-core cluster has no remote peer.
    bool need_remote = V::kShapes[rs.shape].ncc > 1;
    if (ref.kinds[dlmem_ref::NONE] == 0 || ref.kinds[dlmem_ref::LOCAL] == 0 || ref.kinds[dlmem_ref::OOB] == 0
     || ref.fences == 0 || ref.lst_events == 0
     || (need_remote && (ref.kinds[dlmem_ref::REMOTE] == 0 || ref.wt_events == 0))) {
      fail(std::string("random ") + V::kShapes[rs.shape].name, "traffic missed a lane kind");
    }
    std::cout << "random " << V::kShapes[rs.shape].name << " blocks=" << rs.blocks << " lanes=" << rs.lanes
              << ": requests=" << ref.requests << " cycles=" << ref.cycles
              << " local=" << ref.kinds[1] << " remote=" << ref.kinds[2] << " oob=" << ref.kinds[3]
              << " pass=" << ref.kinds[0] << " fences=" << ref.fences << " wt_inc=" << ref.wt_events
              << " lst_inc=" << ref.lst_events << "/" << ref.lst_lanes
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
  for (uint32_t b = 0; b < 2; ++b) {
    dut->WtIncOut.at(b).bind(&wt->IncIn.at(b));
  }
  dut->set_drained_query([w = wt.get()](uint32_t wid) { return w->drained(wid); });
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
  if (!dut->empty() || !wt->empty() || !platform.idle()) {
    fail(ctx, "stage or tracker not empty at the end");
  }
  platform.finalize();
  std::cout << ctx << ": held behind its warp's remote accesses until the registered drain (+1 cycle): ok"
            << std::endl;
}

// The fence hold on local stores, with both trackers as the core has them: the
// stage counts the local lanes of a store to the warp's own rank and of a plain
// local-memory store when they leave (delay 0, so the count is visible after
// that tick), and the fence waits for both trackers. A bank write at tick T
// pulses the commit with delay 1, the tracker counts it at T+1 and the fence
// leaves at T+2; a host send before tick T lands as a bank send in tick T does.
// Loads and AMOs to local addresses count nothing and hold nothing.
void run_fence_hold_local(MemOp op) {
  auto& platform = SimPlatform::instance();
  platform.set_num_workers(1);
  constexpr uint32_t kBanks = 4;
  auto cfg = make_cfg(2, 4, 2, 14);
  const uint64_t db = cfg.dlmem_base;
  const uint64_t sram = uint64_t(1) << cfg.lmem_log;
  const bool st = (op == MemOp::ST);
  std::string ctx = std::string("fence-hold-local-") + (st ? "st" : (op == MemOp::LD ? "ld" : "amo"));
  auto dut = DlmemXlate::Create("fence-hold-local", cfg, 0, 2, true, false);
  auto wt = DlmemWTrack::Create("fence-hold-local-wtrack", 2, cfg.nw, 255);
  auto lst = DlmemWTrack::Create("fence-hold-local-lst", 2, cfg.nw, 255, kBanks);
  if (lst->CommitIn.size() != kBanks || wt->CommitIn.size() != 0) {
    fail(ctx, "one commit port per bank expected");
  }
  for (uint32_t b = 0; b < 2; ++b) {
    dut->WtIncOut.at(b).bind(&wt->IncIn.at(b));
    dut->LstIncOut.at(b).bind(&lst->IncIn.at(b));
  }
  dut->set_drained_query([w = wt.get(), l = lst.get()](uint32_t wid) { return w->drained(wid) && l->drained(wid); });
  platform.reset();

  uint64_t cycle = 0;
  std::vector<int64_t> left(16, -1);  // tick each uuid left the stage
  auto step = [&]() {
    platform.tick();
    for (uint32_t b = 0; b < 2; ++b) {
      auto& out = dut->ReqOut.at(b);
      while (!out.empty()) {
        left.at(out.peek().uuid) = cycle;
        out.pop();
      }
    }
    ++cycle;
  };
  auto fence = [&](uint64_t uuid, uint32_t wid) {
    LsuReq req = window_req(uuid, wid, 0, MemOp::FLUSH);
    req.flags.fence = 1;
    return req;
  };
  auto check_count = [&](uint32_t want, const char* when) {
    if (lst->count(2) != want || lst->drained(2) != (want == 0)) {
      fail(ctx, std::string(when) + ": local-store count " + std::to_string(lst->count(2)) + " (drained="
                + std::to_string(lst->drained(2)) + "), expected " + std::to_string(want));
    }
  };
  auto commit = [&](std::initializer_list<uint32_t> banks) {
    for (uint32_t bank : banks) {
      lst->CommitIn.at(bank).send(2, 1);
    }
  };

  // Slot 2: rank 0 is this core (slot 2), rank 1 lives on core 1.
  dut->PeerWrIn.send({2, dlmem_ref::Row{2, {{0, 2}, {1, 3}}}, 0x1000}, 1);
  step();
  // Both lanes on the warp's own rank; then one plain local lane and one global lane.
  dut->ReqIn.at(0).send(window_req(1, 2, db + 0x40, op), 1);
  step();
  {
    LsuReq req = window_req(2, 2, cfg.lmem_base + 0x200, op);
    req.addrs[1] = 0x10040;
    dut->ReqIn.at(0).send(req, 1);
  }
  step();
  if (left[1] != 2) {
    fail(ctx, "the self-rank access left at tick " + std::to_string(left[1]) + ", expected 2");
  }
  check_count(st ? 2 : 0, "after the self-rank access left");
  dut->ReqIn.at(0).send(fence(3, 2), 1);
  dut->ReqIn.at(1).send(fence(4, 1), 1);
  step();
  if (left[2] != 3) {
    fail(ctx, "the plain local access left at tick " + std::to_string(left[2]) + ", expected 3");
  }
  check_count(st ? 3 : 0, "after the plain local access left");
  dut->ReqIn.at(0).send(window_req(5, 2, db + 0x80), 1);
  for (uint32_t k = 0; k < 6; ++k) {
    step();
  }
  if (left[4] != 4) {
    fail(ctx, "the fence of warp 1 left at tick " + std::to_string(left[4]) + ", expected 4");
  }
  if (wt->count(2) != 0) {
    fail(ctx, "local accesses counted as remote");
  }
  if (!st) {
    if (left[3] != 4 || left[5] != 5) {
      fail(ctx, "the fence behind local accesses left at tick " + std::to_string(left[3])
                + " and the load at " + std::to_string(left[5]) + ", expected 4 and 5");
    }
    if (!dut->empty() || !wt->empty() || !lst->empty() || !platform.idle()) {
      fail(ctx, "stage or trackers not empty at the end");
    }
    platform.finalize();
    std::cout << ctx << ": local loads and AMOs count nothing and do not hold the fence: ok" << std::endl;
    return;
  }

  auto check_held = [&](uint64_t fence_id, uint64_t load_id) {
    if (left[fence_id] >= 0 || left[load_id] >= 0) {
      fail(ctx, "the fence of warp 2 left at tick " + std::to_string(left[fence_id]) + " and the load behind it at "
                + std::to_string(left[load_id]) + " with " + std::to_string(lst->count(2))
                + " local and " + std::to_string(wt->count(2)) + " remote lanes outstanding");
    }
  };
  check_held(3, 5);
  // Two banks write in one cycle: both commits land in one tracker tick.
  commit({0, 3});
  step();
  check_held(3, 5);
  step();
  check_count(1, "after two commits in one cycle");
  check_held(3, 5);
  // The last bank write, at tick T.
  uint64_t T = cycle;
  commit({1});
  step();
  check_count(1, "in the tick of the last bank write");
  check_held(3, 5);
  step();
  check_count(0, "the tick after the last bank write");
  check_held(3, 5);
  for (uint32_t k = 0; k < 3; ++k) {
    step();
  }
  if (int64_t(T) + 2 != left[3] || int64_t(T) + 3 != left[5]) {
    fail(ctx, "last bank write at tick " + std::to_string(T) + ", the fence left at " + std::to_string(left[3])
              + " and the load at " + std::to_string(left[5]) + ", expected T+2 and T+3");
  }

  // Remote and local lanes outstanding together: the fence waits for both trackers.
  dut->ReqIn.at(0).send(window_req(6, 2, db + sram + 0x40, MemOp::ST), 1);
  step();
  dut->ReqIn.at(0).send(window_req(7, 2, db + 0x100, MemOp::ST), 1);
  step();
  dut->ReqIn.at(0).send(fence(8, 2), 1);
  step();
  dut->ReqIn.at(0).send(window_req(9, 2, db + 0xc0), 1);
  for (uint32_t k = 0; k < 4; ++k) {
    step();
  }
  if (left[6] < 0 || left[7] < 0 || wt->count(2) != 2) {
    fail(ctx, "mixed: the stores did not leave or the remote count is " + std::to_string(wt->count(2)));
  }
  check_count(2, "mixed: after the local store left");
  check_held(8, 9);
  commit({0, 1});
  for (uint32_t k = 0; k < 4; ++k) {
    step();
  }
  check_count(0, "mixed: after the local commits");
  check_held(8, 9);
  int64_t ack = -1;
  for (uint32_t k = 0; k < 2; ++k) {
    ack = cycle;
    wt->DecIn.send(2, 1);
    step();
    check_held(8, 9);
  }
  for (uint32_t k = 0; k < 4; ++k) {
    step();
  }
  if (left[8] != ack + 2 || left[9] != ack + 3) {
    fail(ctx, "mixed: last remote ack sent at " + std::to_string(ack) + ", the fence left at "
              + std::to_string(left[8]) + " and the load at " + std::to_string(left[9]));
  }
  if (!dut->empty() || !wt->empty() || !lst->empty() || !platform.idle()) {
    fail(ctx, "stage or trackers not empty at the end");
  }
  platform.finalize();
  std::cout << ctx << ": held behind its warp's local stores until the commit (bank write +2 cycles)"
            << " and behind both trackers together: ok" << std::endl;
}

// The release AMO hold, with both trackers as the core has them: an AMO with
// .rl right behind its warp's uncommitted plain local-memory store, then
// behind its unacknowledged remote store, waits as a fence does and holds the
// load behind it; it leaves in the cycle the registered drain rises. A plain
// AMO in the same spot leaves right behind the store, and a drained warp's
// release AMO on another block is not held.
void run_rl_hold() {
  auto& platform = SimPlatform::instance();
  platform.set_num_workers(1);
  constexpr uint32_t kBanks = 4;
  auto cfg = make_cfg(2, 4, 2, 14);
  const uint64_t db = cfg.dlmem_base;
  const uint64_t sram = uint64_t(1) << cfg.lmem_log;
  const uint64_t amo_addr = cfg.lmem_base + 0x300;
  std::string ctx = "rl-hold";
  auto dut = DlmemXlate::Create("rl-hold", cfg, 0, 2, true, false);
  auto wt = DlmemWTrack::Create("rl-hold-wtrack", 2, cfg.nw, 255);
  auto lst = DlmemWTrack::Create("rl-hold-lst", 2, cfg.nw, 255, kBanks);
  for (uint32_t b = 0; b < 2; ++b) {
    dut->WtIncOut.at(b).bind(&wt->IncIn.at(b));
    dut->LstIncOut.at(b).bind(&lst->IncIn.at(b));
  }
  dut->set_drained_query([w = wt.get(), l = lst.get()](uint32_t wid) { return w->drained(wid) && l->drained(wid); });
  platform.reset();

  uint64_t cycle = 0;
  std::vector<int64_t> left(32, -1);  // tick each uuid left the stage
  auto step = [&]() {
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
  };
  auto amo = [&](uint64_t uuid, uint32_t wid, bool rl) {
    LsuReq req = window_req(uuid, wid, amo_addr, MemOp::AMO_ADD);
    req.addrs[1] = amo_addr + 8;
    req.amo_rl = rl;
    return req;
  };
  auto check_held = [&](const char* what, uint64_t amo_id, uint64_t load_id) {
    if (left[amo_id] >= 0 || left[load_id] >= 0) {
      fail(ctx, std::string(what) + ": the release AMO left at tick " + std::to_string(left[amo_id])
                + " and the load behind it at " + std::to_string(left[load_id]) + " with "
                + std::to_string(lst->count(2)) + " local and " + std::to_string(wt->count(2))
                + " remote lanes outstanding");
    }
  };

  // Slot 2: rank 0 is this core (slot 2), rank 1 lives on core 1.
  dut->PeerWrIn.send({2, dlmem_ref::Row{2, {{0, 2}, {1, 3}}}, 0x1000}, 1);
  step();

  // A plain local-memory store, then warp 2's release AMO and a load behind it
  // on block 0, while drained warp 1 sends a release AMO on block 1.
  dut->ReqIn.at(0).send(window_req(1, 2, cfg.lmem_base + 0x200, MemOp::ST), 1);
  step();
  dut->ReqIn.at(0).send(amo(3, 2, true), 1);
  dut->ReqIn.at(1).send(amo(4, 1, true), 1);
  uint64_t sent = cycle;
  step();
  if (left[1] != 2 || lst->count(2) != 2) {
    fail(ctx, "local: the store left at tick " + std::to_string(left[1]) + " counting " + std::to_string(lst->count(2)));
  }
  dut->ReqIn.at(0).send(window_req(5, 2, db + 0x80), 1);
  for (uint32_t k = 0; k < 6; ++k) {
    step();
  }
  check_held("local", 3, 5);
  if (left[4] != int64_t(sent) + 1) {
    fail(ctx, "the drained warp's release AMO left at tick " + std::to_string(left[4]) + ", expected "
              + std::to_string(sent + 1));
  }
  // Both bank writes at tick T: the tracker counts them at T+1, the AMO leaves at T+2.
  uint64_t T = cycle;
  lst->CommitIn.at(0).send(2, 1);
  lst->CommitIn.at(1).send(2, 1);
  step();
  check_held("local, in the tick of the bank writes", 3, 5);
  step();
  check_held("local, the tick after the bank writes", 3, 5);
  for (uint32_t k = 0; k < 3; ++k) {
    step();
  }
  if (int64_t(T) + 2 != left[3] || int64_t(T) + 3 != left[5]) {
    fail(ctx, "local: bank writes at tick " + std::to_string(T) + ", the release AMO left at "
              + std::to_string(left[3]) + " and the load at " + std::to_string(left[5]) + ", expected T+2 and T+3");
  }

  // A remote store, then the release AMO and a load behind it.
  dut->ReqIn.at(0).send(window_req(6, 2, db + sram + 0x40, MemOp::ST), 1);
  step();
  dut->ReqIn.at(0).send(amo(9, 2, true), 1);
  step();
  if (left[6] < 0 || wt->count(2) != 2) {
    fail(ctx, "remote: the store did not leave or the remote count is " + std::to_string(wt->count(2)));
  }
  dut->ReqIn.at(0).send(window_req(10, 2, db + 0xc0), 1);
  for (uint32_t k = 0; k < 6; ++k) {
    step();
  }
  check_held("remote", 9, 10);
  int64_t ack = -1;
  for (uint32_t k = 0; k < 2; ++k) {
    ack = cycle;
    wt->DecIn.send(2, 1);
    step();
    check_held("remote, during the acks", 9, 10);
  }
  for (uint32_t k = 0; k < 4; ++k) {
    step();
  }
  if (left[9] != ack + 2 || left[10] != ack + 3) {
    fail(ctx, "remote: last ack sent at " + std::to_string(ack) + ", the release AMO left at "
              + std::to_string(left[9]) + " and the load at " + std::to_string(left[10]) + ", expected +2 and +3");
  }

  // A plain AMO right behind an uncommitted local store is not held.
  dut->ReqIn.at(0).send(window_req(11, 2, cfg.lmem_base + 0x240, MemOp::ST), 1);
  step();
  dut->ReqIn.at(0).send(amo(12, 2, false), 1);
  step();
  step();
  if (left[11] < 0 || left[12] != left[11] + 1 || lst->count(2) != 2) {
    fail(ctx, "plain: the store left at tick " + std::to_string(left[11]) + " and the plain AMO at "
              + std::to_string(left[12]) + " with " + std::to_string(lst->count(2)) + " local lanes outstanding");
  }
  lst->CommitIn.at(0).send(2, 1);
  lst->CommitIn.at(1).send(2, 1);
  for (uint32_t k = 0; k < 3; ++k) {
    step();
  }
  if (!dut->empty() || !wt->empty() || !lst->empty() || !platform.idle()) {
    fail(ctx, "stage or trackers not empty at the end");
  }
  platform.finalize();
  std::cout << ctx << ": a release AMO waits behind its warp's local stores and remote accesses like a fence;"
            << " a plain AMO and a drained warp's release AMO do not: ok" << std::endl;
}

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
    std::string err = run_child([&]() { abort_body(c.name, c.row, bad, c.abort_on_remote, 2); },
                                std::string("abort ") + c.name, c.expect_abort);
    if (!c.expect_abort) {
      std::cout << "abort: abort_on_remote=false passes remote lanes and counts them (2 lanes)" << std::endl;
      continue;
    }
    if (err.find(c.expect) == std::string::npos) {
      fail(std::string("abort ") + c.name, "unexpected abort message: " + err);
    }
    std::cout << "abort: " << c.name << " aborts with: " << err.substr(err.find(' ') + 1);
  }
}

void usage() {
  std::cout << "usage: dlmem_xlate_test [--mode=vectors|timing|fence|rl|random|abort|all] [--seed=N]\n"
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
   && mode != "random" && mode != "abort") {
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
    for (MemOp op : {MemOp::ST, MemOp::LD, MemOp::AMO_ADD}) {
      run_fence_hold_local(op);
    }
  }
  if (mode == "all" || mode == "rl") {
    run_rl_hold();
  }
  if (mode == "all" || mode == "random") {
    run_random_all(seed, reqs, workers);
  }
  std::cout << "PASSED" << std::endl;
  return 0;
}
