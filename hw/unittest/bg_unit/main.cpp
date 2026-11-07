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

// VX_bg_unit testbench. The bench plays the DXA issue (one event per cycle:
// an operation, a commit or a wait of some warp, honouring the unit's stalls
// as the issue would) and the socket's DXA (each operation's read_done and
// write_done after their own random delays, possibly together, at most one
// completion per cycle).
//
// Every cycle it compares the unit's stalls, open sequences and unlock pulse
// with a cycle model of the rule (8 slots per warp indexed by seq, ncommit
// 0..8, head = seq - ncommit; stall_op = 63 writes pending in the open group
// or 8 committed groups; stall_commit = 8 committed groups; one head retire per
// cycle; rd_pending = a read count of the warp non-zero, wr_pending = a write
// count of any warp non-zero). Independently of that model, it checks each
// unlock against the groups themselves: every committed group but the N newest
// has all its writes (or reads) done, and the warp was waiting; that no
// satisfied wait is left waiting; and that the pending flags match the
// operations in flight. Directed runs pin the timing; random runs drain at the end.

#include "vl_simulator.h"
#include "VVX_bg_unit_top.h"
#include "VX_config.h"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#if !defined(VX_CFG_EXT_MBAR_ENABLE) || !defined(VX_CFG_EXT_DXA_ENABLE)
#error "the bulk-group unit needs a DXA build with mbarriers"
#endif

static constexpr uint32_t NW      = VX_CFG_NUM_WARPS;
static constexpr uint32_t GROUPS  = 8;
static constexpr uint32_t OPS_MAX = 63;

enum : uint32_t { K_OP = 0, K_COMMIT = 1, K_WAIT = 2, K_WAIT_READ = 3 };

static std::string strf(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  return buf;
}

template <typename T>
static uint64_t get_bits(const T& sig, uint32_t lsb, uint32_t width) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    return (static_cast<uint64_t>(sig) >> lsb) & mask;
  } else {
    uint64_t value = 0;
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i;
      uint32_t w = bit / 32;
      uint32_t o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      value |= static_cast<uint64_t>((sig[w] >> o) & m) << i;
      i += n;
    }
    return value;
  }
}

struct Evt {
  uint32_t wid, kind, n;
};

struct Compl {
  uint32_t wid, seq;
  bool rd, wr;
};

// The rule, cycle by cycle: the registered state of one warp.
struct ModelWarp {
  uint32_t seq = 0, ncommit = 0;
  uint32_t rd[GROUPS] = {}, wr[GROUPS] = {};
  bool waiting = false, ready = false, wait_rd = false;
  uint32_t wait_n = 0;

  bool stall_op() const { return wr[seq] == OPS_MAX || ncommit == GROUPS; }
  bool stall_commit() const { return ncommit == GROUPS; }
  bool rd_pending() const { return std::any_of(std::begin(rd), std::end(rd), [](uint32_t v) { return v != 0; }); }
  bool wr_pending() const { return std::any_of(std::begin(wr), std::end(wr), [](uint32_t v) { return v != 0; }); }
  uint32_t head() const { return (seq - ncommit) & (GROUPS - 1); }
  bool done(uint32_t n, bool rdk) const {
    for (uint32_t i = 0; i < GROUPS; ++i) {
      uint32_t s = (head() + i) & (GROUPS - 1);
      if (i + n < ncommit && (rdk ? rd[s] : wr[s]) != 0) {
        return false;
      }
    }
    return true;
  }
};

struct Model {
  ModelWarp w[NW];
  bool unlock_valid = false;
  uint32_t unlock_wid = 0;

  bool wr_pending() const {
    return std::any_of(std::begin(w), std::end(w), [](const ModelWarp& m) { return m.wr_pending(); });
  }

  void step(const std::optional<Evt>& e, const std::optional<Compl>& c) {
    ModelWarp n[NW];
    std::copy(std::begin(w), std::end(w), std::begin(n));
    // waits on the registered state
    for (uint32_t i = 0; i < NW; ++i) {
      if (w[i].waiting && w[i].done(w[i].wait_n, w[i].wait_rd)) {
        n[i].waiting = false;
        n[i].ready = true;
      }
    }
    if (e && (e->kind == K_WAIT || e->kind == K_WAIT_READ)) {
      bool rdk = (e->kind == K_WAIT_READ);
      if (w[e->wid].done(e->n, rdk)) {
        n[e->wid].ready = true;
      } else {
        n[e->wid].waiting = true;
      }
      n[e->wid].wait_n = e->n;
      n[e->wid].wait_rd = rdk;
    }
    int pick = -1;
    for (uint32_t i = 0; i < NW; ++i) {
      if (n[i].ready) {
        pick = static_cast<int>(i);
        break;
      }
    }
    // counts, retire
    for (uint32_t i = 0; i < NW; ++i) {
      bool retire = w[i].ncommit != 0 && w[i].rd[w[i].head()] == 0 && w[i].wr[w[i].head()] == 0;
      if (retire) {
        n[i].ncommit -= 1;
      }
    }
    if (e && e->kind == K_OP) {
      n[e->wid].rd[w[e->wid].seq] += 1;
      n[e->wid].wr[w[e->wid].seq] += 1;
    }
    if (e && e->kind == K_COMMIT) {
      n[e->wid].seq = (w[e->wid].seq + 1) & (GROUPS - 1);
      n[e->wid].ncommit += 1;
    }
    if (c) {
      n[c->wid].rd[c->seq] -= c->rd ? 1 : 0;
      n[c->wid].wr[c->seq] -= c->wr ? 1 : 0;
    }
    if (pick >= 0) {
      n[pick].ready = false;
    }
    std::copy(std::begin(n), std::end(n), std::begin(w));
    unlock_valid = (pick >= 0);
    unlock_wid = (pick >= 0) ? static_cast<uint32_t>(pick) : unlock_wid;
  }
};

// The groups as the program sees them, for the semantic checks.
struct Operation {
  uint32_t seq;
  bool rd_done = false, wr_done = false;
  bool rd_sent = false, wr_sent = false;
  uint64_t rd_due = 0, wr_due = 0;
};

struct Group {
  std::vector<Operation*> ops;
  bool committed = false;
  bool all(bool rdk) const {
    for (auto* o : ops) {
      if (!(rdk ? o->rd_done : o->wr_done)) {
        return false;
      }
    }
    return true;
  }
};

struct WarpProg {
  std::deque<Group> groups;  // committed (oldest first), then the open one at the back
  bool waiting = false;
  uint32_t wait_n = 0;
  bool wait_rd = false;
  uint64_t wait_cycle = 0;
  WarpProg() { groups.emplace_back(); }
};

struct Options {
  uint32_t seed = 1;
  uint32_t ops = 4000;
};

class Bench {
public:
  explicit Bench(const Options& o) : opts_(o), rng_(o.seed) {
    auto* t = sim_.operator->();
    this->clear_inputs();
    t->reset = 1;
    for (int i = 0; i < 4; ++i) {
      tick_ = sim_.step(tick_, 2);
    }
    t->reset = 0;
  }

  int run() {
    std::printf("bg_unit: warps=%u groups=%u seed=%u\n", NW, GROUPS, opts_.seed);
    this->directed();
    this->random_run();
    std::printf("bg_unit: %" PRIu64 " ops, %" PRIu64 " commits, %" PRIu64 " waits, %" PRIu64 " unlocks, %" PRIu64
                " op stalls, %" PRIu64 " commit stalls, %" PRIu64 " cycles\n",
                n_ops_, n_commits_, n_waits_, n_unlocks_, n_stall_op_, n_stall_commit_, cycle_);
    if (failures_ != 0) {
      std::printf("FAILED: %u check(s) failed\n", failures_);
      return 1;
    }
    std::printf("PASSED\n");
    return 0;
  }

private:
  void fail(const std::string& m) {
    if (failures_ < 20) {
      std::printf("FAIL [cycle %" PRIu64 "] %s\n", cycle_, m.c_str());
    }
    ++failures_;
  }
  void expect(bool c, const std::string& m) {
    if (!c) {
      this->fail(m);
    }
  }

  uint32_t rnd(uint32_t n) { return static_cast<uint32_t>(rng_() % n); }
  bool chance(uint32_t pct) { return this->rnd(100) < pct; }

  void clear_inputs() {
    auto* t = sim_.operator->();
    t->evt_valid = 0;
    t->evt_wid = 0;
    t->evt_kind = 0;
    t->evt_n = 0;
    t->compl_valid = 0;
    t->compl_wid = 0;
    t->compl_seq = 0;
    t->compl_read_done = 0;
    t->compl_write_done = 0;
  }

  bool dut_stall_op(uint32_t w) { return get_bits(sim_->stall_op, w, 1) != 0; }
  bool dut_stall_commit(uint32_t w) { return get_bits(sim_->stall_commit, w, 1) != 0; }
  uint32_t dut_seq(uint32_t w) { return static_cast<uint32_t>(get_bits(sim_->seq, w * 3, 3)); }
  bool dut_rd_pending(uint32_t w) { return get_bits(sim_->rd_pending, w, 1) != 0; }
  bool dut_wr_pending() { return sim_->wr_pending != 0; }

  // The pending flags the operations in flight imply.
  bool prog_rd_pending(uint32_t w) const {
    for (auto& [ow, o] : inflight_) {
      if (ow == w && !o->rd_done) return true;
    }
    return false;
  }
  bool prog_wr_pending() const {
    for (auto& [ow, o] : inflight_) {
      if (!o->wr_done) return true;
    }
    return false;
  }

  // One cycle: compares the outputs with the model, applies the event and
  // completion (the caller has checked the stalls), and advances both.
  void cycle(const std::optional<Evt>& e, const std::optional<Compl>& c) {
    auto* t = sim_.operator->();
    this->clear_inputs();
    if (e) {
      t->evt_valid = 1;
      t->evt_wid = e->wid;
      t->evt_kind = e->kind;
      t->evt_n = e->n;
    }
    if (c) {
      t->compl_valid = 1;
      t->compl_wid = c->wid;
      t->compl_seq = c->seq;
      t->compl_read_done = c->rd;
      t->compl_write_done = c->wr;
    }
    t->eval();
    for (uint32_t w = 0; w < NW; ++w) {
      const ModelWarp& m = model_.w[w];
      if (dut_stall_op(w) != m.stall_op() || dut_stall_commit(w) != m.stall_commit() || dut_seq(w) != m.seq) {
        this->fail(strf("warp %u: stall_op=%d stall_commit=%d seq=%u, model %d %d %u", w, dut_stall_op(w),
                        dut_stall_commit(w), dut_seq(w), m.stall_op(), m.stall_commit(), m.seq));
      }
      if (dut_rd_pending(w) != m.rd_pending() || dut_rd_pending(w) != this->prog_rd_pending(w)) {
        this->fail(strf("warp %u: rd_pending=%d, model %d, operations %d", w, dut_rd_pending(w), m.rd_pending(),
                        this->prog_rd_pending(w)));
      }
    }
    if (dut_wr_pending() != model_.wr_pending() || dut_wr_pending() != this->prog_wr_pending()) {
      this->fail(strf("wr_pending=%d, model %d, operations %d", dut_wr_pending(), model_.wr_pending(),
                      this->prog_wr_pending()));
    }
    bool uv = t->unlock_valid != 0;
    if (uv != model_.unlock_valid || (uv && t->unlock_wid != model_.unlock_wid)) {
      this->fail(strf("unlock valid=%d wid=%u, model %d %u", uv, t->unlock_wid, model_.unlock_valid,
                      model_.unlock_wid));
    }
    if (uv) {
      this->on_unlock(t->unlock_wid);
    }
    last_unlock_ = uv ? std::optional<uint32_t>(t->unlock_wid) : std::nullopt;
    if (e) {
      this->on_event(*e);
    }
    if (c) {
      this->on_compl(*c);
    }
    model_.step(e, c);
    tick_ = sim_.step(tick_, 2);
    ++cycle_;
    this->check_no_lost_wait();
  }

  ///////////////////////////////////////////////////////////////////////////
  // Program view

  void on_event(const Evt& e) {
    WarpProg& p = prog_[e.wid];
    switch (e.kind) {
    case K_OP: {
      ops_.push_back(std::make_unique<Operation>());
      Operation* o = ops_.back().get();
      o->seq = model_.w[e.wid].seq;
      p.groups.back().ops.push_back(o);
      o->rd_due = cycle_ + 2 + this->rnd(lat_);
      o->wr_due = std::max(o->rd_due, cycle_ + 2 + this->rnd(lat_ * 2));
      inflight_.push_back({e.wid, o});
      ++n_ops_;
      break;
    }
    case K_COMMIT:
      p.groups.back().committed = true;
      p.groups.emplace_back();
      ++n_commits_;
      break;
    default:
      if (p.waiting) {
        this->fail(strf("warp %u: a second wait", e.wid));
      }
      p.waiting = true;
      p.wait_n = e.n;
      p.wait_rd = (e.kind == K_WAIT_READ);
      p.wait_cycle = cycle_;
      ++n_waits_;
      break;
    }
  }

  void on_compl(const Compl& c) {
    for (auto it = inflight_.begin(); it != inflight_.end(); ++it) {
      if (it->first == c.wid && it->second->seq == c.seq && ((c.rd && it->second->rd_sent && !it->second->rd_done) ||
                                                              (c.wr && it->second->wr_sent && !it->second->wr_done))) {
        Operation* o = it->second;
        if (c.rd) o->rd_done = true;
        if (c.wr) o->wr_done = true;
        if (o->rd_done && o->wr_done) {
          inflight_.erase(it);
        }
        break;
      }
    }
    this->prune(c.wid);
  }

  // Drops the warp's oldest committed groups once complete (they no longer affect any wait).
  void prune(uint32_t w) {
    WarpProg& p = prog_[w];
    while (p.groups.size() > 1 && p.groups.front().committed && p.groups.front().all(false) &&
           p.groups.front().all(true)) {
      p.groups.pop_front();
    }
  }

  bool satisfied(uint32_t w) const {
    const WarpProg& p = prog_[w];
    uint32_t ncom = static_cast<uint32_t>(p.groups.size()) - 1;
    for (uint32_t i = 0; i + p.wait_n < ncom; ++i) {
      if (!p.groups[i].all(p.wait_rd)) {
        return false;
      }
    }
    return true;
  }

  void on_unlock(uint32_t w) {
    ++n_unlocks_;
    if (w >= NW || !prog_[w].waiting) {
      this->fail(strf("unlock of warp %u, which is not waiting", w));
      return;
    }
    if (!this->satisfied(w)) {
      this->fail(strf("unlock of warp %u before its groups are done (n=%u, read=%d)", w, prog_[w].wait_n,
                      prog_[w].wait_rd));
    }
    prog_[w].waiting = false;
  }

  // A wait satisfied for NW + 2 cycles must have been released.
  void check_no_lost_wait() {
    for (uint32_t w = 0; w < NW; ++w) {
      WarpProg& p = prog_[w];
      if (p.waiting && this->satisfied(w)) {
        if (++sat_age_[w] > NW + 3) {
          this->fail(strf("warp %u: a satisfied wait was not released", w));
          sat_age_[w] = 0;
        }
      } else {
        sat_age_[w] = 0;
      }
    }
  }

  // The completion to send this cycle, if any: the oldest due half.
  std::optional<Compl> next_compl(bool allow = true) {
    if (!allow || hold_compl_) {
      return std::nullopt;
    }
    for (auto& [w, o] : inflight_) {
      bool rd = !o->rd_sent && o->rd_due <= cycle_;
      bool wr = !o->wr_sent && o->wr_due <= cycle_ && (o->rd_sent || rd);
      if (rd || wr) {
        if (rd && wr && this->chance(50)) {
          wr = false;
        }
        if (rd) o->rd_sent = true;
        if (wr) o->wr_sent = true;
        return Compl{w, o->seq, rd, wr};
      }
    }
    return std::nullopt;
  }

  ///////////////////////////////////////////////////////////////////////////
  // Directed timing

  void idle(uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
      this->cycle(std::nullopt, std::nullopt);
    }
  }

  // The pending flags: an OP at T raises them from T+1, the completion that
  // clears the last count at T drops them from T+1; one warp's reads, every warp's writes.
  void directed_pending() {
    auto done = [&](Operation* o, bool rd, bool wr) {
      if (rd) o->rd_sent = true;
      if (wr) o->wr_sent = true;
      return Compl{0, o->seq, rd, wr};
    };
    sim_->eval();
    this->expect(!dut_rd_pending(0) && !dut_wr_pending(), "pending flags out of reset");
    this->cycle(Evt{0, K_OP, 0}, std::nullopt);  // T: the flags are still low in T (checked against the model)
    this->expect(dut_rd_pending(0) && dut_wr_pending(), "pending flags not raised at T+1 after an op at T");
    Operation* a = ops_.back().get();
    this->cycle(std::nullopt, done(a, true, false));
    this->expect(!dut_rd_pending(0) && dut_wr_pending(), "read_done at T: rd_pending not dropped at T+1");
    this->cycle(std::nullopt, done(a, false, true));
    this->expect(!dut_wr_pending(), "write_done at T: wr_pending not dropped at T+1");
    // An op and the completion of the last other one in the same cycle.
    this->cycle(Evt{0, K_OP, 0}, std::nullopt);
    Operation* b = ops_.back().get();
    this->cycle(Evt{0, K_OP, 0}, done(b, true, true));
    Operation* c = ops_.back().get();
    this->expect(dut_rd_pending(0) && dut_wr_pending(), "pending flags dropped by a completion beside an op");
    this->cycle(std::nullopt, done(c, true, true));
    this->expect(!dut_rd_pending(0) && !dut_wr_pending(), "pending flags kept after the last completion");
    // Another warp's write keeps wr_pending; rd_pending is per warp.
    this->cycle(Evt{0, K_OP, 0}, std::nullopt);
    Operation* d = ops_.back().get();
    const uint32_t v = NW - 1;
    this->cycle(Evt{v, K_OP, 0}, std::nullopt);
    Operation* e = ops_.back().get();
    this->cycle(std::nullopt, done(d, true, true));
    this->expect(!dut_rd_pending(0) && dut_rd_pending(v) && dut_wr_pending(), "a warp's flags follow another's");
    e->rd_sent = e->wr_sent = true;
    this->cycle(std::nullopt, Compl{v, e->seq, true, true});
    this->expect(!dut_rd_pending(v) && !dut_wr_pending(), "pending flags kept after every completion");
    this->idle(2);
  }

  void directed() {
    hold_compl_ = true;
    this->directed_pending();
    // A wait with nothing committed: released by a pulse the next cycle.
    this->cycle(Evt{0, K_WAIT, 0}, std::nullopt);
    this->expect(last_unlock_ == std::nullopt, "an unlock in the wait's own cycle");
    this->cycle(std::nullopt, std::nullopt);
    this->expect(last_unlock_ == std::optional<uint32_t>(0), "a satisfied wait is released at T+1");
    // OP then COMMIT at back-to-back cycles: the commit's seq shows at T+1.
    this->cycle(Evt{1, K_OP, 0}, std::nullopt);
    this->expect(dut_seq(1) == 0, "seq changed before the commit");
    this->cycle(Evt{1, K_COMMIT, 0}, std::nullopt);
    // (outputs read between cycles are those of the next cycle)
    this->expect(dut_seq(1) == 1, "seq is not the new group's at T+1");
    this->cycle(Evt{1, K_OP, 0}, std::nullopt);  // the next group's first operation at T+1
    this->expect(ops_.back()->seq == 1, "the operation at T+1 joined the old group");
    // wait_group 0 waits for the committed group; the open one is ignored.
    this->cycle(Evt{1, K_WAIT, 0}, std::nullopt);
    this->idle(3);
    this->expect(prog_[1].waiting, "wait_group 0 released with a write pending");
    // read_done at T: wait_group 0 still waits; write_done at T: unlock at T+2.
    hold_compl_ = false;
    Operation* g0 = prog_[1].groups.front().ops.front();
    g0->rd_sent = true;
    this->cycle(std::nullopt, Compl{1, 0, true, false});
    this->idle(3);
    this->expect(prog_[1].waiting, "wait_group released by a read completion");
    g0->wr_sent = true;
    this->cycle(std::nullopt, Compl{1, 0, false, true});  // T
    this->cycle(std::nullopt, std::nullopt);              // T+1: state satisfied
    this->expect(last_unlock_ == std::nullopt, "unlock at T+1 after a completion at T");
    this->cycle(std::nullopt, std::nullopt);  // T+2
    this->expect(last_unlock_ == std::optional<uint32_t>(1), "unlock at T+2 after a completion at T");
    hold_compl_ = true;
    // 63 operations in one group stall the 64th from the next cycle.
    for (uint32_t i = 0; i < OPS_MAX; ++i) {
      this->expect(!dut_stall_op(2), "op stall before 63 operations");
      this->cycle(Evt{2, K_OP, 0}, std::nullopt);
    }
    this->expect(dut_stall_op(2), "no op stall after 63 operations");
    // One of them writes at T: the stall drops at T+1 and a 64th operation joins
    // the group; a 65th after another write. A group has no operation limit.
    for (uint32_t k = 0; k < 2; ++k) {
      Operation* o = nullptr;
      for (auto& [w, op] : inflight_) {
        if (w == 2 && !op->wr_sent) { o = op; break; }
      }
      o->rd_sent = o->wr_sent = true;
      this->cycle(std::nullopt, Compl{2, o->seq, true, true});
      this->expect(!dut_stall_op(2), "op stall kept after a write of the open group");
      this->cycle(Evt{2, K_OP, 0}, std::nullopt);
      this->expect(dut_stall_op(2), "no op stall with 63 writes pending again");
    }
    this->expect(!dut_stall_commit(2), "commit stall with one open group");
    this->cycle(Evt{2, K_COMMIT, 0}, std::nullopt);
    this->expect(!dut_stall_op(2), "op stall after the commit");
    // 8 committed groups (7 more, empty): both stalls from the next cycle.
    for (uint32_t i = 0; i < GROUPS - 1; ++i) {
      this->expect(!dut_stall_commit(2), "commit stall before 8 groups");
      this->cycle(Evt{2, K_COMMIT, 0}, std::nullopt);
    }
    this->expect(dut_stall_commit(2) && dut_stall_op(2), "no stall with 8 committed groups");
    // wait_group 8 is released at once; wait_group.read 7 waits for the oldest group's reads.
    this->cycle(Evt{2, K_WAIT, 8}, std::nullopt);
    this->cycle(std::nullopt, std::nullopt);
    this->expect(last_unlock_ == std::optional<uint32_t>(2), "wait_group 8 not released at T+1");
    this->cycle(Evt{2, K_WAIT_READ, 7}, std::nullopt);
    this->idle(3);
    this->expect(prog_[2].waiting, "wait_group.read 7 released with the oldest group's reads pending");
    // Completing the head (63 operations): the head retires at T+2, then the empty groups one per cycle.
    hold_compl_ = false;
    while (!inflight_.empty()) {
      auto c = this->next_compl();
      this->cycle(std::nullopt, c);
    }
    this->expect(dut_stall_commit(2), "stall dropped before the head retired");
    this->cycle(std::nullopt, std::nullopt);
    this->expect(!dut_stall_commit(2) && !dut_stall_op(2), "stall kept two cycles after the head completed");
    this->idle(GROUPS + 2);
    // Two waits ready together (one satisfied by a completion at T, the other
    // arriving satisfied at T+1): the lower warp at T+2, the other at T+3.
    if (NW >= 4) {
      hold_compl_ = true;
      this->cycle(Evt{3, K_OP, 0}, std::nullopt);
      this->cycle(Evt{3, K_COMMIT, 0}, std::nullopt);
      this->cycle(Evt{3, K_WAIT, 0}, std::nullopt);
      this->idle(2);
      Operation* o = inflight_.front().second;
      o->rd_sent = o->wr_sent = true;
      this->cycle(std::nullopt, Compl{3, o->seq, true, true});  // T
      this->cycle(Evt{2, K_WAIT, 0}, std::nullopt);             // T+1
      this->expect(last_unlock_ == std::nullopt, "an unlock at T+1");
      this->cycle(std::nullopt, std::nullopt);
      this->expect(last_unlock_ == std::optional<uint32_t>(2), "the lower ready warp is released first");
      this->cycle(std::nullopt, std::nullopt);
      this->expect(last_unlock_ == std::optional<uint32_t>(3), "the higher ready warp the next cycle");
      hold_compl_ = false;
    }
    this->idle(GROUPS + 4);
    this->expect(inflight_.empty(), "directed runs left operations in flight");
    this->directed_stream();
  }

  // 300 operations in one group under slow completions: the issue keeps
  // hitting the 63-write stall and must keep going; then commit and wait.
  void directed_stream() {
    const uint32_t w = NW - 1;
    const uint32_t saved = lat_;
    lat_ = 120;
    hold_compl_ = false;
    uint32_t issued = 0, stalls = 0;
    for (uint64_t guard = 0; issued < 300; ++guard) {
      if (guard > 100000) {
        this->fail("a long open group hung at the op stall");
        break;
      }
      std::optional<Evt> e;
      sim_->eval();
      if (dut_stall_op(w)) {
        ++stalls;
      } else {
        e = Evt{w, K_OP, 0};
        ++issued;
      }
      this->cycle(e, this->next_compl());
    }
    this->expect(stalls != 0, "the stream never reached the op stall");
    this->cycle(Evt{w, K_COMMIT, 0}, this->next_compl());
    this->cycle(Evt{w, K_WAIT, 0}, this->next_compl());
    for (uint32_t i = 0; i < 20000 && (prog_[w].waiting || !inflight_.empty()); ++i) {
      this->cycle(std::nullopt, this->next_compl());
    }
    this->expect(!prog_[w].waiting && inflight_.empty(), "the long group's wait was not released");
    this->idle(GROUPS + 4);
    n_stall_op_ += stalls;
    lat_ = saved;
  }

  ///////////////////////////////////////////////////////////////////////////
  // Random runs

  void random_run() {
    lat_ = 40;
    uint64_t issued = 0;
    uint64_t guard = 0;
    while (issued < opts_.ops) {
      std::optional<Evt> e;
      uint32_t w = this->rnd(NW);
      if (!prog_[w].waiting && this->chance(85)) {
        uint32_t r = this->rnd(100);
        uint32_t kind = (r < 60) ? K_OP : (r < 85) ? K_COMMIT : (r < 93) ? K_WAIT : K_WAIT_READ;
        uint32_t n = this->chance(10) ? this->rnd(32) : this->rnd(4);
        bool stalled = (kind == K_OP && dut_stall_op_pre(w)) || (kind == K_COMMIT && dut_stall_commit_pre(w));
        if (stalled) {
          if (kind == K_OP) ++n_stall_op_; else ++n_stall_commit_;
        } else {
          e = Evt{w, kind, (kind >= K_WAIT) ? n : 0};
          ++issued;
        }
      }
      auto c = this->next_compl(this->chance(80));
      this->cycle(e, c);
      if (++guard > 200ull * opts_.ops) {
        this->fail("random run did not finish");
        return;
      }
    }
    // drain: release every wait by completing everything, then wait for the heads to retire
    for (uint32_t i = 0; i < 4000 && (!inflight_.empty() || this->any_waiting()); ++i) {
      this->cycle(std::nullopt, this->next_compl());
    }
    this->idle(GROUPS * 2 + 4);
    this->expect(inflight_.empty() && !this->any_waiting(), "random run did not drain");
    for (uint32_t w = 0; w < NW; ++w) {
      this->expect(!dut_stall_commit(w) && model_.w[w].ncommit == 0, strf("warp %u kept committed groups", w));
    }
  }

  // The stalls as the issue sees them in the current cycle (before inputs).
  bool dut_stall_op_pre(uint32_t w) { sim_->eval(); return dut_stall_op(w); }
  bool dut_stall_commit_pre(uint32_t w) { sim_->eval(); return dut_stall_commit(w); }

  bool any_waiting() const {
    for (uint32_t w = 0; w < NW; ++w) {
      if (prog_[w].waiting) return true;
    }
    return false;
  }

  Options opts_;
  std::mt19937_64 rng_;
  vl_simulator<VVX_bg_unit_top> sim_;
  uint64_t tick_ = 0;
  uint64_t cycle_ = 0;
  uint32_t failures_ = 0;
  uint32_t lat_ = 6;
  bool hold_compl_ = false;

  Model model_;
  WarpProg prog_[NW];
  std::vector<std::unique_ptr<Operation>> ops_;
  std::vector<std::pair<uint32_t, Operation*>> inflight_;
  std::optional<uint32_t> last_unlock_;
  uint32_t sat_age_[NW] = {};

  uint64_t n_ops_ = 0, n_commits_ = 0, n_waits_ = 0, n_unlocks_ = 0, n_stall_op_ = 0, n_stall_commit_ = 0;
};

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--seed=", 0) == 0) {
      opts.seed = static_cast<uint32_t>(std::strtoul(a.c_str() + 7, nullptr, 0));
    } else if (a.rfind("--ops=", 0) == 0) {
      opts.ops = static_cast<uint32_t>(std::strtoul(a.c_str() + 6, nullptr, 0));
    } else if (a.rfind("+verilator", 0) == 0) {
      continue;
    } else {
      std::printf("usage: %s [--seed=N] [--ops=N]\n", argv[0]);
      return 2;
    }
  }
  Bench b(opts);
  return b.run();
}
