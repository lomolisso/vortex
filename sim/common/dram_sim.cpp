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

#include "dram_sim.h"
#include <VX_config.h>
#include <vx_dram_map.h>
#include "vx_dram_config.h"
#include "vx_dram_visit.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

using namespace vortex;

namespace {

constexpr bool is_pow2(uint64_t x) {
  return x != 0 && (x & (x - 1)) == 0;
}

constexpr uint32_t log2_exact(uint64_t x) {
  uint32_t n = 0;
  while ((uint64_t(1) << n) < x) ++n;
  return n;
}

constexpr uint32_t kChannels   = VX_CFG_DRAM_CHANNELS;
constexpr uint32_t kLgChannels = log2_exact(kChannels);
constexpr uint32_t kLgPort     = log2_exact(VX_CFG_DRAM_PORT_DATA_SIZE);
constexpr uint32_t kSelShift   = VX_CFG_DRAM_INTERLEAVE_SHIFT;
constexpr uint32_t kLgLine     = log2_exact(VX_CFG_DRAM_LINE_BYTES);
constexpr uint32_t kLgAtom     = log2_exact(VX_CFG_DRAM_ATOM_BYTES);
constexpr uint32_t kSectorsPerLine = VX_CFG_DRAM_LINE_BYTES / VX_CFG_DRAM_ATOM_BYTES;
// Without interleave the channel is the top of a DRAM_ADDR_WIDTH-bit block
// address (VX_mem_bank_adapter's split).
constexpr uint32_t kBankBlockBits = VX_CFG_DRAM_ADDR_WIDTH - kLgChannels;

static_assert(is_pow2(VX_CFG_GLOBAL_MEM_SIZE), "VX_CFG_GLOBAL_MEM_SIZE must be a power of two");
static_assert(VX_CFG_MEM_ADDR_WIDTH >= 64 || uint64_t(VX_CFG_GLOBAL_MEM_SIZE) <= (uint64_t(1) << VX_CFG_MEM_ADDR_WIDTH),
              "VX_CFG_GLOBAL_MEM_SIZE exceeds the memory address space");
static_assert(uint64_t(VX_MEM_STACK_BASE_ADDR) < uint64_t(VX_CFG_GLOBAL_MEM_SIZE), "the stack must lie in global memory");
static_assert(uint64_t(VX_MEM_STARTUP_ADDR) >= 0x80000000 && uint64_t(VX_MEM_STARTUP_ADDR) < uint64_t(VX_MEM_STACK_BASE_ADDR),
              "the program image needs a global memory of at least 4 GiB");
#ifdef VX_CFG_XLEN_64
static_assert(uint64_t(VX_CFG_GLOBAL_MEM_SIZE) <= uint64_t(VX_MEM_DLMEM_BASE_ADDR),
              "the DLMEM window must lie above global memory");
#endif
static_assert(kSectorsPerLine <= vortex::dram::VISIT_MAX_K, "a DRAM line holds at most 4 sectors");
static_assert(VX_DRAM_NUM_MS * VX_DRAM_CHANNELS_PER_MS == VX_CFG_DRAM_CHANNELS,
              "vx_dram_config.h was generated for another channel count");
static_assert(VX_DRAM_TX_BYTES == VX_CFG_DRAM_ATOM_BYTES, "one Ramulator transaction per sector");

#ifdef VX_CFG_DRAM_MODEL_H100
static_assert(VX_DRAM_MODEL_H100, "vx_dram_config.h was generated for another DRAM model");
static_assert(is_pow2(VX_CFG_DRAM_WIDTH_FACTOR), "VX_CFG_DRAM_WIDTH_FACTOR must be a power of two");
static_assert(uint64_t(VX_CFG_DRAM_CHANNEL_BITS) * 2 * 1000 / 8 * VX_CFG_NUM_SLICES * VX_CFG_H100_NUM_SMS
              >= uint64_t(VX_CFG_H100_SLICE_RD_MILLIBYTES_PER_DRAMCLK) * VX_CFG_NUM_GPCS * VX_CFG_SMS_PER_GPC * VX_CFG_H100_NUM_SLICES,
              "the DRAM channel is narrower than its per-slice read share");
static_assert(VX_CFG_DRAM_PCS <= VX_CFG_DRAM_BANKS, "more pseudo-channels than banks");
static_assert(VX_CFG_DRAM_PCS * VX_CFG_DRAM_BGS * VX_CFG_DRAM_BANKS_PER_BG == VX_CFG_DRAM_BANKS,
              "pseudo-channels x bank groups x banks per group must equal VX_CFG_DRAM_BANKS");
static_assert(is_pow2(VX_CFG_DRAM_ROWS_PER_BANK), "VX_CFG_DRAM_ROWS_PER_BANK must be a power of two");
static_assert(uint64_t(VX_CFG_DRAM_ROWS_PER_BANK) * VX_CFG_DRAM_BANKS * VX_CFG_DRAM_CHANNELS * VX_CFG_DRAM_ROW_BYTES
              == uint64_t(VX_CFG_GLOBAL_MEM_SIZE), "rows x banks x channels x row bytes must equal VX_CFG_GLOBAL_MEM_SIZE");
static_assert(VX_DRAM_PCS == VX_CFG_DRAM_PCS && VX_DRAM_BGS == VX_CFG_DRAM_BGS
              && VX_DRAM_BANKS_PER_BG == VX_CFG_DRAM_BANKS_PER_BG && VX_DRAM_ROWS == VX_CFG_DRAM_ROWS_PER_BANK,
              "vx_dram_config.h was generated for another organisation");

constexpr uint64_t kHashMasks[] = VX_DRAM_HB_H100;
constexpr vx_dram_geom_t kGeom = {log2_exact(VX_CFG_DRAM_ROW_BYTES / VX_CFG_DRAM_LINE_BYTES),
                                  log2_exact(VX_CFG_DRAM_BANKS), kHashMasks};
#else
static_assert(VX_DRAM_MODEL_HBM2, "vx_dram_config.h was generated for another DRAM model");
constexpr uint64_t kIntraMask = (uint64_t(1) << VX_DRAM_INTRA_CHANNEL_BITS) - 1;
#endif

// RD, RDA, WR and WRA carry the request they serve.
bool is_column_command(const char* cmd) {
  return cmd[0] == 'R' || cmd[0] == 'W';
}

}

class DramSim::Impl {
public:
  explicit Impl(const Config& config) : config_(config), channels_(config.num_channels) {
    if (config.num_channels != VX_DRAM_CHANNELS_PER_MS) {
      fprintf(stderr, "DramSim: %u channels, the generated configuration has %u per memory system\n",
              config.num_channels, VX_DRAM_CHANNELS_PER_MS);
      std::abort();
    }
    auto cfg = Ramulator::Config::parse_config_string(VX_DRAM_RAMULATOR_YAML);
    this->add_plugins(cfg);
    frontend_ = Ramulator::Factory::create_frontend(cfg);
    memsys_ = Ramulator::Factory::create_memory_system(cfg);
    frontend_->connect_memory_system(memsys_);
    memsys_->connect_frontend(frontend_);
    this->reset();
  }

  ~Impl() {
    std::ofstream null_stream("/dev/null");
    auto saved = std::cout.rdbuf(null_stream.rdbuf());
    frontend_->finalize();
    memsys_->finalize();
    std::cout.rdbuf(saved);
    vortex::dram::vx_dram_release(memsys_);
    delete frontend_;
    delete memsys_;
  }

  // The DRAM keeps its state across a reset (a new launch): what Ramulator
  // still holds (a launch's last writes) completes there unreported.
  void reset() {
    for (auto& e : inflight_) {
      e.second.reported = false;
    }
    for (auto& ch : channels_) {
      ch.read_pipe.clear();
      ch.write_pipe.clear();
      ch.rsps.clear();
      ch.unanswered.clear();
      ch.next_order = 0;
      ch.stats = {};
    }
    sm_cycles_ = 0;
    dram_ticks_ = 0;
    clock_acc_ = 0;
  }

  void tick() {
    ++sm_cycles_;
    clock_acc_ += VX_CFG_DRAM_CLOCK_MHZ;
    while (clock_acc_ >= VX_CFG_SM_CLOCK_MHZ) {
      clock_acc_ -= VX_CFG_SM_CLOCK_MHZ;
      this->dram_tick();
    }
    for (auto& ch : channels_) {
      while (!ch.read_pipe.empty() && ch.read_pipe.front().ready <= sm_cycles_) {
        ch.rsps.push_back(ch.read_pipe.front().rsp);
        ch.read_pipe.pop_front();
      }
    }
  }

  bool can_accept(uint32_t channel, bool write) const {
    auto& ch = channels_.at(channel);
    if (ch.skid[write].valid) {
      ++ch.stats.bp_stalls;
      return false;
    }
    return true;
  }

  void send(const Req& req) {
    auto& ch = channels_.at(req.channel);
    assert(!ch.skid[req.write].valid);
    assert(req.sector_mask != 0 && req.sector_mask < (1u << kSectorsPerLine));
    uint32_t k = __builtin_popcount(req.sector_mask);
    uint64_t bytes = uint64_t(k) << kLgAtom;
    if (req.write) {
      ++ch.stats.wr_reqs;
      ch.stats.wr_bytes += bytes;
    } else {
      ++ch.stats.rd_reqs;
      ch.stats.rd_bytes += bytes;
    }
    uint32_t seq = next_seq_;
    next_seq_ = (next_seq_ + 1) & vortex::dram::VISIT_SEQ_MASK;
    assert(inflight_.count(seq) == 0);
    bool reported = !req.write || config_.report_writes;
    uint64_t order = ch.next_order++;
    inflight_[seq] = Inflight{req.tag, req.channel, req.write, k, order, reported};
    if (reported) {
      ch.unanswered.insert(order);
    }
    auto& skid = ch.skid[req.write];
    skid.valid = true;
    skid.seq = seq;
    skid.k = k;
    skid.sent = 0;
    skid.ntx = 0;
    uint64_t line = req.addr & ~((uint64_t(1) << kLgLine) - 1);
    for (uint32_t s = 0; s < kSectorsPerLine; ++s) {
      if ((req.sector_mask >> s) & 1) {
        skid.tx[skid.ntx++] = this->tx_addr(req.channel, line, s);
      }
    }
  }

  bool rsp_valid(uint32_t channel) const {
    return !channels_.at(channel).rsps.empty();
  }

  const Rsp& rsp_peek(uint32_t channel) const {
    return channels_.at(channel).rsps.front().rsp;
  }

  void rsp_pop(uint32_t channel) {
    auto& ch = channels_.at(channel);
    auto& r = ch.rsps.front();
    if (*ch.unanswered.begin() != r.order) {
      ++ch.stats.ooo_rsps;
    }
    ch.unanswered.erase(r.order);
    ch.rsps.pop_front();
  }

  const Stats& stats(uint32_t channel) const {
    auto& ch = channels_.at(channel);
    ch.stats.dram_ticks = dram_ticks_;
    if (auto cs = vortex::dram::vx_dram_ctrl_stats(memsys_, channel)) {
      ch.stats.dir_switches = cs->dir_switches;
      ch.stats.max_rd_queue = cs->max_rd_heads;
      ch.stats.max_wr_queue = cs->max_wr_heads;
    }
    return ch.stats;
  }

  uint32_t num_channels() const {
    return config_.num_channels;
  }

  void set_issue_observer(IssueObserver observer) {
    observer_ = std::move(observer);
    vortex::dram::vx_dram_set_issue_hook(memsys_, observer_ ? &Impl::issue_hook : nullptr, this);
  }

private:
  struct Inflight {
    uint64_t tag;
    uint32_t channel;
    bool     write;
    uint32_t remaining;
    uint64_t order;
    bool     reported;
  };

  // The request at the channel's input; its sectors enter Ramulator together
  // once the controller has room.
  struct Skid {
    bool     valid = false;
    uint32_t seq = 0;
    uint32_t k = 0;
    uint32_t sent = 0;
    uint32_t ntx = 0;
    uint64_t tx[vortex::dram::VISIT_MAX_K] = {};
  };

  struct OrderedRsp {
    Rsp      rsp;
    uint64_t order;
  };

  struct Pending {
    uint64_t   ready;
    OrderedRsp rsp;
  };

  struct Channel {
    Skid skid[2];
    std::deque<Pending> read_pipe;   // ready time in SM cycles
    std::deque<Pending> write_pipe;  // ready time in DRAM ticks
    std::deque<OrderedRsp> rsps;
    std::set<uint64_t> unanswered;
    uint64_t next_order = 0;
    mutable Stats stats = {};
  };

  // Ramulator's flat address of one sector: the channel above the packed
  // intra-channel address.
  uint64_t tx_addr(uint32_t channel, uint64_t line, uint32_t sector) const {
#ifdef VX_CFG_DRAM_MODEL_H100
    auto loc = vx_dram_map(line >> kLgLine, &kGeom);
    loc.row &= VX_CFG_DRAM_ROWS_PER_BANK - 1;
    auto f = vx_dram_fields(loc, sector, log2_exact(kSectorsPerLine),
                            log2_exact(VX_CFG_DRAM_PCS), log2_exact(VX_CFG_DRAM_BGS));
    uint64_t intra = vx_dram_pack(f, log2_exact(VX_CFG_DRAM_PCS), log2_exact(VX_CFG_DRAM_BGS),
                                  log2_exact(VX_CFG_DRAM_BANKS_PER_BG), log2_exact(VX_CFG_DRAM_ROWS_PER_BANK),
                                  log2_exact(VX_DRAM_TX_PER_ROW), kLgAtom);
#else
    uint64_t intra = (line | (uint64_t(sector) << kLgAtom)) & kIntraMask;
#endif
    return (uint64_t(channel) << VX_DRAM_INTRA_CHANNEL_BITS) | intra;
  }

  void drain_skid(uint32_t channel, bool write) {
    auto& skid = channels_[channel].skid[write];
    if (!skid.valid) {
      return;
    }
    int type = write ? Ramulator::Request::Type::Write : Ramulator::Request::Type::Read;
    while (skid.sent < skid.ntx) {
      vortex::dram::Visit v{skid.seq, skid.sent == 0, write, skid.k};
      uint32_t seq = skid.seq;
      bool ok = frontend_->receive_external_requests(
        type, skid.tx[skid.sent], -1, vortex::dram::visit_encode(v),
        [this, seq](Ramulator::Request&) { this->tx_done(seq); }, VX_CFG_DRAM_ATOM_BYTES);
      if (!ok) {
        return;
      }
      ++skid.sent;
    }
    skid.valid = false;
  }

  // A read completes with its last sector's data, a write when its last
  // sector's burst ends (Ramulator calls back at WR issue).
  void tx_done(uint32_t seq) {
    auto it = inflight_.find(seq);
    assert(it != inflight_.end());
    auto& e = it->second;
    if (--e.remaining != 0) {
      return;
    }
    if (e.reported) {
      auto& ch = channels_[e.channel];
      if (e.write) {
        ch.write_pipe.push_back({dram_ticks_ + VX_DRAM_WRITE_DONE_TICKS, {{e.tag, true}, e.order}});
      } else {
        ch.read_pipe.push_back({sm_cycles_ + VX_CFG_DRAM_MISS_PREMIUM, {{e.tag, false}, e.order}});
      }
    }
    retired_.push_back(seq);
  }

  void dram_tick() {
    for (uint32_t c = 0; c < config_.num_channels; ++c) {
      this->drain_skid(c, false);
      this->drain_skid(c, true);
    }
    memsys_->tick();
    ++dram_ticks_;
    for (auto& ch : channels_) {
      while (!ch.write_pipe.empty() && ch.write_pipe.front().ready <= dram_ticks_) {
        ch.rsps.push_back(ch.write_pipe.front().rsp);
        ++ch.stats.wr_rsps;
        ch.write_pipe.pop_front();
      }
    }
    // Erased after the tick, so the issue hook of a request's last command still finds it.
    for (auto seq : retired_) {
      inflight_.erase(seq);
    }
    retired_.clear();
  }

  static void issue_hook(void* ctx, int channel, const char* command,
                         const Ramulator::AddrVec_t& addr_vec, int ingress_id, int64_t) {
    auto self = reinterpret_cast<Impl*>(ctx);
    uint32_t fields[7] = {};
    for (size_t i = 0; i < 7 && i < addr_vec.size(); ++i) {
      fields[i] = uint32_t(addr_vec[i]);
    }
    uint64_t tag = ~uint64_t(0);
    if (ingress_id >= 0 && is_column_command(command)) {
      auto it = self->inflight_.find(vortex::dram::visit_decode(ingress_id).seq);
      if (it != self->inflight_.end()) {
        tag = it->second.tag;
      }
    }
    self->observer_(self->config_.first_channel + uint32_t(channel), command, fields, tag);
  }

  // DRAM_TRACE=<path> records every command, DRAM_CMD_COUNT=<path> counts
  // them; one file per channel.
  void add_plugins(Ramulator::ConfigNode& cfg) const {
    const char* trace = getenv("DRAM_TRACE");
    const char* counts = getenv("DRAM_CMD_COUNT");
    if (!trace && !counts) {
      return;
    }
    auto ms = cfg["memory_system"];
    auto ctrls = ms["controllers"].seq();
    for (size_t i = 0; i < ctrls.size(); ++i) {
      std::string suffix = ".ch" + std::to_string(config_.first_channel + i);
      Ramulator::ConfigNode plugins = ctrls[i]["controller_plugins"];
      if (trace) {
        Ramulator::ConfigNode p;
        p.set("impl", "CmdTraceRecorder");
        p.set("path", std::string(trace[0] ? trace : "./trace/ramulator.log") + suffix);
        plugins.push_back(p);
      }
      if (counts) {
        Ramulator::ConfigNode names;
        for (auto n : {"ACT", "PREpb", "PREab", "RD", "RDA", "WR", "WRA"}) {
          names.push_back(n);
        }
        Ramulator::ConfigNode p;
        p.set("impl", "CommandCounter");
        p.set("commands_to_count", names);
        p.set("path", std::string(counts) + suffix);
        plugins.push_back(p);
      }
      ctrls[i].set("controller_plugins", plugins);
    }
    ms.set("controllers", Ramulator::ConfigNode(ctrls));
    cfg.set("memory_system", ms);
  }

  Config config_;
  Ramulator::IFrontEnd* frontend_ = nullptr;
  Ramulator::IMemorySystem* memsys_ = nullptr;
  std::vector<Channel> channels_;
  std::unordered_map<uint32_t, Inflight> inflight_;
  std::vector<uint32_t> retired_;
  uint32_t next_seq_ = 0;
  uint64_t sm_cycles_ = 0;
  uint64_t dram_ticks_ = 0;
  uint64_t clock_acc_ = 0;
  IssueObserver observer_;
};

///////////////////////////////////////////////////////////////////////////////

DramSim::DramSim(const Config& config) : impl_(new Impl(config)) {}

DramSim::~DramSim() {
  delete impl_;
}

void DramSim::reset() {
  impl_->reset();
}

void DramSim::tick() {
  impl_->tick();
}

bool DramSim::can_accept(uint32_t channel, bool write) const {
  return impl_->can_accept(channel, write);
}

void DramSim::send(const Req& req) {
  impl_->send(req);
}

bool DramSim::rsp_valid(uint32_t channel) const {
  return impl_->rsp_valid(channel);
}

const DramSim::Rsp& DramSim::rsp_peek(uint32_t channel) const {
  return impl_->rsp_peek(channel);
}

void DramSim::rsp_pop(uint32_t channel) {
  impl_->rsp_pop(channel);
}

const DramSim::Stats& DramSim::stats(uint32_t channel) const {
  return impl_->stats(channel);
}

void DramSim::set_issue_observer(IssueObserver observer) {
  impl_->set_issue_observer(std::move(observer));
}

uint32_t DramSim::sector_mask(uint64_t addr, uint32_t size, uint64_t byteen) {
  uint32_t mask = 0;
  uint64_t off = addr & ((uint64_t(1) << kLgLine) - 1);
  for (uint32_t i = 0; i < size; ++i) {
    if (i >= 64 || ((byteen >> i) & 1)) {
      mask |= 1u << ((off + i) >> kLgAtom);
    }
  }
  return mask;
}

uint32_t DramSim::num_systems() {
  return VX_DRAM_NUM_MS;
}

uint32_t DramSim::channels_per_system() {
  return VX_DRAM_CHANNELS_PER_MS;
}

uint32_t DramSim::channel_of(uint64_t addr) {
  uint64_t blk = addr >> kLgPort;
#if VX_CFG_DRAM_INTERLEAVE
  return uint32_t((blk >> kSelShift) & (kChannels - 1));
#else
  return uint32_t((blk >> kBankBlockBits) & (kChannels - 1));
#endif
}

uint64_t DramSim::local_addr(uint64_t addr) {
  uint64_t blk = addr >> kLgPort;
  uint64_t off = addr & ((uint64_t(1) << kLgPort) - 1);
#if VX_CFG_DRAM_INTERLEAVE
  uint64_t low = blk & ((uint64_t(1) << kSelShift) - 1);
  uint64_t high = blk >> (kSelShift + kLgChannels);
  uint64_t local = (high << kSelShift) | low;
#else
  uint64_t local = blk & ((uint64_t(1) << kBankBlockBits) - 1);
#endif
  return (local << kLgPort) | off;
}

uint64_t DramSim::global_addr(uint32_t channel, uint64_t local) {
  uint64_t blk = local >> kLgPort;
  uint64_t off = local & ((uint64_t(1) << kLgPort) - 1);
#if VX_CFG_DRAM_INTERLEAVE
  uint64_t low = blk & ((uint64_t(1) << kSelShift) - 1);
  uint64_t high = blk >> kSelShift;
  uint64_t global = (((high << kLgChannels) | channel) << kSelShift) | low;
#else
  uint64_t global = (uint64_t(channel) << kBankBlockBits) | blk;
#endif
  return (global << kLgPort) | off;
}

void DramSim::print_perf(const DramSim* const* systems, uint32_t count) {
  Stats t{};
  for (uint32_t i = 0; i < count; ++i) {
    for (uint32_t c = 0; c < systems[i]->impl_->num_channels(); ++c) {
      auto& s = systems[i]->stats(c);
      t.rd_bytes += s.rd_bytes;
      t.wr_bytes += s.wr_bytes;
      t.rd_reqs += s.rd_reqs;
      t.wr_reqs += s.wr_reqs;
      t.ooo_rsps += s.ooo_rsps;
      t.bp_stalls += s.bp_stalls;
      t.wr_rsps += s.wr_rsps;
      t.dir_switches += s.dir_switches;
    }
  }
  fprintf(stdout, "PERF: dram rd_reqs=%lu wr_reqs=%lu rd_bytes=%lu wr_bytes=%lu ooo_rsps=%lu bp_stalls=%lu wr_rsps=%lu dir_switches=%lu\n",
          (unsigned long)t.rd_reqs, (unsigned long)t.wr_reqs, (unsigned long)t.rd_bytes, (unsigned long)t.wr_bytes, (unsigned long)t.ooo_rsps,
          (unsigned long)t.bp_stalls, (unsigned long)t.wr_rsps, (unsigned long)t.dir_switches);
}
