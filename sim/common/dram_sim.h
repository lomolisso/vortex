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
#include <functional>

namespace vortex {

// One DRAM memory system: one partition's channels in the H100 model, every
// channel in the HBM2 model. Free of Ramulator headers: Ramulator is C++20 and
// the users of this header are C++17.
class DramSim {
public:
  struct Config {
    uint32_t num_channels;   // channels of this memory system
    uint32_t first_channel;  // chip-wide index of its channel 0
    bool     report_writes;  // return write completions (AXI B)
  };

  // channel: local to this memory system; addr: channel-local byte address.
  struct Req {
    uint32_t channel;
    uint64_t addr;
    uint32_t sector_mask;
    bool     write;
    uint64_t tag;
  };

  struct Rsp {
    uint64_t tag;
    bool     write;
  };

  struct Stats {
    uint64_t rd_bytes;
    uint64_t wr_bytes;
    uint64_t rd_reqs;
    uint64_t wr_reqs;
    uint64_t max_rd_queue;
    uint64_t max_wr_queue;
    uint64_t dir_switches;
    uint64_t ooo_rsps;
    uint64_t bp_stalls;
    uint64_t wr_rsps;
    uint64_t dram_ticks;
  };

  // fields: Ramulator's address vector in its level order (Channel,
  // PseudoChannel, Sid, BankGroup, Bank, Row, Column); tag: the request a
  // column command serves, ~0 for the other commands.
  using IssueObserver = std::function<void(uint32_t channel, const char* cmd,
                                           const uint32_t fields[7], uint64_t tag)>;

  explicit DramSim(const Config& config);
  ~DramSim();

  void reset();

  // One SM cycle.
  void tick();

  bool can_accept(uint32_t channel, bool write) const;

  // Requires can_accept(req.channel, req.write).
  void send(const Req& req);

  // Responses of one channel, in completion order.
  bool rsp_valid(uint32_t channel) const;
  const Rsp& rsp_peek(uint32_t channel) const;
  void rsp_pop(uint32_t channel);

  const Stats& stats(uint32_t channel) const;

  void set_issue_observer(IssueObserver observer);

  // Sectors of the DRAM line touched by [addr, addr + size) with a byte
  // enable set; reads pass all ones.
  static uint32_t sector_mask(uint64_t addr, uint32_t size, uint64_t byteen);

  // Memory systems of the configured model and the channels of each.
  static uint32_t num_systems();
  static uint32_t channels_per_system();

  // Channel select of a device byte address (VX_mem_bank_adapter's rule),
  // the channel-local address, and back.
  static uint32_t channel_of(uint64_t addr);
  static uint64_t local_addr(uint64_t addr);
  static uint64_t global_addr(uint32_t channel, uint64_t local);

  // One "PERF: dram ..." line with the totals of the given memory systems.
  static void print_perf(const DramSim* const* systems, uint32_t count);

private:
  class Impl;
  Impl* impl_;
};

}
