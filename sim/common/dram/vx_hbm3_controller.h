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

#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

#include "ramulator/controller/impl/hbm_controller_base.h"
#include "vx_dram_visit.h"

namespace vortex::dram {

// Names the Ramulator registration macros use unqualified.
using Ramulator::ConfigNode;
using Ramulator::Factory;

struct MsEntry;

// The H100's per-slice DRAM controller on Ramulator 2.1's HBMControllerBase: reads and writes
// served in turns with an age switch (the device's tWTR/tRTW time each switch), a request's sectors
// served in one row visit, and a queue depth counted in requests waiting for their visit.
class VxHBM3 final : public Ramulator::HBMControllerBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(Ramulator::IController, VxHBM3, Ramulator::HBMControllerBase, "VxHBM3")

 public:
  void init() override;
  void setup(Ramulator::IFrontEnd* frontend, Ramulator::IMemorySystem* memory_system) override;
  void tick() override;
  bool send(Ramulator::Request& req) override;

  // True if a queued request other than `exclude` targets the bank and row of addr_vec.
  bool queued_row_hit(const Ramulator::AddrVec_t& addr_vec, const Ramulator::Request* exclude) const;

  const CtrlStats& ctrl_stats() const { return m_ctrl_stats; }

 private:
  enum Dir { RD = 0, WR = 1 };

  struct Entry {
    Dir dir;
    uint32_t remaining;  // transactions not yet retired
    bool started;        // the head issued its first command
    Ramulator::Clk_t arrive;
    int bank;
    int final_command;
    Ramulator::AddrVec_t addr;
    bool pinned = false;  // started in the DRAM and counted in m_visit_pins
  };

  bool m_age_policy = true;
  int m_qd[2] = {0, 0};
  Ramulator::Clk_t m_t_age = 0;
  int m_row_level = -1;
  int m_pc_level = -1;
  int m_cmd_buses = 1;
  int m_pcs_per_cmd_bus = 1;

  std::unordered_map<uint32_t, Entry> m_entries;  // by request seq
  std::deque<uint32_t> m_waiting[2];              // seq of heads not yet started, in arrival order
  int m_unstarted[2] = {0, 0};                    // queued requests per direction
  std::vector<int> m_visit_pins;                  // per flat bank: started visits not yet done
  std::vector<Ramulator::Clk_t> m_bus_free[2];    // per slot type and command bus: first free tick

  Dir m_dir = RD;
  Ramulator::Clk_t m_t_sw = 0;

  MsEntry* m_ms_entry = nullptr;
  CtrlStats m_ctrl_stats = {};

  bool eligible(const Ramulator::Request& req) const;
  void update_direction();
  std::optional<Ramulator::Clk_t> oldest_unstarted(Dir dir);
  void on_tx_done(uint32_t seq);
  bool on_bus(const Ramulator::Request& req, int bus) const;
  bool closes_started_visit(const Ramulator::Request& req);
  std::optional<IssuedCommand> issue_slot(SlotType slot, int bus);
};

}  // namespace vortex::dram
