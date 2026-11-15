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

#include <stdexcept>
#include <vector>

#include "ramulator/controller/rowpolicy/i_row_policy.h"
#include "vx_hbm3_controller.h"

namespace vortex::dram {

using namespace Ramulator;

// Closed page with shared activation (H100, doc 07 §4 and §8 item 2): a column access closes its
// row (RDA/WRA) unless another queued request hits the same row. ClosedCAP closes after a fixed
// count of accesses and cannot express this. When auto-precharge is not ready, the row is closed
// by an explicit PREpb as ClosedCAP does.
class VxClosedShared final : public IRowPolicy, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IRowPolicy, VxClosedShared, "VxClosedShared")

 private:
  VxHBM3* m_ctrl = nullptr;
  DRAMDevice* m_device = nullptr;
  int m_cmd_rd = -1, m_cmd_wr = -1, m_cmd_rda = -1, m_cmd_wra = -1, m_cmd_prepb = -1;
  int m_row_level = -1;
  std::vector<AddrVec_t> m_close;    // per flat bank: the row to close, empty if none
  std::vector<char> m_prepb_sent;

 public:
  void init() override {
    m_ctrl = cast_parent<VxHBM3>();
    m_device = &m_ctrl->m_device;
    const auto* spec = m_device->m_spec;
    m_cmd_rd = spec->get_command_id("RD");
    m_cmd_wr = spec->get_command_id("WR");
    m_cmd_rda = spec->get_command_id("RDA");
    m_cmd_wra = spec->get_command_id("WRA");
    m_cmd_prepb = spec->get_command_id("PREpb");
    m_row_level = spec->get_level_id("Row");
    m_close.assign(m_device->m_bank_nodes.size(), AddrVec_t{});
    m_prepb_sent.assign(m_device->m_bank_nodes.size(), 0);
  }

  void try_upgrade_command(Request& req) override {
    if (req.command != m_cmd_rd && req.command != m_cmd_wr) {
      return;
    }
    if (m_ctrl->queued_row_hit(req.addr_vec, &req)) {
      return;
    }
    const int ap = (req.command == m_cmd_rd) ? m_cmd_rda : m_cmd_wra;
    if (m_ctrl->check_timing(ap, req.addr_vec)) {
      req.command = ap;
      req.final_command = ap;
      return;
    }
    const int bank = m_device->get_flat_bank_id(req.addr_vec);
    m_close[bank] = req.addr_vec;
    m_prepb_sent[bank] = 0;
  }

  void on_issue(const Request& req) override {
    const auto& meta = m_device->m_spec->command_meta[req.command];
    if (meta.is_closing || meta.is_opening) {
      m_device->for_each_target_bank(req.command, req.addr_vec, [&](int id) {
        m_close[id].clear();
        m_prepb_sent[id] = 0;
      });
    }
  }

  void pre_schedule() override {
    for (size_t i = 0; i < m_close.size(); ++i) {
      if (m_close[i].empty() || m_prepb_sent[i]) {
        continue;
      }
      if (m_ctrl->queued_row_hit(m_close[i], nullptr)) {
        m_close[i].clear();
        continue;
      }
      AddrVec_t addr = m_close[i];
      addr[m_row_level] = -1;
      for (size_t l = m_row_level + 1; l < addr.size(); ++l) {
        addr[l] = -1;
      }
      if (!m_ctrl->check_timing(m_cmd_prepb, addr)) {
        continue;
      }
      Request pre(std::move(addr), Request::Cmd, m_cmd_prepb);
      if (m_ctrl->priority_send(pre)) {
        m_prepb_sent[i] = 1;
      }
    }
  }
};

}  // namespace vortex::dram
