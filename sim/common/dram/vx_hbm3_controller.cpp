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

#include "vx_hbm3_controller.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ramulator/controller/refresh/i_refresh_manager.h"
#include "ramulator/controller/rowpolicy/i_row_policy.h"

namespace vortex::dram {

using namespace Ramulator;

struct MsEntry {
  vx_dram_issue_hook_t hook = nullptr;
  void* ctx = nullptr;
  std::vector<VxHBM3*> ctrls;
};

namespace {
std::mutex g_registry_mutex;
// unordered_map keeps element addresses stable, so controllers hold MsEntry pointers.
std::unordered_map<IMemorySystem*, MsEntry> g_registry;
}  // namespace

void vx_dram_set_issue_hook(IMemorySystem* ms, vx_dram_issue_hook_t hook, void* ctx) {
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  auto& e = g_registry[ms];
  e.hook = hook;
  e.ctx = ctx;
}

const CtrlStats* vx_dram_ctrl_stats(IMemorySystem* ms, int channel) {
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  auto it = g_registry.find(ms);
  if (it == g_registry.end() || channel < 0 || channel >= (int)it->second.ctrls.size() || !it->second.ctrls[channel])
    return nullptr;
  return &it->second.ctrls[channel]->ctrl_stats();
}

void vx_dram_release(IMemorySystem* ms) {
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  g_registry.erase(ms);
}

void VxHBM3::init() {
  HBMControllerBase::init();
  std::string rw_policy;
  RAMULATOR_PARSE_PARAM(rw_policy, std::string, "rw_policy").default_val("AGE");
  RAMULATOR_PARSE_PARAM(m_qd[RD], int, "qd_rd").default_val(0);
  RAMULATOR_PARSE_PARAM(m_qd[WR], int, "qd_wr").default_val(0);
  RAMULATOR_PARSE_PARAM(m_t_age, Clk_t, "t_age_ticks").default_val(0);
  RAMULATOR_PARSE_PARAM(m_cmd_buses, int, "cmd_buses").default_val(1);
  RAMULATOR_PARSE_PARAM(m_pcs_per_cmd_bus, int, "pcs_per_cmd_bus").default_val(1);
  if (rw_policy == "AGE") {
    m_age_policy = true;
  } else if (rw_policy == "WATERMARK") {
    m_age_policy = false;
  } else {
    throw std::runtime_error("VxHBM3: unknown rw_policy " + rw_policy);
  }
  if (m_qd[RD] <= 0 || m_qd[WR] <= 0) {
    throw std::runtime_error("VxHBM3: qd_rd and qd_wr must be >= 1");
  }
  // A whole request's transactions must always fit behind an admitted head.
  if (m_read_buffer_size < m_qd[RD] * (int)VISIT_MAX_K || m_write_buffer_size < m_qd[WR] * (int)VISIT_MAX_K) {
    throw std::runtime_error("VxHBM3: read/write buffers must hold qd x 4 transactions");
  }
  if (m_cmd_buses < 1 || m_pcs_per_cmd_bus < 1) {
    throw std::runtime_error("VxHBM3: cmd_buses and pcs_per_cmd_bus must be >= 1");
  }
  m_row_level = m_device.m_spec->get_level_id("Row");
  m_pc_level = m_device.m_spec->get_level_id("PseudoChannel");
  m_visit_pins.assign(m_device.m_bank_nodes.size(), 0);
  for (auto& slot : m_bus_free) {
    slot.assign(m_cmd_buses, 0);
  }
}

void VxHBM3::setup(IFrontEnd* frontend, IMemorySystem* memory_system) {
  HBMControllerBase::setup(frontend, memory_system);
  m_stats.add("dir_switches", m_ctrl_stats.dir_switches);
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  auto& e = g_registry[memory_system];
  if ((int)e.ctrls.size() <= m_channel_id) {
    e.ctrls.resize(m_channel_id + 1, nullptr);
  }
  e.ctrls[m_channel_id] = this;
  m_ms_entry = &e;
}

bool VxHBM3::send(Request& req) {
  const Visit v = visit_decode(req.ingress_id);
  const Dir dir = (req.type_id == Request::Type::Write) ? WR : RD;
  auto& buffer = (dir == WR) ? m_write_buffer : m_read_buffer;
  if (v.head) {
    // A request holds its queue slot until its visit starts. The buffer check keeps every tail of an
    // admitted head admissible.
    if (m_unstarted[dir] >= m_qd[dir] || buffer.size() + v.k > buffer.max_size) {
      return false;
    }
    if (m_entries.count(v.seq)) {
      throw std::runtime_error("VxHBM3: request sequence reused while in flight");
    }
  } else if (!m_entries.count(v.seq)) {
    throw std::runtime_error("VxHBM3: a tail arrived without its head");
  }

  const size_t before = buffer.size();
  if (!ControllerBase::send(req)) {
    if (!v.head) {
      throw std::runtime_error("VxHBM3: a tail was refused behind its admitted head");
    }
    return false;
  }
  const bool buffered = buffer.size() > before;

  if (v.head) {
    m_entries.emplace(v.seq, Entry{dir, v.k, !buffered, m_clk, m_device.get_flat_bank_id(req.addr_vec),
                                   req.final_command, req.addr_vec});
    if (buffered) {
      m_waiting[dir].push_back(v.seq);
      ++m_unstarted[dir];
      auto& max_heads = (dir == WR) ? m_ctrl_stats.max_wr_heads : m_ctrl_stats.max_rd_heads;
      max_heads = std::max<uint64_t>(max_heads, m_unstarted[dir]);
    }
  }
  // Forwarded reads and coalesced writes are done at once and never reach a buffer.
  if (!buffered) {
    on_tx_done(v.seq);
  }
  return true;
}

void VxHBM3::on_tx_done(uint32_t seq) {
  auto it = m_entries.find(seq);
  if (--it->second.remaining == 0) {
    if (!it->second.started) {
      --m_unstarted[it->second.dir];
    }
    if (it->second.pinned) {
      --m_visit_pins[it->second.bank];
    }
    m_entries.erase(it);
  }
}

bool VxHBM3::queued_row_hit(const AddrVec_t& addr_vec, const Request* exclude) const {
  const int bank = m_device.get_flat_bank_id(addr_vec);
  const int row = addr_vec[m_row_level];
  auto hits = [&](const ReqBuffer& buffer) {
    for (const auto& r : buffer.buffer) {
      if (&r != exclude && r.addr_vec[m_row_level] == row && m_device.get_flat_bank_id(r.addr_vec) == bank) {
        return true;
      }
    }
    return false;
  };
  return hits(m_active_buffer) || hits(m_read_buffer) || hits(m_write_buffer);
}

std::optional<Clk_t> VxHBM3::oldest_unstarted(Dir dir) {
  auto& q = m_waiting[dir];
  while (!q.empty()) {
    auto it = m_entries.find(q.front());
    if (it != m_entries.end() && !it->second.started) {
      return it->second.arrive;
    }
    q.pop_front();
  }
  return std::nullopt;
}

void VxHBM3::update_direction() {
  if (!m_age_policy) {
    set_write_mode();
    m_dir = m_is_write_mode ? WR : RD;
    return;
  }
  const Dir cur = m_dir;
  const Dir oth = (cur == RD) ? WR : RD;
  oldest_unstarted(cur);  // drops started heads from the front, so the queue stays bounded
  auto oth_oldest = oldest_unstarted(oth);
  if (!oth_oldest) {
    return;
  }
  const bool cur_empty = (m_unstarted[cur] == 0);
  const bool aged = (m_clk - std::max(*oth_oldest, m_t_sw)) > m_t_age;
  if (cur_empty || aged) {
    m_dir = oth;
    m_t_sw = m_clk;
    ++m_ctrl_stats.dir_switches;
  }
}

bool VxHBM3::eligible(const Request& req) const {
  if (req.type_id != Request::Type::Read && req.type_id != Request::Type::Write) {
    return true;
  }
  const Visit v = visit_decode(req.ingress_id);
  auto it = m_entries.find(v.seq);
  if (it == m_entries.end() || it->second.started) {
    return true;  // in progress: its visit is already paid
  }
  if (!v.head) {
    return false;  // never ahead of its head
  }
  const Dir dir = (req.type_id == Request::Type::Write) ? WR : RD;
  return dir == m_dir;
}

bool VxHBM3::on_bus(const Request& req, int bus) const {
  if (m_cmd_buses == 1) {
    return true;
  }
  const int pc = req.addr_vec[m_pc_level];
  return (pc < 0 ? 0 : pc / m_pcs_per_cmd_bus) == bus;
}

// A visit is one row opening for all its transactions: once started, its row
// stays open until its last transaction, so a tail held behind other column commands is not
// stranded by another request's precharge.
bool VxHBM3::closes_started_visit(const Request& req) {
  if (!m_device.m_spec->command_meta[req.command].is_closing) {
    return false;
  }
  const int bank = m_device.get_flat_bank_id(req.addr_vec);
  if (m_visit_pins[bank] == 0) {
    return false;
  }
  const uint32_t own = visit_decode(req.ingress_id).seq;
  for (const auto& [seq, e] : m_entries) {
    if (e.pinned && e.bank == bank && seq != own && m_device.check_rowbuffer_hit(e.final_command, e.addr, m_clk)) {
      return true;
    }
  }
  return false;
}

std::optional<HBMControllerBase::IssuedCommand> VxHBM3::issue_slot(SlotType slot, int bus) {
  auto& bus_free = m_bus_free[static_cast<int>(slot)][bus];
  if (m_clk < bus_free) {
    return std::nullopt;
  }
  auto in_slot = [&](const Request& req) { return slot_matches(req, slot) && on_bus(req, bus) && eligible(req); };

  Candidate cand;
  if (slot == SlotType::ColumnBus) {
    cand = pick_best_ready_from(m_active_buffer, in_slot);
  }
  if (!cand.valid) {
    cand = pick_priority_if([&](const Request& req) { return slot_matches(req, slot) && on_bus(req, bus); });
  }
  if (!cand.valid) {
    // Requests already in progress may sit in either buffer; new heads only in the current direction's.
    const bool any_reserved = mark_reserved_banks();
    auto rw_filter = [&](const Request& req) {
      if (any_reserved && m_reserved_banks[m_device.get_flat_bank_id(req.addr_vec)]) {
        return false;
      }
      return !would_close_active(req) && in_slot(req) && !closes_started_visit(req);
    };
    ReqBuffer& cur = (m_dir == WR) ? m_write_buffer : m_read_buffer;
    ReqBuffer& oth = (m_dir == WR) ? m_read_buffer : m_write_buffer;
    cand = pick_best_ready_from(cur, rw_filter);
    if (!cand.valid) {
      cand = pick_best_ready_from(oth, rw_filter);
    }
  }
  if (!cand.valid) {
    return std::nullopt;
  }

  const int original_cmd = cand.it->command;
  upgrade_by_row_policy(cand);
  if (cand.it->command != original_cmd && !slot_matches(*cand.it, slot)) {
    cand.it->command = original_cmd;
  }
  if (!cand.it->is_stat_updated) {
    update_request_stats(cand.it);
  }

  m_device.issue_command(cand.it->command, cand.it->addr_vec, m_clk);
  bus_free = m_clk + m_device.m_spec->command_cycles[cand.it->command];
  IssuedCommand issued{slot, cand.it->command, cand.it->addr_vec, m_clk};

  m_rowpolicy->on_issue(*cand.it);
  for (auto* p : m_plugins) {
    p->on_issue(*cand.it);
  }
  if (m_ms_entry->hook) {
    m_ms_entry->hook(m_ms_entry->ctx, m_channel_id, m_device.m_spec->command_names[cand.it->command].c_str(),
                     cand.it->addr_vec, cand.it->ingress_id, m_clk);
  }

  const bool is_rw = (cand.it->type_id == Request::Type::Read || cand.it->type_id == Request::Type::Write);
  if (is_rw) {
    const Visit v = visit_decode(cand.it->ingress_id);
    auto it = m_entries.find(v.seq);
    if (v.head && it != m_entries.end() && !it->second.started) {
      it->second.started = true;
      it->second.pinned = true;
      ++m_visit_pins[it->second.bank];
      --m_unstarted[it->second.dir];
    }
  }

  if (cand.it->command == cand.it->final_command) {
    const int ingress_id = cand.it->ingress_id;
    retire_request(cand.it, *cand.buffer);
    if (is_rw) {
      on_tx_done(visit_decode(ingress_id).seq);
    }
  } else if (m_device.m_spec->command_meta[cand.it->command].is_opening) {
    promote_to_active(cand.it, *cand.buffer);
  }
  return issued;
}

void VxHBM3::tick() {
  hbm_tick_prologue();
  update_direction();
  // A channel wider than 64 bits has the command buses of the 64-bit HBM3 channels it spans (JESD238).
  for (int bus = 0; bus < m_cmd_buses; ++bus) {
    issue_slot(SlotType::ColumnBus, bus);
    issue_slot(SlotType::RowBus, bus);
  }
  hbm_tick_epilogue();
}

}  // namespace vortex::dram
