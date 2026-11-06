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

#include <assert.h>
#include <util.h>

#include "sequencer.h"
#include "instr_trace.h"
#include "instr.h"
#include "core.h"

using namespace vortex;

Sequencer::Sequencer(const SimContext& ctx, const char* name, Core* core, PoolAllocator<Instr, 64>& instr_pool)
  : SimObject<Sequencer>(ctx, name)
  , core_(core)
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  , instr_pool_(instr_pool)
#endif
  , lsu_uop_gen_(instr_pool)
#ifdef VX_CFG_EXT_TCU_ENABLE
  , tcu_uop_gen_(instr_pool)
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
  , rtu_uop_gen_(instr_pool)
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
  , om_uop_gen_(instr_pool)
#endif
{}

void Sequencer::on_reset() {
  state_.reset();
}

#ifdef VX_CFG_EXT_DLMEM_ENABLE
bool Sequencer::is_release(const instr_trace_t* trace) {
  if (std::get_if<AmoType>(&trace->op_type)) {
    return std::get<IntrAmoArgs>(trace->instr_ptr->get_args()).rl;
  }
  auto wctl = std::get_if<WctlType>(&trace->op_type);
  if (wctl && *wctl == WctlType::BAR) {
    auto args = std::get<IntrWctlArgs>(trace->instr_ptr->get_args());
    return args.is_sync_bar || args.is_bar_arrive;
  }
  return false;
}

instr_trace_t* Sequencer::make_cfence(instr_trace_t* trace) {
  uint64_t uuid = trace->uuid;
  uint64_t uop_uuid = uuid & ~uint64_t(0xff000000); // uop index 0 in bits [31:24], as the RTL get_uop_uuid
  auto instr = std::allocate_shared<Instr>(instr_pool_, uop_uuid, FUType::LSU);
  instr->set_parent_uuid(uuid);
  instr->set_op_type(LsuType::CFENCE);
  instr->set_args(IntrLsuArgs{0, 0, 0});
  auto uop = core_->trace_pool().allocate(1);
  new (uop) instr_trace_t(uop_uuid);
  uop->cid       = trace->cid;
  uop->wid       = trace->wid;
  uop->cta_id    = trace->cta_id;
  uop->PC        = trace->PC;
  uop->code      = trace->code;
  uop->tmask     = trace->tmask;
  uop->instr_ptr = instr;
  uop->fu_type   = FUType::LSU;
  uop->op_type   = LsuType::CFENCE;
  return uop;
}
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
bool Sequencer::is_async_q(const instr_trace_t* trace) {
  auto type = std::get_if<LsuType>(&trace->op_type);
  return type && (*type == LsuType::ST_ASYNC || *type == LsuType::CP_ASYNC)
      && std::get<IntrLsuArgs>(trace->instr_ptr->get_args()).width == 4;
}

// The low half is a new trace, an 8-byte st.async of x[rs2] at offset 0. The
// instruction's own trace becomes the high half: x[rs2|1] at offset 8. A
// cp.async keeps rs2, its src-size, and its funct3, which tells the LSU each
// half's share of it; offset 8 applies to its source and destination.
instr_trace_t* Sequencer::split_q(instr_trace_t* trace) {
  uint64_t uuid = trace->uuid;
  // The low half gets uop index 0 in bits [31:24], as the RTL get_uop_uuid; the high
  // half keeps the instruction's uuid, where the RTL writes uop index 1.
  uint64_t uop_uuid = uuid & ~uint64_t(0xff000000);
  auto& q = *trace->instr_ptr;
  const LsuType type = std::get<LsuType>(trace->op_type);
  auto instr = std::allocate_shared<Instr>(instr_pool_, uop_uuid, FUType::LSU);
  instr->set_parent_uuid(uuid);
  instr->set_op_type(type);
  IntrLsuArgs args = std::get<IntrLsuArgs>(q.get_args());
  args.width  = 3;
  args.offset = 0;
  instr->set_args(args);
  for (uint32_t i = 0; i < NUM_SRC_REGS; ++i) {
    auto r = q.get_src_reg(i);
    instr->set_src_reg(i, r.idx, r.type);
  }
  auto uop = core_->trace_pool().allocate(1);
  new (uop) instr_trace_t(uop_uuid);
  uop->cid       = trace->cid;
  uop->wid       = trace->wid;
  uop->cta_id    = trace->cta_id;
  uop->PC        = trace->PC;
  uop->code      = trace->code;
  uop->tmask     = trace->tmask;
  uop->instr_ptr = instr;
  uop->fu_type   = FUType::LSU;
  uop->op_type   = type;
  uop->src_regs  = trace->src_regs;

  args.offset = 8;
  q.set_args(args);
  auto rs2 = q.get_src_reg(1);
  if (type == LsuType::ST_ASYNC && rs2.idx != 0) {
    rs2.idx |= 1;
    q.set_src_reg(1, rs2.idx, rs2.type);
    trace->src_regs[1] = rs2;
  }
  return uop;
}
#else
#line 79
#endif

instr_trace_t* Sequencer::get(instr_trace_t* trace) {
  // Idempotent: return cached micro-op if available
  if (state_.current_uop)
    return state_.current_uop;

#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // The CFENCE rewrites the waiting instruction, so it can issue in the cycle
  // the instruction would have; the instruction follows.
  if (state_.release == Release::None && is_release(trace)) {
    state_.release = Release::Fence;
    state_.current_uop = this->make_cfence(trace);
    return state_.current_uop;
  }
  if (state_.release == Release::Op) {
    state_.current_uop = trace;
    return trace;
  }
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
  // Like the release pair, the low half issues in the cycle the instruction
  // would have, and the high half follows.
  if (state_.qpair == QPair::None && is_async_q(trace)) {
    state_.qpair = QPair::Low;
    state_.current_uop = this->split_q(trace);
    return state_.current_uop;
  }
  if (state_.qpair == QPair::High) {
    state_.current_uop = trace;
    return trace;
  }
#else
#line 98
#endif

  // Activate sequencing for macro-op instructions
  if (!state_.active && trace->instr_ptr->is_macro_op()) {
    state_.active = true;
    state_.uop_index = 0;

    // Bind the appropriate generator based on FU type
    switch (trace->instr_ptr->get_fu_type()) {
    case FUType::LSU:
      state_.uop_count = LsuUopGen::uop_count(*trace->instr_ptr);
      state_.gen_fn = [this](const Instr& m, uint32_t i) {
        return lsu_uop_gen_.get(m, i);
      };
      break;
  #ifdef VX_CFG_EXT_TCU_ENABLE
    case FUType::TCU:
      state_.uop_count = TcuUopGen::uop_count(*trace->instr_ptr);
      state_.gen_fn = [this](const Instr& m, uint32_t i) {
        return tcu_uop_gen_.get(m, i);
      };
      // VX_uop_sequencer latches the expansion in the cycle the macro-op
      // reaches it and emits the first uop in the next one.
      state_.setup_cycle = SimPlatform::instance().cycles();
      break;
  #endif
  #if defined(VX_CFG_EXT_RTU_ENABLE) || defined(VX_CFG_EXT_OM_ENABLE)
    case FUType::SFU: {
      // Two SFU producers set is_macro_op(): the RTU ISA-v2 TRACE/WAIT ops, and
      // vx_om_export when its aperture record spans more than one beat. They are
      // told apart by op type.
      auto op_type = trace->instr_ptr->get_op_type();
      __unused(op_type);
    #ifdef VX_CFG_EXT_OM_ENABLE
      if (std::get_if<OmType>(&op_type)) {
        state_.uop_count = OmUopGen::uop_count(*trace->instr_ptr);
        state_.gen_fn = [this](const Instr& m, uint32_t i) {
          return om_uop_gen_.get(m, i);
        };
        break;
      }
    #endif
    #ifdef VX_CFG_EXT_RTU_ENABLE
      state_.uop_count = RtuUopGen::uop_count(*trace->instr_ptr);
      state_.gen_fn = [this](const Instr& m, uint32_t i) {
        return rtu_uop_gen_.get(m, i);
      };
      break;
    #else
      std::abort();
    #endif
    }
  #endif
    // Future generators:
    // case FUType::VEC: ...
    default:
      std::abort();
    }
  }

  if (state_.active) {
    if (state_.setup_cycle == SimPlatform::instance().cycles())
      return nullptr;
    // Generate micro-op Instr
    auto uop_instr = state_.gen_fn(*trace->instr_ptr, state_.uop_index);

    // Allocate micro-op trace and fill metadata (like decode does)
    auto uop_trace = core_->trace_pool().allocate(1);
    new (uop_trace) instr_trace_t(uop_instr->get_uuid());
    uop_trace->cid       = trace->cid;
    uop_trace->wid       = trace->wid;
    uop_trace->PC        = trace->PC;
    uop_trace->tmask     = uop_instr->get_tmask().any() ? uop_instr->get_tmask() : trace->tmask;
    uop_trace->instr_ptr = uop_instr;
    uop_trace->fu_type   = uop_instr->get_fu_type();
    uop_trace->op_type   = uop_instr->get_op_type();
    uop_trace->dst_reg   = uop_instr->get_dest_reg();
    uop_trace->dst_bytesel= uop_instr->get_dst_bytesel();
    for (uint32_t i = 0; i < NUM_SRC_REGS; ++i) {
      uop_trace->src_regs[i] = uop_instr->get_src_reg(i);
    }
    uop_trace->wb = (uop_instr->get_dest_reg().type != RegType::None);

    state_.current_uop = uop_trace;
    return uop_trace;
  }

  // Simple instruction — pass through
  state_.current_uop = trace;
  return trace;
}

bool Sequencer::advance() {
  state_.current_uop = nullptr;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  if (state_.release == Release::Fence) {
    state_.release = Release::Op;
    return false;
  }
  if (state_.release == Release::Op) {
    state_.release = Release::None;
    return true;
  }
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
  if (state_.qpair == QPair::Low) {
    state_.qpair = QPair::High;
    return false;
  }
  if (state_.qpair == QPair::High) {
    state_.qpair = QPair::None;
    return true;
  }
#else
#line 201
#endif
  if (state_.active) {
    ++state_.uop_index;
    if (state_.uop_index >= state_.uop_count || state_.uop_index >= 255) {
      state_.active = false;
    }
  }
  return !state_.active;
}
