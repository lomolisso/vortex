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

// DLMEM endpoints unit test on the SimX models: per core an egress, an ingress
// and a store tracker, joined by the request and response networks, run by the
// shared bench. Multi-socket shapes also run with a domain per socket, serially
// and on worker threads, and must match the single-domain run bit for bit.

#include "dlmem/dlmem_egress.h"
#include "dlmem/dlmem_ingress.h"
#include "dlmem/dlmem_wtrack.h"
#include "dlmem/dlmem_switch.h"
#include "dlmem/dlmem_xbar.h"
#include "mem_block_pool.h"
#include "dlmem_ep_bench.h"

using namespace vortex;

namespace {

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

MemOp to_memop(uint8_t kind, uint8_t amo_op) {
  switch (kind) {
  case ep::K_LD: return MemOp::LD;
  case ep::K_ST: return MemOp::ST;
  case ep::K_LR: return MemOp::AMO_LR;
  case ep::K_SC: return MemOp::AMO_SC;
  default: break;
  }
  switch (amo_op) {
  case ep::OP_ADD:  return MemOp::AMO_ADD;
  case ep::OP_SWAP: return MemOp::AMO_SWAP;
  case ep::OP_XOR:  return MemOp::AMO_XOR;
  case ep::OP_OR:   return MemOp::AMO_OR;
  case ep::OP_AND:  return MemOp::AMO_AND;
  case ep::OP_MIN:  return MemOp::AMO_MIN;
  case ep::OP_MAX:  return MemOp::AMO_MAX;
  case ep::OP_CAS:  return MemOp::AMO_CAS;
  default: break;
  }
  std::abort();
}

uint8_t to_amo_op(MemOp op) {
  switch (op) {
  case MemOp::AMO_LR:   return ep::OP_LR;
  case MemOp::AMO_SC:   return ep::OP_SC;
  case MemOp::AMO_ADD:  return ep::OP_ADD;
  case MemOp::AMO_SWAP: return ep::OP_SWAP;
  case MemOp::AMO_XOR:  return ep::OP_XOR;
  case MemOp::AMO_OR:   return ep::OP_OR;
  case MemOp::AMO_AND:  return ep::OP_AND;
  case MemOp::AMO_MIN:  return ep::OP_MIN;
  case MemOp::AMO_MAX:  return ep::OP_MAX;
  case MemOp::AMO_CAS:  return ep::OP_CAS;
  default:              return 0;
  }
}

// Offset of a local-memory word inside its memory block.
uint32_t word_off(uint32_t waddr) {
  return (waddr * 8) & (VX_CFG_MEM_BLOCK_SIZE - 1);
}

uint64_t load_word(const std::shared_ptr<mem_block_t>& blk, uint32_t off) {
  uint64_t v = 0;
  if (blk) {
    std::memcpy(&v, blk->data() + off, 8);
  }
  return v;
}

// The SimX endpoints and NOC fabric of one cluster. Host context plays the
// RTL testbench: inputs are sent with no delay before the tick, so the
// endpoints see them in the same cycle; buffered outputs are sampled before
// the tick and popped after it when the bench was ready, so a freed slot is
// visible the next cycle, as an OUT_BUF=3 buffer's registered ready.
class SimxDut final : public ep::Dut {
public:
  using ReqSwitch = DlmemSocketSwitch<DlmemReqPkt>;
  using RspSwitch = DlmemSocketSwitch<DlmemRspPkt>;
  using ReqXbar = DlmemXbar<DlmemReqPkt>;
  using RspXbar = DlmemXbar<DlmemRspPkt>;

  SimxDut(Variant variant, bool lrsc_assert)
    : req_sent_(ep::NP, false)
    , had_rsp_(ep::NP, false)
    , had_lmem_(ep::NCC, false)
    , lanes_(ep::NP, std::vector<std::vector<uint32_t>>(ep::TAGS))
    , slot_waddr_(ep::NCC, std::vector<uint32_t>(ep::SLOTS, 0))
    , slot_store_(ep::NCC, std::vector<bool>(ep::SLOTS, false)) {
    auto& platform = SimPlatform::instance();
    platform.set_num_workers(variant == Variant::Threads ? SIMX_NUM_WORKERS : 1);
    const uint32_t S = ep::SOCKET_SIZE;
    for (uint32_t s = 0; s < ep::NUM_SOCKETS; ++s) {
      uint32_t domain = (variant == Variant::Flat) ? 0 : platform.alloc_domain();
      SimPlatform::DomainScope scope(domain);
      // Tick order inside the socket: the egresses before the request switch
      // they feed, the ingresses after it and before the response switch.
      for (uint32_t i = 0; i < S; ++i) {
        uint32_t c = s * S + i;
        DlmemEgress::Config cfg;
        cfg.core_id = c;
        cfg.ncc = ep::NCC;
        cfg.lmem_log = ep::LMEM_LOG;
        cfg.dlmem_base = ep::DLMEM_BASE;
        cfg.num_blocks = ep::NB;
        cfg.num_lanes = ep::NL;
        cfg.store_credits = ep::CREDITS;
        cfg.pending_size = ep::PENDING;
        cfg.lrsc_assert = lrsc_assert;
        egress_.push_back(DlmemEgress::Create(("dlmem-egress" + std::to_string(c)).c_str(), cfg));
      }
      req_sw_.push_back(ReqSwitch::Create(("dlmem-req-switch" + std::to_string(s)).c_str(), s, S, ep::NUM_SOCKETS));
      for (uint32_t i = 0; i < S; ++i) {
        uint32_t c = s * S + i;
        DlmemIngress::Config cfg;
        cfg.num_slots = ep::SLOTS;
        cfg.lmem_base = ep::LMEM_BASE;
        ingress_.push_back(DlmemIngress::Create(("dlmem-ingress" + std::to_string(c)).c_str(), cfg));
      }
      rsp_sw_.push_back(RspSwitch::Create(("dlmem-rsp-switch" + std::to_string(s)).c_str(), s, S, ep::NUM_SOCKETS));
      for (uint32_t i = 0; i < S; ++i) {
        uint32_t c = s * S + i;
        wtrack_.push_back(DlmemWTrack::Create(("dlmem-wtrack" + std::to_string(c)).c_str(),
                                              ep::NB, ep::NW, (1u << ep::WT_W) - 1));
      }
      for (uint32_t i = 0; i < S; ++i) {
        uint32_t c = s * S + i;
        egress_[c]->NetReqOut.bind(&req_sw_[s]->CoreIn.at(i));
        req_sw_[s]->CoreOut.at(i).bind(&ingress_[c]->NetReqIn);
        ingress_[c]->NetRspOut.bind(&rsp_sw_[s]->CoreIn.at(i));
        rsp_sw_[s]->CoreOut.at(i).bind(&egress_[c]->NetRspIn);
      }
    }
    if (ep::NUM_SOCKETS > 1) {
      req_xbar_ = ReqXbar::Create("dlmem-req-xbar", S, ep::NUM_SOCKETS);
      rsp_xbar_ = RspXbar::Create("dlmem-rsp-xbar", S, ep::NUM_SOCKETS);
      for (uint32_t s = 0; s < ep::NUM_SOCKETS; ++s) {
        dlmem_connect(req_sw_[s].get(), req_xbar_.get(), s);
        dlmem_connect(rsp_sw_[s].get(), rsp_xbar_.get(), s);
      }
    }
    platform.reset();
  }

  ~SimxDut() override {
    egress_.clear();
    ingress_.clear();
    wtrack_.clear();
    req_sw_.clear();
    rsp_sw_.clear();
    req_xbar_.reset();
    rsp_xbar_.reset();
    SimPlatform::instance().finalize();
  }

  uint32_t slot_bound() const override {
    return ep::SLOTS;
  }

  void cycle(const ep::In& in, ep::Out& out) override {
    auto& platform = SimPlatform::instance();
    for (uint32_t p = 0; p < ep::NP; ++p) {
      uint32_t c = p / ep::NB;
      uint32_t b = p % ep::NB;
      if (in.req[p] && !req_sent_[p]) {
        egress_[c]->ReqIn.at(b).send(this->make_req(p, *in.req[p]), 0);
        req_sent_[p] = true;
      }
      had_rsp_[p] = !egress_[c]->RspOut.at(b).empty();
      if (in.inc_valid[p]) {
        wtrack_[c]->IncIn.at(b).send(WtInc{in.inc_wid[p], in.inc_cnt[p]}, 0);
      }
    }
    for (uint32_t c = 0; c < ep::NCC; ++c) {
      had_lmem_[c] = !ingress_[c]->LmemReqOut.empty();
      if (in.lrsp_valid[c]) {
        uint32_t s = in.lrsp_tag[c];
        MemRsp rsp(s);
        if (!slot_store_[c].at(s)) {
          auto blk = make_mem_block();
          std::memset(blk->data(), 0, blk->size());
          std::memcpy(blk->data() + word_off(slot_waddr_[c][s]), &in.lrsp_data[c], 8);
          rsp.data = blk;
        }
        ingress_[c]->LmemRspIn.send(rsp, 0);
      }
      // The egress registers its ack; the tracker takes it the cycle after.
      out.ack_valid[c] = 0;
      if (!egress_[c]->AckOut.empty()) {
        uint32_t wid = egress_[c]->AckOut.peek();
        egress_[c]->AckOut.pop();
        out.ack_valid[c] = 1;
        out.ack_wid[c] = wid;
        wtrack_[c]->DecIn.send(wid, 0);
      }
    }

    platform.tick();

    for (uint32_t p = 0; p < ep::NP; ++p) {
      uint32_t c = p / ep::NB;
      uint32_t b = p % ep::NB;
      out.req_fire[p] = req_sent_[p] && egress_[c]->ReqIn.at(b).empty();
      if (out.req_fire[p]) {
        req_sent_[p] = false;
      }
      auto& r = out.rsp[p];
      r.fire = false;
      if (had_rsp_[p] && in.rsp_ready[p]) {
        auto rsp = egress_[c]->RspOut.at(b).peek();
        egress_[c]->RspOut.at(b).pop();
        r.fire = true;
        r.tag = uint32_t(rsp.tag);
        r.mask = 0;
        r.data.assign(ep::NL, 0);
        for (uint32_t l = 0; l < ep::NL; ++l) {
          if (rsp.mask.test(l)) {
            r.mask |= 1u << l;
            r.data[l] = load_word(rsp.data.at(l), word_off(lanes_[p].at(r.tag).at(l)));
          }
        }
      }
    }
    for (uint32_t c = 0; c < ep::NCC; ++c) {
      auto& q = out.lmem[c];
      q.fire = false;
      if (had_lmem_[c] && in.lmem_ready[c]) {
        auto req = ingress_[c]->LmemReqOut.peek();
        ingress_[c]->LmemReqOut.pop();
        uint64_t lb = req.addr - ep::LMEM_BASE;
        q.fire = true;
        q.rw = (req.op == MemOp::ST);
        q.amo = memop_is_amo_rmw(req.op);
        q.amo_op = to_amo_op(req.op);
        q.uns = req.flags.amo_unsigned;
        q.local = req.flags.local;
        q.waddr = uint32_t(lb >> 3);
        q.byteen = uint8_t(req.byteen >> word_off(q.waddr));
        q.data = load_word(req.data, word_off(q.waddr));
        q.cmp = req.amo_cmp;
        q.hart = req.hart_id;
        q.tag = req.tag;
        if (q.rw && !req.flags.strsp) {
          std::fprintf(stderr, "core %u: remote store without a commit response\n", c);
          std::abort();
        }
        slot_waddr_[c].at(q.tag) = q.waddr;
        slot_store_[c].at(q.tag) = q.rw;
      }
      out.sinks_ready[c] = ingress_[c]->LmemRspIn.empty();
      uint32_t d = 0;
      for (uint32_t w = 0; w < ep::NW; ++w) {
        d |= wtrack_[c]->drained(w) ? (1u << w) : 0;
      }
      out.drained[c] = d;
      out.eg_empty[c] = egress_[c]->empty();
      out.in_empty[c] = ingress_[c]->empty();
      out.wt_empty[c] = wtrack_[c]->empty();
      out.st_out[c] = ep::CREDITS - egress_[c]->credits();
      out.ld_out[c] = egress_[c]->pending();
      out.slot_out[c] = ingress_[c]->busy_slots();
    }
  }

private:
  LsuReq make_req(uint32_t p, const ep::ReqView& v) {
    uint32_t c = p / ep::NB;
    LsuReq r(ep::NL);
    // One instruction: every active lane carries the same op and signedness.
    const auto& first = v.lanes.at(__builtin_ctz(v.mask));
    r.op = to_memop(v.kind, first.amo_op);
    r.flags.amo_unsigned = first.uns;
    // Stores carry no LSU tag; only loads and atomics get a response to place.
    std::vector<uint32_t> lanes(ep::NL, 0);
    for (uint32_t l = 0; l < ep::NL; ++l) {
      if (!(v.mask >> l & 1)) {
        continue;
      }
      const auto& lv = v.lanes[l];
      uint64_t lb = uint64_t(lv.waddr) * 8 + __builtin_ctz(lv.byteen);
      uint32_t off = word_off(lv.waddr);
      r.mask.set(l);
      r.addrs[l] = ep::DLMEM_BASE | (uint64_t(lv.dst) << ep::LMEM_LOG) | lb;
      r.byteen[l] = uint64_t(lv.byteen) << off;
      if (v.kind != ep::K_LD) {
        auto blk = make_mem_block();
        std::memset(blk->data(), 0, blk->size());
        for (uint32_t i = 0; i < 8; ++i) {
          if (lv.byteen >> i & 1) {
            (*blk)[off + i] = uint8_t(lv.data >> (8 * i));
          }
        }
        r.data[l] = blk;
      }
      r.amo_cmp[l] = lv.cmp;
      r.tids[l] = lv.tid;
      r.dlmem_remote.set(l);
      r.dlmem_core[l] = lv.dst;
      lanes[l] = lv.waddr;
    }
    if (!v.rw) {
      lanes_[p].at(v.tag) = lanes;
    }
    r.tag = v.tag;
    r.cid = c;
    r.wid = v.wid;
    r.uuid = v.id;
    return r;
  }

  std::vector<DlmemEgress::Ptr> egress_;
  std::vector<DlmemIngress::Ptr> ingress_;
  std::vector<DlmemWTrack::Ptr> wtrack_;
  std::vector<ReqSwitch::Ptr> req_sw_;
  std::vector<RspSwitch::Ptr> rsp_sw_;
  ReqXbar::Ptr req_xbar_;
  RspXbar::Ptr rsp_xbar_;
  std::vector<bool> req_sent_;
  std::vector<bool> had_rsp_;
  std::vector<bool> had_lmem_;
  std::vector<std::vector<std::vector<uint32_t>>> lanes_;  // port, tag -> lane word addresses
  std::vector<std::vector<uint32_t>> slot_waddr_;          // core, slot -> word address
  std::vector<std::vector<bool>> slot_store_;
};

struct Result {
  uint32_t failures = 0;
  std::string failure;
  uint64_t digest = 0;
  uint64_t cycles = 0;
  size_t lane_ops = 0;
  uint64_t lmem_ops = 0;
  uint32_t max_st = 0, max_ld = 0, max_slot = 0;
  std::vector<ep::LatRow> lat;
};

Result run_case(const ep::Options& opts, Variant variant, bool lrsc_assert, std::ostream* trace) {
  Result res;
  SimxDut dut(variant, lrsc_assert);
  ep::Bench bench(dut, opts, trace);
  res.failures = bench.run();
  res.failure = bench.first_failure();
  res.digest = bench.digest();
  res.cycles = bench.cycles();
  res.lane_ops = bench.lane_ops();
  res.lmem_ops = bench.lmem_ops();
  res.max_st = bench.max_st();
  res.max_ld = bench.max_ld();
  res.max_slot = bench.max_slot();
  res.lat = bench.latency();
  return res;
}

bool same_lat(const std::vector<ep::LatRow>& a, const std::vector<ep::LatRow>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::tie(a[i].src, a[i].dst, a[i].op, a[i].seq, a[i].issue, a[i].done)
     != std::tie(b[i].src, b[i].dst, b[i].op, b[i].seq, b[i].issue, b[i].done)) {
      return false;
    }
  }
  return true;
}

void usage() {
  std::cout << "usage: dlmem_ep_test [--mode=random|hotspot|storeflood|loadflood|backpressure|lrsc|latency|all]\n"
               "                     [--seed=N] [--reqs=N] [--csv=FILE] [--trace=FILE] [--lrsc-assert]\n"
               "  --lrsc-assert: a remote LR/SC aborts, as in the core, instead of the defined fallback\n";
}

} // namespace

int main(int argc, char** argv) {
  std::vector<std::string> modes = ep::all_modes();
  std::vector<uint32_t> seeds = {1, 2};
  uint32_t reqs = 0;
  std::string csv;
  std::string trace_path;
  bool lrsc_assert = false;
  for (int a = 1; a < argc; ++a) {
    std::string arg = argv[a];
    auto val = [&](const char* k) -> const char* {
      size_t n = std::strlen(k);
      return (arg.compare(0, n, k) == 0) ? arg.c_str() + n : nullptr;
    };
    if (auto v = val("--mode=")) {
      if (std::string(v) != "all") {
        if (std::find(modes.begin(), modes.end(), v) == modes.end()) {
          usage();
          return 2;
        }
        modes = {v};
      }
    } else if (auto v = val("--seed=")) {
      seeds = {uint32_t(std::atoi(v))};
    } else if (auto v = val("--reqs=")) {
      reqs = std::atoi(v);
    } else if (auto v = val("--csv=")) {
      csv = v;
    } else if (auto v = val("--trace=")) {
      trace_path = v;
    } else if (arg == "--lrsc-assert") {
      lrsc_assert = true;
    } else {
      usage();
      return 2;
    }
  }
  if (!trace_path.empty() && (modes.size() != 1 || seeds.size() != 1)) {
    std::cerr << "--trace needs a single --mode and --seed" << std::endl;
    return 2;
  }

  std::vector<Variant> variants = {Variant::Flat};
  if (ep::NUM_SOCKETS > 1) {
    variants.push_back(Variant::Domains);
    if (SIMX_NUM_WORKERS > 1) {
      variants.push_back(Variant::Threads);
    }
  }

  std::cout << "dlmem_ep: sockets=" << ep::NUM_SOCKETS << " socket_size=" << ep::SOCKET_SIZE
            << " blocks=" << ep::NB << " lanes=" << ep::NL << " warps=" << ep::NW
            << " zacas=" << ep::ZACAS << " workers=" << SIMX_NUM_WORKERS << std::endl;

  for (auto& mode : modes) {
    for (auto seed : (mode == "latency") ? std::vector<uint32_t>{seeds.front()} : seeds) {
      ep::Options opts;
      opts.mode = mode;
      opts.seed = seed;
      opts.reqs = reqs;
      Result ref;
      std::string names;
      for (auto variant : variants) {
        std::ofstream trace_ofs;
        std::ostream* trace = nullptr;
        if (variant == Variant::Flat && !trace_path.empty()) {
          trace_ofs.open(trace_path);
          trace = &trace_ofs;
        }
        Result r = run_case(opts, variant, lrsc_assert, trace);
        if (r.failures != 0) {
          std::cout << "FAILED: mode " << mode << " seed " << seed << " (" << variant_name(variant)
                    << "): " << r.failure << " [" << r.failures << " check(s)]" << std::endl;
          return 1;
        }
        names += (names.empty() ? "" : "=") + std::string(variant_name(variant));
        if (variant == Variant::Flat) {
          ref = r;
          continue;
        }
        if (r.digest != ref.digest || r.cycles != ref.cycles || !same_lat(r.lat, ref.lat)) {
          std::cout << "FAILED: mode " << mode << " seed " << seed << ": variant '"
                    << variant_name(variant) << "' differs from 'flat'" << std::endl;
          return 1;
        }
      }
      std::cout << "mode " << mode << " seed " << seed << ": cycles=" << ref.cycles
                << " lane_ops=" << ref.lane_ops << " lmem_ops=" << ref.lmem_ops
                << " max_st=" << ref.max_st << " max_ld=" << ref.max_ld << " max_slots=" << ref.max_slot
                << " digest=" << std::hex << ref.digest << std::dec << " (" << names << ")" << std::endl;
      if (mode == "latency" && !csv.empty()) {
        ep::Bench::write_csv(csv, ref.lat);
        std::cout << "wrote " << csv << std::endl;
      }
    }
  }
  std::cout << "PASSED" << std::endl;
  return 0;
}

