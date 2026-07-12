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

#include "cluster.h"
#include "core.h"
#include "local_mem.h"
#include "constants.h"
#include "types.h"
#include "debug.h"
#ifdef VX_CFG_DSMEM_ENABLE
#include "cta_residency_table.h"
#include "ideal_transport.h"
#include <unordered_map>
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
#include "tex_core.h"
#include "tex_unit.h"
#include "sfu_unit.h"
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
#include "om_core.h"
#include "om_unit.h"
#include "sfu_unit.h"
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
#include "raster_core.h"
#include "raster_unit.h"
#include "sfu_unit.h"
#endif

using namespace vortex;

class Cluster::Impl {
public:
  Impl(Cluster* simobject)
    : simobject_(simobject)
    , cores_(VX_CFG_NUM_CORES)
    , gbarriers_(VX_CFG_NUM_BARRIERS)
#ifdef VX_CFG_DSMEM_ENABLE
    , cta_table_(VX_CFG_CTA_TABLE_SIZE)
#endif
  {
    const std::string& name = simobject_->name();
    char sname[100];

    uint32_t cores_per_cluster = cores_.size();
    uint32_t cluster_id = simobject_->id();

    // create cores

    for (uint32_t i = 0; i < cores_per_cluster; ++i) {
      uint32_t core_id = cluster_id * cores_per_cluster + i;
      snprintf(sname, 100, "%s-core%d", name.c_str(), i);
      cores_.at(i) = Core::Create(sname, core_id, simobject_);
    }

    // Create l2cache

    snprintf(sname, 100, "%s-l2cache", name.c_str());
    // L2 is the LLC iff L2 is enabled and L3 is not.
    l2cache_ = Cache::Create(sname, Cache::Config{
      !VX_CFG_L2_ENABLED,
      log2ceil(VX_CFG_L2_CACHE_SIZE),// C
      log2ceil(VX_CFG_MEM_BLOCK_SIZE),// L
      log2ceil(VX_CFG_L1_LINE_SIZE), // W
      log2ceil(VX_CFG_L2_NUM_WAYS),  // A
      log2ceil(VX_CFG_L2_NUM_BANKS), // B
      VX_CFG_XLEN,                   // address bits
      VX_CFG_L2_NUM_REQS,            // request size
      VX_CFG_L2_MEM_PORTS,           // memory ports
      VX_CFG_L2_WRITEBACK,           // write-back
      false,                  // write response
      VX_CFG_L2_MSHR_SIZE,           // mshr size
      2,                      // pipeline latency
      VX_CFG_L2_REPL_POLICY,         // replacement policy
      (VX_CFG_L2_ENABLED != 0) && (VX_CFG_L3_ENABLED == 0), // is_llc
    });


    // connect l2cache memory interface
    for (uint32_t i = 0; i < VX_CFG_L2_MEM_PORTS; ++i) {
      l2cache_->mem_req_out.at(i).bind(&simobject_->mem_req_out.at(i));
      simobject_->mem_rsp_in.at(i).bind(&l2cache_->mem_rsp_in.at(i));
    }

    // ── L2 fan-in: cores + optional extension caches ───────────────────
    // Row 0 = cores (high priority). DXA is no longer here — it is core-
    // resident (per-SM) and merges into each core's L1→L2 arbiter upstream.
    // Row 1 = tcache (if enabled).
    // Row 2 = ocache (if enabled).
    // Row 3 = rcache (if enabled).
    // The priority arbiter lets cores win over extension traffic on contention.
#if defined(VX_CFG_EXT_TEX_ENABLE) || defined(VX_CFG_EXT_OM_ENABLE) || defined(VX_CFG_EXT_RASTER_ENABLE)
    constexpr uint32_t kL2Rows = 1
        + VX_CFG_EXT_TEX_ENABLED + VX_CFG_EXT_OM_ENABLED + VX_CFG_EXT_RASTER_ENABLED;
    snprintf(sname, 100, "%s-l2arb", name.c_str());
    auto l2arb = MemArbiter::Create(sname, ArbiterType::Priority,
                                    kL2Rows * VX_CFG_L2_NUM_REQS, VX_CFG_L2_NUM_REQS);
    // cores → row 0
    for (uint32_t i = 0; i < cores_per_cluster; ++i) {
      for (uint32_t j = 0; j < VX_CFG_L1_MEM_PORTS; ++j) {
        uint32_t port = i * VX_CFG_L1_MEM_PORTS + j;
        cores_.at(i)->mem_req_out.at(j).bind(&l2arb->ReqIn.at(kL2Rows * port + 0));
        l2arb->RspOut.at(kL2Rows * port + 0).bind(&cores_.at(i)->mem_rsp_in.at(j));
      }
    }
#else
    // No L2-sharing extensions: direct cores → L2.
    for (uint32_t i = 0; i < cores_per_cluster; ++i) {
      for (uint32_t j = 0; j < VX_CFG_L1_MEM_PORTS; ++j) {
        cores_.at(i)->mem_req_out.at(j).bind(&l2cache_->core_req_in.at(i * VX_CFG_L1_MEM_PORTS + j));
        l2cache_->core_rsp_out.at(i * VX_CFG_L1_MEM_PORTS + j).bind(&cores_.at(i)->mem_rsp_in.at(j));
      }
    }
#endif // any L2-sharing extension

#ifdef VX_CFG_DSMEM_ENABLE
    // ── SM-to-SM transport: cross-core LMEM connectivity (DSMEM mode) ──────
    // Purely additive to the global path: shared-memory traffic between cores
    // rides this transport; global traffic keeps the L2/bypass path above.
    // IdealTransport is a fixed-delay placeholder — NoC modeling is delegated
    // to the Omelet co-simulator (roadmap Task 4).
    snprintf(sname, 100, "%s-dsmem-transport", name.c_str());
    dsmem_transport_ = IdealTransport::Create(sname, cores_per_cluster);
    for (uint32_t i = 0; i < cores_per_cluster; ++i) {
      cores_.at(i)->dsmem_req_out.at(0).bind(&dsmem_transport_->ReqIn.at(i));
      dsmem_transport_->RspOut.at(i).bind(&cores_.at(i)->dsmem_rsp_in.at(0));
      dsmem_transport_->ReqOut.at(i).bind(&cores_.at(i)->dsmem_remote_req_in.at(0));
      cores_.at(i)->dsmem_remote_rsp_out.at(0).bind(&dsmem_transport_->RspIn.at(i));
    }
#endif

#ifdef VX_CFG_EXT_TEX_ENABLE
    // ── Cluster-shared TEX engine + tcache ──────────────────────────────
    snprintf(sname, 100, "%s-tex-core", name.c_str());
    tex_core_ = TexCore::Create(sname, simobject_);

    // tcache: read-only TLM Cache, config from VX_config.toml [tcache] section.
    snprintf(sname, 100, "%s-tcache", name.c_str());
    constexpr uint32_t kTcacheLineSize = VX_CFG_MEM_BLOCK_SIZE; // = TCACHE_LINE_SIZE = VX_CFG_L1_LINE_SIZE
    constexpr uint32_t kTcacheWordSize = 4;              // = TCACHE_WORD_SIZE
    constexpr uint32_t kTcacheNumReqs  = VX_CFG_TCACHE_NUM_BANKS;
    constexpr uint32_t kTcacheMemPorts = 1;              // = TCACHE_MEM_PORTS
    auto tcache = Cache::Create(sname, Cache::Config{
      false,                       // bypass
      log2ceil(VX_CFG_TCACHE_SIZE),       // C
      log2ceil(kTcacheLineSize),   // L
      log2ceil(kTcacheWordSize),   // W
      log2ceil(VX_CFG_TCACHE_NUM_WAYS),   // A
      log2ceil(VX_CFG_TCACHE_NUM_BANKS),  // B
      VX_CFG_XLEN,                        // address bits
      kTcacheNumReqs,              // request size
      kTcacheMemPorts,             // memory ports
      false,                       // write-back (read-only cache)
      false,                       // write response
      VX_CFG_TCACHE_MSHR_SIZE,            // mshr size
      2,                           // pipeline latency
      uint8_t(VX_CFG_L2_REPL_POLICY),     // replacement policy (use L2 policy as default)
      false,                       // is_llc (TCACHE is auxiliary, not LLC)
    });
    tcache_ = tcache;

    // tex_core ↔ tcache (per-port).
    for (uint32_t i = 0; i < kTcacheNumReqs; ++i) {
      tex_core_->tcache_req_out.at(i).bind(&tcache->core_req_in.at(i));
      tcache->core_rsp_out.at(i).bind(&tex_core_->tcache_rsp_in.at(i));
    }
    // tcache memory side → l2arb. Row index = kL2Rows-1 if no OM, else
    // kL2Rows-2 (OM occupies the last row when both are present).
    constexpr uint32_t kTexRow = 1;
    for (uint32_t i = 0; i < kTcacheMemPorts; ++i) {
      tcache->mem_req_out.at(i).bind(&l2arb->ReqIn.at(kL2Rows * i + kTexRow));
      l2arb->RspOut.at(kL2Rows * i + kTexRow).bind(&tcache->mem_rsp_in.at(i));
    }

    // Cluster-level TexBus arbiter: NUM_CORES_PER_CLUSTER inputs (one per
    // SfuUnit) → 1 TEX-core lane (kNumTexCores=1 by default).
    snprintf(sname, 100, "%s-tex-bus", name.c_str());
    auto tex_bus = TexBusArbiter::Create(sname, ArbiterType::RoundRobin,
                                         cores_per_cluster, 1);
    tex_bus_arb_ = tex_bus;
    for (uint32_t cid = 0; cid < cores_per_cluster; ++cid) {
      auto sfu = cores_.at(cid)->sfu_unit();
      sfu->tex_req_out.bind(&tex_bus->ReqIn.at(cid));
      tex_bus->RspOut.at(cid).bind(&sfu->tex_rsp_in);
    }
    tex_bus->ReqOut.at(0).bind(&tex_core_->tex_req_in.at(0));
    tex_core_->tex_rsp_out.at(0).bind(&tex_bus->RspIn.at(0));
#endif

#if defined(VX_CFG_EXT_TEX_ENABLE) || defined(VX_CFG_EXT_OM_ENABLE) || defined(VX_CFG_EXT_RASTER_ENABLE)
    // L2 arb outputs → l2cache (after all rows are bound).
    for (uint32_t i = 0; i < VX_CFG_L2_NUM_REQS; ++i) {
      l2arb->ReqOut.at(i).bind(&l2cache_->core_req_in.at(i));
      l2cache_->core_rsp_out.at(i).bind(&l2arb->RspIn.at(i));
    }
#endif

#ifdef VX_CFG_EXT_OM_ENABLE
    // ── Cluster-shared OM engine + ocache ───────────────────────────────
    snprintf(sname, 100, "%s-om-core", name.c_str());
    om_core_ = OmCore::Create(sname, simobject_);

    // ocache: write-through TLM Cache, config from VX_config.toml [ocache] section.
    snprintf(sname, 100, "%s-ocache", name.c_str());
    constexpr uint32_t kOcacheLineSize = VX_CFG_MEM_BLOCK_SIZE;
    constexpr uint32_t kOcacheWordSize = 4;
    constexpr uint32_t kOcacheNumReqs  = VX_CFG_OCACHE_NUM_BANKS;
    constexpr uint32_t kOcacheMemPorts = 1;
    auto ocache = Cache::Create(sname, Cache::Config{
      false,                        // bypass
      log2ceil(VX_CFG_OCACHE_SIZE),        // C
      log2ceil(kOcacheLineSize),    // L
      log2ceil(kOcacheWordSize),    // W
      log2ceil(VX_CFG_OCACHE_NUM_WAYS),    // A
      log2ceil(VX_CFG_OCACHE_NUM_BANKS),   // B
      VX_CFG_XLEN,                         // address bits
      kOcacheNumReqs,               // request size
      kOcacheMemPorts,              // memory ports
      false,                        // write-back (write-through)
      false,                        // write response
      VX_CFG_OCACHE_MSHR_SIZE,             // mshr size
      2,                            // pipeline latency
      uint8_t(VX_CFG_L2_REPL_POLICY),      // replacement policy
      false,                        // is_llc (OCACHE is auxiliary, not LLC)
    });
    ocache_ = ocache;

    // om_core ↔ ocache (per-port).
    for (uint32_t i = 0; i < kOcacheNumReqs; ++i) {
      om_core_->ocache_req_out.at(i).bind(&ocache->core_req_in.at(i));
      ocache->core_rsp_out.at(i).bind(&om_core_->ocache_rsp_in.at(i));
    }

    // ocache memory side → l2arb. Row index is cores + TEX (if present).
    constexpr uint32_t kOmRow = 1 + VX_CFG_EXT_TEX_ENABLED;
    for (uint32_t i = 0; i < kOcacheMemPorts; ++i) {
      ocache->mem_req_out.at(i).bind(&l2arb->ReqIn.at(kL2Rows * i + kOmRow));
      l2arb->RspOut.at(kL2Rows * i + kOmRow).bind(&ocache->mem_rsp_in.at(i));
    }

    // Per-core SFU.om_req_out (OmUnit decodes onto it) → OmCore::om_req_in[cid].
    // OM has no return value — no rsp channel back to SfuUnit.
    for (uint32_t cid = 0; cid < cores_per_cluster; ++cid) {
      auto sfu = cores_.at(cid)->sfu_unit();
      sfu->om_req_out.bind(&om_core_->om_req_in.at(cid));
    }
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
    // ── Cluster-shared RASTER engine + rcache ───────────────────────────
    snprintf(sname, 100, "%s-raster-core", name.c_str());
    raster_core_ = RasterCore::Create(sname, simobject_);

    // rcache: read-only TLM Cache, config from VX_config.toml [rcache] section.
    snprintf(sname, 100, "%s-rcache", name.c_str());
    constexpr uint32_t kRcacheLineSize = VX_CFG_MEM_BLOCK_SIZE;
    constexpr uint32_t kRcacheWordSize = 4;
    constexpr uint32_t kRcacheNumReqs  = VX_CFG_RCACHE_NUM_BANKS;
    constexpr uint32_t kRcacheMemPorts = 1;
    auto rcache = Cache::Create(sname, Cache::Config{
      false,                        // bypass
      log2ceil(VX_CFG_RCACHE_SIZE),        // C
      log2ceil(kRcacheLineSize),    // L
      log2ceil(kRcacheWordSize),    // W
      log2ceil(VX_CFG_RCACHE_NUM_WAYS),    // A
      log2ceil(VX_CFG_RCACHE_NUM_BANKS),   // B
      VX_CFG_XLEN,                         // address bits
      kRcacheNumReqs,               // request size
      kRcacheMemPorts,              // memory ports
      false,                        // write-back (read-only)
      false,                        // write response
      VX_CFG_RCACHE_MSHR_SIZE,             // mshr size
      2,                            // pipeline latency
      uint8_t(VX_CFG_L2_REPL_POLICY),      // replacement policy
      false,                        // is_llc (RCACHE is auxiliary, not LLC)
    });
    rcache_ = rcache;

    // raster_core ↔ rcache (per-port).
    for (uint32_t i = 0; i < kRcacheNumReqs; ++i) {
      raster_core_->rcache_req_out.at(i).bind(&rcache->core_req_in.at(i));
      rcache->core_rsp_out.at(i).bind(&raster_core_->rcache_rsp_in.at(i));
    }

    // rcache memory side → l2arb (RASTER is last row when present).
    constexpr uint32_t kRasterRow = kL2Rows - 1;
    for (uint32_t i = 0; i < kRcacheMemPorts; ++i) {
      rcache->mem_req_out.at(i).bind(&l2arb->ReqIn.at(kL2Rows * i + kRasterRow));
      l2arb->RspOut.at(kL2Rows * i + kRasterRow).bind(&rcache->mem_rsp_in.at(i));
    }

    // Cluster-level RasterBus arbiter: NUM_CORES_PER_CLUSTER inputs (one per
    // SfuUnit) → 1 lane (kNumRasterLanes=1).
    snprintf(sname, 100, "%s-raster-bus", name.c_str());
    auto raster_bus = RasterBusArbiter::Create(sname, ArbiterType::RoundRobin,
                                               cores_per_cluster, 1);
    raster_bus_arb_ = raster_bus;
    for (uint32_t cid = 0; cid < cores_per_cluster; ++cid) {
      auto sfu = cores_.at(cid)->sfu_unit();
      sfu->raster_req_out.bind(&raster_bus->ReqIn.at(cid));
      raster_bus->RspOut.at(cid).bind(&sfu->raster_rsp_in);
    }
    raster_bus->ReqOut.at(0).bind(&raster_core_->raster_req_in.at(0));
    raster_core_->raster_rsp_out.at(0).bind(&raster_bus->RspIn.at(0));
#endif
  }

  void reset() {
    for (auto& gbar : gbarriers_) {
      gbar.reset();
    }
#ifdef VX_CFG_DSMEM_ENABLE
    cta_table_.reset();
    cbar_arrivals_.clear();
#endif
    // Cores are SimObjects; reset by SimPlatform.
  }

  bool running() const {
    for (auto& core : cores_) {
      if (core->running())
        return true;
    }
    return false;
  }

  int get_exitcode() const {
    int exitcode = 0;
    for (auto& core : cores_) {
      exitcode |= core->get_exitcode();
    }
    return exitcode;
  }

  void global_barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id) {
    auto bar_index = bar_id % gbarriers_.size();
    auto& gbar = gbarriers_.at(bar_index);

    uint32_t cores_per_cluster = cores_.size();
    uint32_t local_core_id = core_id % cores_per_cluster;

    // set core arrival bit
    gbar.mask.set(local_core_id);

    DT(4, "*** Global barrier arrive: cluster #" << simobject_->id() << ", core #" << core_id << " at barrier #" << bar_id << ", arrived=" << gbar.mask.count());

    if (gbar.mask.count() == (size_t)count) {
      // resume all suspended cores
      for (uint32_t i = 0; i < cores_per_cluster; ++i) {
        if (gbar.mask.test(i)) {
          cores_.at(i)->global_barrier_resume(bar_id);
        }
      }
      // reset mask and advance phase
      gbar.mask.reset();
    }
  }

#ifdef VX_CFG_DSMEM_ENABLE
  Core* core_by_id(uint32_t gid) const {
    for (auto& c : cores_) {
      if (c->id() == gid)
        return c.get();
    }
    return nullptr;
  }

  // Master admit + replica broadcast. The master records the rank; once the
  // cluster is fully placed, its complete rank->{core,base} set is broadcast to
  // each participating SM's read replica (so hot-path resolves are local). The
  // broadcast is a control-plane event (once per cluster launch), off the DSMEM
  // access critical path — the residency read latency modeled on the hot path is
  // VX_CFG_CTA_TABLE_LOOKUP_LATENCY (charged at the DSMEM egress).
  void cta_admit(uint32_t cluster_id, uint32_t rank, uint32_t num_ranks,
                 uint32_t core_id, uint32_t cta_slot, uint64_t lmem_base) {
    cta_table_.admit(cluster_id, rank, num_ranks, core_id, cta_slot, lmem_base);
    if (!cta_table_.fully_admitted(cluster_id))
      return;
    // Placement complete: push every rank entry into every participant replica.
    for (uint32_t p = 0; p < num_ranks; ++p) {
      auto* pslot = cta_table_.resolve(cluster_id, p);
      assert(pslot != nullptr);
      Core* pcore = core_by_id(pslot->core_id);
      assert(pcore != nullptr && "cluster rank on a core outside this hw cluster");
      for (uint32_t r = 0; r < num_ranks; ++r) {
        auto* rslot = cta_table_.resolve(cluster_id, r);
        assert(rslot != nullptr);
        pcore->cta_replica()->install(cluster_id, r, num_ranks, *rslot);
      }
    }
  }

  void cta_retire(uint32_t cluster_id, uint32_t rank) {
    bool freed = cta_table_.retire(cluster_id, rank);
    if (freed) {
      // Cluster torn down: drop it from every SM's replica (no-op where absent).
      for (auto& c : cores_) {
        c->cta_replica()->evict(cluster_id);
      }
    }
  }

  void cluster_barrier_arrive(uint32_t cluster_id, uint32_t bar_id) {
    uint32_t expected = cta_table_.num_ranks(cluster_id);
    assert(expected != 0 && "cluster barrier arrival for a non-resident cluster");
    uint64_t key = (uint64_t(cluster_id) << 32) | (bar_id & 0x3fffffff);
    uint32_t arrived = ++cbar_arrivals_[key];
    DT(4, "*** Cluster-barrier arrive: cluster #" << simobject_->id()
       << ", cta-cluster #" << cluster_id << " at barrier #" << bar_id
       << ", arrived=" << arrived << "/" << expected);
    if (arrived == expected) {
      cbar_arrivals_.erase(key);
      uint64_t participants = cta_table_.participant_mask(cluster_id);
      uint32_t cores_per_cluster = cores_.size();
      for (uint32_t i = 0; i < cores_per_cluster; ++i) {
        if (participants & (1ull << cores_.at(i)->id())) {
          cores_.at(i)->cluster_barrier_resume(bar_id);
        }
      }
    }
  }
#endif

  Cluster::PerfStats perf_stats() const {
    Cluster::PerfStats perf_stats;
    perf_stats.l2cache = l2cache_->perf_stats();
#ifdef VX_CFG_EXT_DXA_ENABLE
    // DXA is now per-core (per-SM); aggregate each core's engine.
    for (auto& core : cores_)
      perf_stats.dxa += core->dxa_perf_stats();
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
    perf_stats.tex    = tex_core_->perf_stats();
    perf_stats.tcache = tcache_->perf_stats();
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
    perf_stats.raster = raster_core_->perf_stats();
    perf_stats.rcache = rcache_->perf_stats();
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
    perf_stats.om     = om_core_->perf_stats();
    perf_stats.ocache = ocache_->perf_stats();
#endif
    return perf_stats;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    // DXA descriptor DCRs are handled per-core (broadcast to every SM below).
#ifdef VX_CFG_EXT_TEX_ENABLE
    if (addr >= VX_DCR_TEX_STATE_BEGIN && addr < VX_DCR_TEX_STATE_END) {
      return tex_core_->dcr_write(addr, value);
    }
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
    if (addr >= VX_DCR_OM_STATE_BEGIN && addr < VX_DCR_OM_STATE_END) {
      return om_core_->dcr_write(addr, value);
    }
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
    if (addr >= VX_DCR_RASTER_STATE_BEGIN && addr < VX_DCR_RASTER_STATE_END) {
      return raster_core_->dcr_write(addr, value);
    }
#endif
    for (auto& core : cores_) {
      int ret = core->dcr_write(addr, value);
      if (ret != 0)
        return ret;
    }
    return 0;
  }

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
    for (auto& core : cores_) {
      uint16_t core_id = tag & 0xfffff;
      if (core_id != core->id())
        continue; // skip cores that don't match the tag
      uint32_t tag_value = tag >> 16;
      int ret = core->dcr_read(addr, tag_value, value);
      if (ret != 0)
        return ret;
    }
    return 0;
  }

  void dcache_flush_begin() {
    for (auto& core : cores_) {
      core->dcache_flush_begin();
    }
  }

  bool dcache_flush_done() const {
    for (auto& core : cores_) {
      if (!core->dcache_flush_done()) return false;
    }
    return true;
  }

  void icache_flush_begin() {
    for (auto& core : cores_) {
      core->icache_flush_begin();
    }
  }

  bool icache_flush_done() const {
    for (auto& core : cores_) {
      if (!core->icache_flush_done()) return false;
    }
    return true;
  }

#ifdef VX_CFG_EXT_TEX_ENABLE
  void tcache_flush_begin() { tcache_->flush_begin(); }
  bool tcache_flush_done() const { return tcache_->flush_done(); }
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
  void rcache_flush_begin() { rcache_->flush_begin(); }
  bool rcache_flush_done() const { return rcache_->flush_done(); }
#endif

#ifdef VX_CFG_EXT_OM_ENABLE
  void ocache_flush_begin() { ocache_->flush_begin(); }
  bool ocache_flush_done() const { return ocache_->flush_done(); }
#endif

  void l2_flush_begin() {
    l2cache_->flush_begin();
  }

  bool l2_flush_done() const {
    return l2cache_->flush_done();
  }

  Core* get_core(uint32_t idx) const {
    if (idx >= cores_.size()) return nullptr;
    return cores_.at(idx).get();
  }

#ifdef VX_CFG_EXT_RASTER_ENABLE
  RasterCore::Ptr& raster_core() { return raster_core_; }
#endif

#ifdef VX_CFG_DSMEM_ENABLE
  CtaResidencyTable* cta_table() { return &cta_table_; }
#endif

private:
  Cluster*                    simobject_;
  std::vector<Core::Ptr>      cores_;
  std::vector<core_barrier_t> gbarriers_;
#ifdef VX_CFG_DSMEM_ENABLE
  CtaResidencyTable           cta_table_;
  IdealTransport::Ptr         dsmem_transport_;
  // (cluster_id, bar_index) → rank arrivals for in-flight cluster barriers.
  std::unordered_map<uint64_t, uint32_t> cbar_arrivals_;
#endif
  Cache::Ptr                  l2cache_;
#ifdef VX_CFG_EXT_TEX_ENABLE
  TexCore::Ptr                tex_core_;
  Cache::Ptr                  tcache_;
  TexBusArbiter::Ptr          tex_bus_arb_;
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
  OmCore::Ptr                 om_core_;
  Cache::Ptr                  ocache_;
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
  RasterCore::Ptr             raster_core_;
  Cache::Ptr                  rcache_;
  RasterBusArbiter::Ptr       raster_bus_arb_;
#endif
};

///////////////////////////////////////////////////////////////////////////////

Cluster::Cluster(const SimContext& ctx,
                 const char* name,
                 uint32_t cluster_id,
                 ProcessorImpl* processor)
  : SimObject(ctx, name)
  , mem_req_out(VX_CFG_L2_MEM_PORTS, this)
  , mem_rsp_in(VX_CFG_L2_MEM_PORTS, this)
  , cluster_id_(cluster_id)
  , processor_(processor)
  , impl_(new Impl(this))
{}

Cluster::~Cluster() {
  delete impl_;
}

void Cluster::on_reset() {
  impl_->reset();
}

void Cluster::on_tick() {
  //--
}

bool Cluster::running() const {
  return impl_->running();
}

int Cluster::get_exitcode() const {
  return impl_->get_exitcode();
}

void Cluster::global_barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  impl_->global_barrier_arrive(bar_id, count, core_id);
}

Cluster::PerfStats Cluster::perf_stats() const {
  return impl_->perf_stats();
}

int Cluster::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

int Cluster::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  return impl_->dcr_read(addr, tag, value);
}

Core* Cluster::get_core(uint32_t idx) const {
  return impl_->get_core(idx);
}

#ifdef VX_CFG_DSMEM_ENABLE
CtaResidencyTable* Cluster::cta_table() const {
  return impl_->cta_table();
}

void Cluster::cta_admit(uint32_t cluster_id, uint32_t rank, uint32_t num_ranks,
                        uint32_t core_id, uint32_t cta_slot, uint64_t lmem_base) {
  impl_->cta_admit(cluster_id, rank, num_ranks, core_id, cta_slot, lmem_base);
}

void Cluster::cta_retire(uint32_t cluster_id, uint32_t rank) {
  impl_->cta_retire(cluster_id, rank);
}

void Cluster::cluster_barrier_arrive(uint32_t cluster_id, uint32_t bar_id) {
  impl_->cluster_barrier_arrive(cluster_id, bar_id);
}
#endif

void Cluster::dcache_flush_begin() {
  impl_->dcache_flush_begin();
}

bool Cluster::dcache_flush_done() const {
  return impl_->dcache_flush_done();
}

void Cluster::icache_flush_begin() {
  impl_->icache_flush_begin();
}

bool Cluster::icache_flush_done() const {
  return impl_->icache_flush_done();
}

#ifdef VX_CFG_EXT_TEX_ENABLE
void Cluster::tcache_flush_begin() { impl_->tcache_flush_begin(); }
bool Cluster::tcache_flush_done() const { return impl_->tcache_flush_done(); }
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
void Cluster::rcache_flush_begin() { impl_->rcache_flush_begin(); }
bool Cluster::rcache_flush_done() const { return impl_->rcache_flush_done(); }
#endif

#ifdef VX_CFG_EXT_OM_ENABLE
void Cluster::ocache_flush_begin() { impl_->ocache_flush_begin(); }
bool Cluster::ocache_flush_done() const { return impl_->ocache_flush_done(); }
#endif

void Cluster::l2_flush_begin() {
  impl_->l2_flush_begin();
}

bool Cluster::l2_flush_done() const {
  return impl_->l2_flush_done();
}

#ifdef VX_CFG_EXT_RASTER_ENABLE
RasterCore::Ptr& Cluster::raster_core() {
  return impl_->raster_core();
}
#endif

