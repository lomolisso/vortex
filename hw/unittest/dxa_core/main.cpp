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

// DXA core unit test: reset and idle cycles, then (in builds without DXA
// mbarriers) legacy multicast copies with dense and sparse cta masks. The
// receiver of mask bit r must get the tile at issuer + r * smem_stride and its
// barrier release at barrier + r warps, the receivers of each local-memory
// word in increasing bit order.

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <map>
#include <vector>
#include <VX_config.h>
#include <VX_types.h>
#include "VVX_dxa_core_top.h"
#include "verilated.h"
#ifdef VCD_OUTPUT
#include "verilated_vcd_c.h"
#endif
#ifdef SAIF_OUTPUT
#include "verilated_saif_c.h"
#endif

#if defined(VCD_OUTPUT) && defined(SAIF_OUTPUT)
#error "VCD_OUTPUT and SAIF_OUTPUT cannot both be defined"
#endif

#define MAX_SIM_CYCLES 1000

static VVX_dxa_core_top* dut;
static uint64_t sim_time = 0;

#ifdef VCD_OUTPUT
static VerilatedVcdC* vcd;
#endif
#ifdef SAIF_OUTPUT
static VerilatedSaifC* saif;
#endif

static void tick() {
    dut->clk ^= 1;
    dut->eval();
#ifdef VCD_OUTPUT
    vcd->dump(sim_time);
#endif
#ifdef SAIF_OUTPUT
    saif->dump(sim_time);
#endif
    ++sim_time;
}

static void clock_cycle() {
    tick(); // posedge
    tick(); // negedge
}

static void reset_dut(int cycles = 4) {
    dut->reset = 1;
    for (int i = 0; i < cycles; i++)
        clock_cycle();
    dut->reset = 0;
}

#if !defined(VX_CFG_EXT_MBAR_ENABLE)

static constexpr int log2_up(int n) {
    int b = 0;
    while ((1 << b) < n) ++b;
    return b;
}

// Field positions of dxa_req_data_t (packed, first field on top) in a build
// without DXA mbarriers: cta_mask, coords[0..4], meta, smem_addr, ...
static constexpr int REQ_MASK_LSB = 0;
static constexpr int REQ_META_LSB = VX_CFG_NUM_WARPS + 5 * 32;
static constexpr int REQ_SMEM_LSB = REQ_META_LSB + 32;

static constexpr int CL_BYTES   = VX_CFG_L1_LINE_SIZE;
static constexpr int WORD_BYTES = VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8);
// A local-memory write's attribute: {last, barrier address}.
static constexpr int BAR_W  = log2_up(VX_CFG_NUM_WARPS) + log2_up(VX_CFG_NUM_BARRIERS);
static constexpr int ATTR_W = (BAR_W > 0 ? BAR_W : 1) + 1;

template <typename W>
static void set_bits(W& w, int lsb, int width, uint64_t value) {
    for (int i = 0; i < width; ++i) {
        int b = lsb + i;
        uint32_t bit = (value >> i) & 1;
        w[b / 32] = (w[b / 32] & ~(1u << (b % 32))) | (bit << (b % 32));
    }
}

static void dcr_write(uint32_t addr, uint32_t data) {
    dut->dcr_req_valid = 1;
    dut->dcr_req_rw    = 1;
    dut->dcr_req_addr  = addr;
    dut->dcr_req_data  = data;
    clock_cycle();
    dut->dcr_req_valid = 0;
    dut->dcr_req_rw    = 0;
}

struct LmemWr {
    uint32_t addr;
    uint32_t attr;
    std::vector<uint32_t> data;
};

static uint32_t gmem_word(uint64_t line, int w) {
    return uint32_t((line << 8) | uint32_t(w)) ^ 0x5a000000u;
}

static uint32_t attr_last(uint32_t attr) { return (attr >> (ATTR_W - 1)) & 1; }
static uint32_t attr_bar(uint32_t attr)  { return attr & ((1u << (ATTR_W - 1)) - 1); }

// One legacy multicast of a 1D tile into smem_addr; returns every
// local-memory write in issue order.
static std::vector<LmemWr> run_copy(uint32_t cta_mask, uint32_t smem_addr, uint32_t meta, uint32_t expected) {
    std::vector<LmemWr> wrs;
    std::deque<uint64_t> gline;
    std::deque<uint32_t> gtag;
    std::deque<int>      gdelay;

    for (size_t i = 0; i < sizeof(dut->dxa_req_data) / sizeof(dut->dxa_req_data[0]); ++i)
        dut->dxa_req_data[i] = 0;
    set_bits(dut->dxa_req_data, REQ_MASK_LSB, VX_CFG_NUM_WARPS, cta_mask);
    set_bits(dut->dxa_req_data, REQ_META_LSB, 32, meta);
    set_bits(dut->dxa_req_data, REQ_SMEM_LSB, VX_CFG_LMEM_LOG_SIZE, smem_addr);
    dut->dxa_req_valid  = 1;
    dut->lmem_req_ready = ~0u;
    dut->gmem_req_ready = ~0u;

    for (int cyc = 0; cyc < 5000 && wrs.size() < expected; ++cyc) {
        // Responses, in order, a fixed latency after their request.
        bool rsp = !gline.empty() && gdelay.front() <= 0;
        dut->gmem_rsp_valid = rsp;
        if (rsp) {
            for (int w = 0; w < CL_BYTES / 4; ++w)
                dut->gmem_rsp_data[w] = gmem_word(gline.front(), w);
            dut->gmem_rsp_tag = gtag.front();
        }
        dut->eval();
        bool req_taken = dut->dxa_req_valid && dut->dxa_req_ready;
        if (dut->gmem_req_valid & 1) {
            gline.push_back(dut->gmem_req_addr);
            gtag.push_back(dut->gmem_req_tag);
            gdelay.push_back(6);
        }
        if (rsp && (dut->gmem_rsp_ready & 1)) {
            gline.pop_front();
            gtag.pop_front();
            gdelay.pop_front();
        }
        if (dut->lmem_req_valid & 1) {
            LmemWr wr;
            wr.addr = dut->lmem_req_addr;
            wr.attr = dut->lmem_req_attr;
            for (int w = 0; w < WORD_BYTES / 4; ++w)
                wr.data.push_back(dut->lmem_req_data[w]);
            wrs.push_back(wr);
        }
        clock_cycle();
        if (req_taken) dut->dxa_req_valid = 0;
        for (auto& d : gdelay) --d;
    }
    dut->gmem_rsp_valid = 0;
    for (int i = 0; i < 50; ++i) {
        dut->eval();
        if (dut->lmem_req_valid & 1) {
            printf("unexpected local-memory write after the copy\n");
            wrs.push_back({});
        }
        clock_cycle();
    }
    return wrs;
}

// Receiver of bit r (the set bits in increasing order, one per word) takes
// every tile word at base + r * stride_words, and its one last write releases
// bar0 + r * bar_step.
static bool check_copy(const char* name, uint32_t mask, const std::vector<LmemWr>& wrs,
                       uint32_t base_word, uint32_t stride_words, uint32_t tile_words,
                       uint32_t bar0, uint32_t bar_step, uint64_t gline0) {
    std::vector<int> rcv;
    for (int r = 0; r < VX_CFG_NUM_WARPS; ++r) {
        if (mask & (1u << r))
            rcv.push_back(r);
    }
    if (wrs.size() != rcv.size() * tile_words) {
        printf("%s: %zu local-memory writes, expected %zu\n", name, wrs.size(), rcv.size() * tile_words);
        return false;
    }
    std::map<int, int> released;
    for (size_t i = 0; i < wrs.size(); ++i) {
        int r = rcv[i % rcv.size()];
        const auto& wr = wrs[i];
        uint32_t blk = base_word + r * stride_words;
        uint32_t k = wr.addr - blk;
        if (k >= tile_words) {
            printf("%s: write %zu (bit %d) at word 0x%x, outside its block at word 0x%x\n",
                   name, i, r, wr.addr, blk);
            return false;
        }
        uint32_t byte_off = k * WORD_BYTES;
        uint64_t line = gline0 + byte_off / CL_BYTES;
        for (int w = 0; w < WORD_BYTES / 4; ++w) {
            uint32_t exp = gmem_word(line, (byte_off % CL_BYTES) / 4 + w);
            if (wr.data[w] != exp) {
                printf("%s: write %zu (bit %d) word %u lane %d: 0x%08x, expected 0x%08x\n",
                       name, i, r, k, w, wr.data[w], exp);
                return false;
            }
        }
        if (attr_last(wr.attr)) {
            if (released.count(r)) {
                printf("%s: bit %d released twice\n", name, r);
                return false;
            }
            released[r] = 1;
            uint32_t bar = attr_bar(wr.attr);
            if (bar != bar0 + r * bar_step) {
                printf("%s: bit %d releases barrier 0x%x, expected 0x%x\n", name, r, bar, bar0 + r * bar_step);
                return false;
            }
        }
    }
    if (released.size() != rcv.size()) {
        printf("%s: %zu receivers released, expected %zu\n", name, released.size(), rcv.size());
        return false;
    }
    printf("%s: %zu writes to %zu receivers OK\n", name, wrs.size(), rcv.size());
    return true;
}

static bool run_multicast() {
    // Descriptor 0: a 1D tile of 32 four-byte elements, 1 KB destination stride.
    const uint32_t tile_elems  = 32;
    const uint32_t smem_stride = 0x400;
    const uint64_t gbase       = 0x10000;
    const uint32_t d           = VX_DCR_DXA_DESC_BASE;
    dcr_write(d + VX_DCR_DXA_DESC_BASE_LO_OFF, uint32_t(gbase));
    dcr_write(d + VX_DCR_DXA_DESC_BASE_HI_OFF, 0);
    dcr_write(d + VX_DCR_DXA_DESC_SIZE0_OFF, 4096);
    dcr_write(d + VX_DCR_DXA_DESC_META_OFF, 1u | (2u << 3)); // rank 1, 4-byte elements, row-major
    dcr_write(d + VX_DCR_DXA_DESC_TILESIZE01_OFF, tile_elems);
    dcr_write(d + VX_DCR_DXA_DESC_CFILL_OFF, 0);
    dcr_write(d + VX_DCR_DXA_DESC_SMEM_STRIDE_OFF, smem_stride);
    for (int i = 0; i < 4; ++i)
        clock_cycle();

    const uint32_t smem_addr  = 0x100;
    const uint32_t meta       = 0x80000000u | (1u << (4 + 8)); // descriptor 0, barrier 1 of warp 0
    const uint32_t tile_words = tile_elems * 4 / WORD_BYTES;
    const uint32_t base_word  = smem_addr / WORD_BYTES;
    const uint32_t sw         = smem_stride / WORD_BYTES;
    const uint64_t gline0     = gbase / CL_BYTES;

    // A dense pair gives the barrier of bit 0 and the step per bit.
    auto dense = run_copy(0x3, smem_addr, meta, 2 * tile_words);
    uint32_t bar0 = 0, bar1 = 0;
    for (size_t i = 0; i < dense.size(); ++i) {
        if (attr_last(dense[i].attr)) {
            if (i % 2 == 0) bar0 = attr_bar(dense[i].attr);
            else            bar1 = attr_bar(dense[i].attr);
        }
    }
    uint32_t step = bar1 - bar0;
    bool ok = check_copy("mcast-0x3", 0x3, dense, base_word, sw, tile_words, bar0, step, gline0);
    for (uint32_t m : {0x5u, 0xfu, 0xau, 0x9u, 0x6u}) {
        if (m >> VX_CFG_NUM_WARPS)
            continue;
        char name[32];
        snprintf(name, sizeof(name), "mcast-0x%x", m);
        auto wrs = run_copy(m, smem_addr, meta, __builtin_popcount(m) * tile_words);
        ok = check_copy(name, m, wrs, base_word, sw, tile_words, bar0, step, gline0) && ok;
    }
    return ok;
}

#endif

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    dut = new VVX_dxa_core_top;
    dut->clk   = 0;
    dut->reset = 0;

#ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    vcd = new VerilatedVcdC;
    dut->trace(vcd, 99);
    const char* vcd_file = std::getenv("VCD_FILE");
    vcd->open(vcd_file ? vcd_file : "trace.vcd");
#endif
#ifdef SAIF_OUTPUT
    Verilated::traceEverOn(true);
    saif = new VerilatedSaifC;
    dut->trace(saif, 99);
    const char* saif_file = std::getenv("SAIF_FILE");
    saif->open(saif_file ? saif_file : "trace.saif");
#endif

    // ---- tie off unused inputs -------------------------------------------
    dut->dcr_req_valid  = 0;
    dut->dcr_req_rw     = 0;
    dut->dxa_req_valid  = 0;
    dut->gmem_req_ready = ~0ULL; // all ready
    dut->gmem_rsp_valid = 0;
    dut->lmem_req_ready = ~0ULL; // all ready

    reset_dut();

    // ---- run a few idle cycles -------------------------------------------
    for (int i = 0; i < MAX_SIM_CYCLES; i++)
        clock_cycle();

    bool ok = true;
#if !defined(VX_CFG_EXT_MBAR_ENABLE)
    ok = run_multicast();
#endif

    std::cout << (ok ? "PASSED" : "FAILED") << std::endl;

#ifdef VCD_OUTPUT
    vcd->close();
    delete vcd;
#endif
#ifdef SAIF_OUTPUT
    saif->close();
    delete saif;
#endif
    dut->final();
    delete dut;
    return ok ? 0 : 1;
}
