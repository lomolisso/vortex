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
// word in increasing bit order. With DXA mbarriers, bulk copies instead: the
// engine reads the source lines from the issuer's local memory, whose
// responses come back in order after a random delay under random request
// back-pressure, and must write exactly the destination bytes, complete with
// one last beat carrying the mbarrier, the byte count and the slot, and never
// refuse a read response.

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <map>
#include <vector>
#include <type_traits>
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

#if defined(VX_CFG_EXT_MBAR_ENABLE)

static constexpr int clog2(int n) {
    int b = 0;
    while ((1 << b) < n) ++b;
    return b;
}
static constexpr int up1(int b) { return b > 0 ? b : 1; }
static constexpr int max2(int a, int b) { return a > b ? a : b; }

static constexpr int LMEM_LOG   = VX_CFG_LMEM_LOG_SIZE;
static constexpr int LMEM_BYTES = 1 << LMEM_LOG;
static constexpr int WORD_BYTES = VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8);
static constexpr int NW_W       = up1(clog2(VX_CFG_NUM_WARPS));
static constexpr int NCC        = max2(1, VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE) * VX_CFG_SOCKET_SIZE;
static constexpr int CORE_W     = up1(clog2(NCC));
static constexpr int MAXC       = VX_CFG_DLMEM_MAX_CLUSTER;
static constexpr int PEER_W     = 1 + CORE_W + NW_W + 4;
static constexpr int BAR_W      = up1(clog2(VX_CFG_NUM_WARPS) + clog2(VX_CFG_NUM_BARRIERS));
static constexpr int PAY_W      = max2(BAR_W, LMEM_LOG - 3);

// dxa_req_data_t from its last (lowest) field up, as far as the bench sets it.
static constexpr int R_SIZE   = 0;
static constexpr int R_SRC    = R_SIZE + LMEM_LOG + 1;
static constexpr int R_BULK   = R_SRC + LMEM_LOG;
static constexpr int R_PEERS  = R_BULK + 1;
static constexpr int R_MCMASK = R_PEERS + MAXC * PEER_W;
static constexpr int R_MCCL   = R_MCMASK + MAXC;
static constexpr int R_ROOM   = R_MCCL + 1;
static constexpr int R_WIN    = R_ROOM + LMEM_LOG + 1;
static constexpr int R_MBAR   = R_WIN + 1;
static constexpr int R_DSLOT  = R_MBAR + LMEM_LOG - 3;
static constexpr int R_DCORE  = R_DSLOT + NW_W;
static constexpr int R_REMOTE = R_DCORE + CORE_W;
static constexpr int R_GEN    = R_REMOTE + 1;
static constexpr int R_SLOT   = R_GEN + 4;
static constexpr int R_MASK   = R_SLOT + NW_W;
static constexpr int R_COORDS = R_MASK + VX_CFG_NUM_WARPS;
static constexpr int R_META   = R_COORDS + 5 * 32;
static constexpr int R_SMEM   = R_META + 32;
static constexpr int R_WID    = R_SMEM + LMEM_LOG;

// dxa_lmem_attr_t from its last field up.
static constexpr int A_REMOTE = CORE_W;
static constexpr int A_GEN    = A_REMOTE + 1;
static constexpr int A_SLOT   = A_GEN + 4;
static constexpr int A_TX     = A_SLOT + NW_W;
static constexpr int A_PAY    = A_TX + LMEM_LOG + 1;
static constexpr int A_MODE   = A_PAY + PAY_W;
static constexpr int A_LAST   = A_MODE + 1;

template <typename W>
static void set_bits(W& w, int lsb, int width, uint64_t value) {
    for (int i = 0; i < width; ++i) {
        int b = lsb + i;
        uint32_t bit = (value >> i) & 1;
        w[b / 32] = (w[b / 32] & ~(1u << (b % 32))) | (bit << (b % 32));
    }
}

// Bit b of a Verilator port, scalar or wide.
template <typename T>
static uint32_t port_bit(const T& v, int b) {
    if constexpr (std::is_integral<T>::value) {
        return uint32_t((uint64_t(v) >> b) & 1);
    } else {
        return (v[b / 32] >> (b % 32)) & 1;
    }
}

static uint64_t field(uint64_t v, int lsb, int width) {
    return (v >> lsb) & ((width >= 64) ? ~0ull : ((1ull << width) - 1));
}

static uint64_t rng_state = 1;
static uint32_t rnd() {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(rng_state >> 33);
}

static uint8_t pattern(uint32_t i) { return uint8_t((i * 37u + (i >> 8) * 11u + 5u) ^ 0xa5u); }

struct RdRsp {
    uint32_t addr;
    uint64_t tag;
    uint64_t due;
};

struct BulkCopy {
    uint32_t src, dst, size, mbar_off, slot, gen;
};

// One or more bulk copies queued back to back; checks every byte of the local
// memory and each copy's completion beat. A read's response comes min_lat to
// max_lat cycles after the request is taken.
static bool run_bulk(const char* name, const std::vector<BulkCopy>& copies, int ready_pct, int max_lat,
                     int min_lat = 1) {
    std::vector<uint8_t> mem(LMEM_BYTES), ref(LMEM_BYTES);
    for (int i = 0; i < LMEM_BYTES; ++i)
        mem[i] = ref[i] = pattern(i);
    for (auto& c : copies)
        for (uint32_t i = 0; i < c.size; ++i)
            ref[c.dst + i] = pattern(c.src + i);

    std::deque<RdRsp> rsps;
    size_t next_req = 0, lasts = 0;
    uint64_t reads = 0, writes = 0, cyc = 0;
    bool ok = true;

    auto load_req = [&](const BulkCopy& c) {
        for (size_t i = 0; i < sizeof(dut->dxa_req_data) / sizeof(dut->dxa_req_data[0]); ++i)
            dut->dxa_req_data[i] = 0;
        set_bits(dut->dxa_req_data, R_SIZE, LMEM_LOG + 1, c.size);
        set_bits(dut->dxa_req_data, R_SRC, LMEM_LOG, c.src);
        set_bits(dut->dxa_req_data, R_BULK, 1, 1);
        set_bits(dut->dxa_req_data, R_MBAR, LMEM_LOG - 3, c.mbar_off);
        set_bits(dut->dxa_req_data, R_GEN, 4, c.gen);
        set_bits(dut->dxa_req_data, R_SLOT, NW_W, c.slot);
        set_bits(dut->dxa_req_data, R_META, 32, 0xC0000000u);
        set_bits(dut->dxa_req_data, R_SMEM, LMEM_LOG, c.dst);
        dut->dxa_req_valid = 1;
    };
    load_req(copies[0]);

    for (; cyc < 200000 && lasts < copies.size(); ++cyc) {
        bool rsp = !rsps.empty() && rsps.front().due <= cyc;
        dut->lmem_rsp_valid = rsp;
        if (rsp) {
            const auto& r = rsps.front();
            for (int w = 0; w < WORD_BYTES / 4; ++w) {
                uint32_t v = 0;
                for (int b = 0; b < 4; ++b)
                    v |= uint32_t(mem[r.addr * WORD_BYTES + w * 4 + b]) << (8 * b);
                dut->lmem_rsp_data[w] = v;
            }
            dut->lmem_rsp_tag = r.tag;
        }
        dut->lmem_req_ready = (int(rnd() % 100) < ready_pct) ? 1 : 0;
        dut->eval();
        if (rsp && !(dut->lmem_rsp_ready & 1)) {
            printf("%s: read response refused at cycle %lu\n", name, (unsigned long)cyc);
            ok = false;
        }
        bool req_taken = dut->dxa_req_valid && dut->dxa_req_ready;
        if ((dut->lmem_req_valid & 1) && (dut->lmem_req_ready & 1)) {
            uint32_t addr = dut->lmem_req_addr;
            if (!(dut->lmem_req_rw & 1)) {
                ++reads;
                uint64_t prev = rsps.empty() ? cyc : rsps.back().due;
                uint64_t due = cyc + min_lat + rnd() % (max_lat - min_lat + 1);
                rsps.push_back({addr, (uint64_t)dut->lmem_req_tag, due > prev ? due : prev});
            } else {
                ++writes;
                for (int b = 0; b < WORD_BYTES; ++b) {
                    if (port_bit(dut->lmem_req_byteen, b)) {
                        uint32_t w = dut->lmem_req_data[b / 4];
                        mem[addr * WORD_BYTES + b] = uint8_t(w >> (8 * (b % 4)));
                    }
                }
                uint64_t attr = dut->lmem_req_attr;
                if (field(attr, A_LAST, 1)) {
                    const auto& c = copies[lasts];
                    if (!field(attr, A_MODE, 1) || field(attr, A_PAY, PAY_W) != c.mbar_off
                     || field(attr, A_TX, LMEM_LOG + 1) != c.size || field(attr, A_SLOT, NW_W) != c.slot
                     || field(attr, A_GEN, 4) != c.gen || field(attr, A_REMOTE, 1)) {
                        printf("%s: copy %zu completion attr 0x%lx: mode=%lu mbar=0x%lx tx=%lu slot=%lu gen=%lu\n",
                               name, lasts, (unsigned long)attr, (unsigned long)field(attr, A_MODE, 1),
                               (unsigned long)field(attr, A_PAY, PAY_W), (unsigned long)field(attr, A_TX, LMEM_LOG + 1),
                               (unsigned long)field(attr, A_SLOT, NW_W), (unsigned long)field(attr, A_GEN, 4));
                        ok = false;
                    }
                    ++lasts;
                }
            }
        }
        if (rsp && (dut->lmem_rsp_ready & 1))
            rsps.pop_front();
        clock_cycle();
        if (req_taken) {
            if (++next_req < copies.size())
                load_req(copies[next_req]);
            else
                dut->dxa_req_valid = 0;
        }
    }
    dut->dxa_req_valid  = 0;
    dut->lmem_rsp_valid = 0;
    dut->lmem_req_ready = 1;
    for (int i = 0; i < 50; ++i) {
        dut->eval();
        if (dut->lmem_req_valid & 1) {
            printf("%s: local-memory request after the copies\n", name);
            ok = false;
        }
        clock_cycle();
    }
    if (lasts != copies.size()) {
        printf("%s: %zu of %zu copies completed\n", name, lasts, copies.size());
        return false;
    }
    if (!rsps.empty()) {
        printf("%s: %zu reads never answered\n", name, rsps.size());
        ok = false;
    }
    // Every line of each source is read once, whole.
    uint64_t exp_reads = 0;
    for (auto& c : copies) {
        uint64_t lines = (c.src + c.size + VX_CFG_L1_LINE_SIZE - 1) / VX_CFG_L1_LINE_SIZE - c.src / VX_CFG_L1_LINE_SIZE;
        exp_reads += lines * ((VX_CFG_L1_LINE_SIZE > WORD_BYTES) ? (VX_CFG_L1_LINE_SIZE / WORD_BYTES) : 1);
    }
    if (reads != exp_reads) {
        printf("%s: %lu local-memory reads, expected %lu\n", name, (unsigned long)reads, (unsigned long)exp_reads);
        ok = false;
    }
    for (int i = 0; i < LMEM_BYTES; ++i) {
        if (mem[i] != ref[i]) {
            printf("%s: byte 0x%x = 0x%02x, expected 0x%02x\n", name, i, mem[i], ref[i]);
            ok = false;
            break;
        }
    }
    if (ok)
        printf("%s: %zu copies, %lu reads, %lu writes in %lu cycles OK\n", name, copies.size(),
               (unsigned long)reads, (unsigned long)writes, (unsigned long)cyc);
    return ok;
}

static bool run_bulk_all() {
    static_assert(R_WID + NW_W <= int(sizeof(((VVX_dxa_core_top*)nullptr)->dxa_req_data)) * 8,
                  "request layout wider than the port");
    const uint32_t q = LMEM_BYTES / 4;
    bool ok = true;
    ok = run_bulk("bulk-16", {{0x100, q + 0x200, 16, 0x21, 1, 3}}, 100, 1) && ok;
    ok = run_bulk("bulk-16-off", {{0x110, q + 0x230, 16, 0x22, 2, 7}}, 100, 1) && ok;
    ok = run_bulk("bulk-48", {{0x1d0, q + 0x310, 48, 0x23, 1, 1}}, 100, 3) && ok;
    ok = run_bulk("bulk-208", {{0x1f0, q + 0x420, 208, 0x24, 3, 15}}, 80, 4) && ok;
    ok = run_bulk("bulk-2064", {{0x40, q + 0x830, 2064, 0x25, 0, 2}}, 70, 5) && ok;
    ok = run_bulk("bulk-tile", {{0, 2 * q, 4096 < q ? 4096u : q, 0x26, 1, 4}}, 100, 2) && ok;
    ok = run_bulk("bulk-stream", {{0x30, q + 0x10, 160, 0x27, 1, 5},
                                  {0x200, q + 0x400, 336, 0x28, 2, 6},
                                  {0x610, 2 * q + 0x20, 1040, 0x29, 3, 9}}, 60, 6) && ok;
    // Throughput of 4 KB copies, aligned and misaligned, with the read latency
    // of a socket of one core (1 cycle past the engine's output buffer) and of
    // a larger one (3, through the core switch's registers).
    const uint32_t big = 4096 < q ? 4096u : q;
    for (int lat : {1, 3}) {
        char nm[64];
        snprintf(nm, sizeof(nm), "tput-aligned-lat%d", lat);
        ok = run_bulk(nm, {{0, 2 * q, big, 0x2a, 1, 4}}, 100, lat, lat) && ok;
        snprintf(nm, sizeof(nm), "tput-misaligned-lat%d", lat);
        ok = run_bulk(nm, {{0x30, 2 * q + 0x10, big - 0x40, 0x2b, 1, 4}}, 100, lat, lat) && ok;
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
#if defined(VX_CFG_EXT_MBAR_ENABLE)
    dut->lmem_rsp_valid = 0;
    dut->slot_stride    = 0x400;
#endif

    reset_dut();

    // ---- run a few idle cycles -------------------------------------------
    for (int i = 0; i < MAX_SIM_CYCLES; i++)
        clock_cycle();

    bool ok = true;
#if !defined(VX_CFG_EXT_MBAR_ENABLE)
    ok = run_multicast();
#else
    ok = run_bulk_all();
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
