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
// refuse a read response. Then tensor loads (PTX tile and im2col modes),
// global-to-local bulk copies and prefetches, against out-of-order global
// responses under back-pressure: every local-memory byte must match the
// reference box walk (signed coordinates, fill on both sides of every
// dimension, element strides, every swizzle, interleave, im2col, a cluster
// multicast), each transfer's completion beat must be its last write and carry
// its byte count, every in-bounds line must be read once per row token, and a
// prefetch must read its lines and write nothing. Then the store direction:
// bulk and tensor stores (out-of-bounds edges, swizzle, strides, interleave,
// im2col), global reductions of every op x type through a last-level-cache
// atomic model, tensor reductions, and cluster reductions as DLMEM atomic
// packets: global memory and the destination block must match dxa_ref.h byte
// for byte, the line writes, atomics, packets, local-memory reads and fences
// must be the counts of the token walk, and every bulk_group operation must
// report one read_done and one write_done, the last after its final response.

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <array>
#include <deque>
#include <map>
#include <set>
#include <vector>
#include <type_traits>
#include <functional>
#include <string>
#include <VX_config.h>
#include <VX_types.h>
#if defined(VX_CFG_EXT_MBAR_ENABLE)
#include <dxa_ref.h>
namespace dxa_ref = vortex::dxa_ref;
namespace amo_ref = vortex::amo_ref;
#endif
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

// ---- Tensor loads, global bulk copies and prefetches ----------------------

#ifdef NDEBUG
static constexpr int UUID_W = 1;
#else
static constexpr int UUID_W = 44;
#endif
static constexpr int NC_W     = up1(clog2(VX_CFG_NUM_CORES));
static constexpr int MADDR_W  = VX_CFG_MEM_ADDR_WIDTH;
// The tensor-load fields sit above the bulk-copy ones: core_id, uuid, then blk_base up.
static constexpr int R_CORE   = R_WID + NW_W + UUID_W;
static constexpr int R_BLK    = R_CORE + NC_W;
static constexpr int R_GSRC   = R_BLK + LMEM_LOG;
static constexpr int R_OFFS   = R_GSRC + MADDR_W;
static constexpr int R_PF     = R_OFFS + 48;
static constexpr int R_BLOAD  = R_PF + 1;
static constexpr int R_TLOAD  = R_BLOAD + 1;
static constexpr int R_END    = R_TLOAD + 1;
static constexpr int CL       = VX_CFG_L1_LINE_SIZE;

template <typename T>
static int port_width(int width) {
    int w = int(sizeof(T)) * 8;
    return width < w ? width : w;
}
template <typename T>
static uint64_t port_bits(const T& v, int lsb, int width) {
    uint64_t r = 0;
    width = port_width<T>(width);
    for (int i = 0; i < width; ++i)
        r |= uint64_t(port_bit(v, lsb + i)) << i;
    return r;
}
template <typename T>
static void port_set(T& v, int lsb, int width, uint64_t value) {
    width = port_width<T>(width);
    if constexpr (std::is_integral<T>::value) {
        for (int i = 0; i < width; ++i) {
            T bit = T(1) << (lsb + i);
            v = ((value >> i) & 1) ? (v | bit) : (v & ~bit);
        }
    } else {
        set_bits(v, lsb, width, value);
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

// Every address bit changes the byte, so a wrong high address bit shows.
static uint8_t gmem_byte(uint64_t a) {
    return uint8_t((a * 131u) ^ (a >> 8) * 7u ^ (a >> 16) * 13u ^ (a >> 24) * 29u ^ (a >> 32) * 37u
                   ^ (a >> 40) * 41u ^ 0x3cu);
}

// A transfer, as the request and descriptor the engine receives.
struct Xfer {
    std::string name;
    // descriptor
    uint32_t rank = 1, esz = 2, swz = 0, il = 0, im2col = 0;
    uint64_t base = 0;
    uint32_t size[5] = {1, 1, 1, 1, 1};
    uint32_t stride[4] = {0, 0, 0, 0};    // bytes, dims 1..4
    uint32_t es[5] = {1, 1, 1, 1, 1};
    uint32_t tile[5] = {1, 1, 1, 1, 1};
    uint32_t cpp = 0, ppc = 0;            // im2col
    int32_t  lower[3] = {0, 0, 0}, upper[3] = {0, 0, 0};
    uint32_t cfill = 0;
    // request
    int32_t  c[5] = {0, 0, 0, 0, 0};
    uint32_t offs[3] = {0, 0, 0};
    uint32_t smem = 0, blk_base = 0, mbar = 0, slot = 1, gen = 3;
    bool     prefetch = false;
    bool     bload = false;               // a global bulk copy (base, bytes)
    uint32_t bytes = 0;
    uint32_t mc_mask = 0;                 // a cluster multicast to local slots 0..MAXC-1
    uint32_t room = 0;                    // bytes from smem to its block's end (0: to the memory's end)
    // tensor maps
    bool     use_map = false;             // the request names the map at `map` instead of a slot
    uint64_t map = 0;
    int      tmop = 0;                    // a tensormap op: 1 prefetch, 2 invalidate (no transfer)
    bool     kmu = false;                 // a launch: a KMU-state DCR write (no transfer)
    uint32_t etype = 7;                   // the map's element type (word 23)
    uint32_t w22 = 0;                     // the map's stride bits 39:32 (must be 0)
    bool     rank0 = false;               // a map with rank 0
    bool     kill = false;                // a launch lands during this request's map fetch
    bool     in_loop = false;             // a launch driven in a cycle of the run loop
};

// Tensor-map support of the request and memory models, defined with the map vectors.
static void load_tm(const Xfer& x);
static bool tm_line(uint64_t line, uint8_t* bytes);
static void tm_store(const Xfer& x);
static int  g_rsp_rev = 0;                // answer a map's lines last first
static bool g_maps_first = false;         // every engine read follows the last map read
static uint64_t g_map_reads = 0;
// A DCR write inside the run loop, given the map responses accepted so far and
// whether a map line is presented this cycle; and a hook after the slots are programmed.
static std::function<bool(uint64_t, bool, uint32_t&, uint32_t&)> g_dcr_hook;
static std::function<void()> g_after_program;
static std::function<bool(uint64_t)> g_map_gate;   // true: no map line is answered this cycle
static uint64_t g_dcr_held = 0, g_dcr_taken = 0;   // map responses presented with a slot DCR write

static dxa_ref::Desc ref_desc(const Xfer& x) {
    dxa_ref::Desc d;
    d.base = x.base; d.rank = x.rank; d.elem = 1u << x.esz; d.layout = 0;
    d.swizzle = x.swz; d.interleave = x.il; d.im2col = x.im2col != 0;
    for (int i = 0; i < 5; ++i) {
        d.size[i] = x.size[i]; d.estride[i] = x.es[i]; d.tile[i] = x.tile[i];
    }
    for (int i = 0; i < 4; ++i)
        d.stride[i] = x.stride[i];
    d.channels = x.cpp; d.pixels = x.ppc;
    for (int i = 0; i < 3; ++i) {
        d.lower[i] = x.lower[i]; d.upper[i] = x.upper[i];
    }
    d.cfill = x.cfill;
    return d;
}

// The golden box walk (dxa_ref.h): f(dense offset, in bounds, global address)
// per element in destination order; returns the tx bytes. A bulk copy is one
// row of doublewords.
static uint32_t walk(const Xfer& x, const std::function<void(uint32_t, bool, uint64_t)>& f) {
    uint32_t e = 1u << x.esz;
    if (x.bload || x.bytes) {
        for (uint32_t i = 0; i < x.bytes / e; ++i)
            f(i * e, true, x.base + uint64_t(i) * e);
        return x.bytes;
    }
    auto d = ref_desc(x);
    dxa_ref::Req q;
    for (int i = 0; i < 5; ++i)
        q.coord[i] = x.c[i];
    for (int i = 0; i < 3; ++i)
        q.offs[i] = x.offs[i];
    dxa_ref::for_each_element(d, q, f);
    return dxa_ref::tx_bytes(d);
}

static uint32_t box_n(const Xfer& x, int d) { return dxa_ref::n_elems(ref_desc(x), uint32_t(d)); }

// The engine's line reads: one per in-bounds row segment's line (a gathered
// segment reads each line holding one of its elements).
static uint64_t expect_reads(const Xfer& x) {
    uint64_t reads = 0;
    int64_t row_first = -1, row_last = -1;
    uint32_t e = 1u << x.esz, cnt = 0, per_row = x.bytes ? x.bytes / e : (x.im2col ? x.cpp : box_n(x, 0));
    // In-bounds elements of a row are one contiguous run (dim-0 split).
    walk(x, [&](uint32_t, bool in, uint64_t a) {
        if (in) {
            if (row_first < 0)
                row_first = int64_t(a);
            row_last = int64_t(a);
        }
        if (++cnt == per_row) {
            if (row_first >= 0)
                reads += uint64_t(row_last / CL - row_first / CL + 1);
            row_first = row_last = -1;
            cnt = 0;
        }
    });
    return reads;
}

static void program_desc(uint32_t slot, const Xfer& x) {
    uint32_t d = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
    uint32_t meta = x.rank | (x.esz << 3) | (x.swz << 7) | (x.il << 9) | (x.im2col << 13);
    dcr_write(d + VX_DCR_DXA_DESC_BASE_LO_OFF, uint32_t(x.base));
    dcr_write(d + VX_DCR_DXA_DESC_BASE_HI_OFF, uint32_t(x.base >> 32));
    for (int i = 0; i < 5; ++i)
        dcr_write(d + VX_DCR_DXA_DESC_SIZE0_OFF + i, x.size[i]);
    for (int i = 0; i < 4; ++i)
        dcr_write(d + VX_DCR_DXA_DESC_STRIDE0_OFF + i, x.stride[i]);
    dcr_write(d + VX_DCR_DXA_DESC_META_OFF, meta);
    for (int i = 0; i < 5; ++i)
        dcr_write(d + VX_DCR_DXA_DESC_ESTRIDE0_OFF + i, x.es[i]);
    uint32_t t23;
    if (x.im2col) {
        dcr_write(d + VX_DCR_DXA_DESC_TILESIZE01_OFF, (x.ppc << 16) | x.cpp);
        auto f = [](int32_t v, int w) { return uint32_t(v) & ((1u << w) - 1); };
        if (x.rank == 3)
            t23 = f(x.lower[0], 16) | (f(x.upper[0], 16) << 16);
        else if (x.rank == 4)
            t23 = f(x.lower[0], 8) | (f(x.lower[1], 8) << 8) | (f(x.upper[0], 8) << 16) | (f(x.upper[1], 8) << 24);
        else
            t23 = f(x.lower[0], 5) | (f(x.lower[1], 5) << 5) | (f(x.lower[2], 5) << 10)
                | (f(x.upper[0], 5) << 15) | (f(x.upper[1], 5) << 20) | (f(x.upper[2], 5) << 25);
    } else {
        dcr_write(d + VX_DCR_DXA_DESC_TILESIZE01_OFF, (x.tile[1] << 16) | x.tile[0]);
        t23 = (x.tile[3] << 16) | x.tile[2];
    }
    dcr_write(d + VX_DCR_DXA_DESC_TILESIZE23_OFF, t23);
    dcr_write(d + VX_DCR_DXA_DESC_TILESIZE4_OFF, x.tile[4]);
    dcr_write(d + VX_DCR_DXA_DESC_CFILL_OFF, x.cfill);
    dcr_write(d + VX_DCR_DXA_DESC_SMEM_STRIDE_OFF, 0);
}

static const uint32_t MC_STRIDE = 0x1000;

static void load_xfer(uint32_t slot, const Xfer& x) {
    for (size_t i = 0; i < sizeof(dut->dxa_req_data) / sizeof(dut->dxa_req_data[0]); ++i)
        dut->dxa_req_data[i] = 0;
    if (x.bload || (x.prefetch && x.bytes)) {
        set_bits(dut->dxa_req_data, R_SIZE, LMEM_LOG + 1, x.bytes);
        set_bits(dut->dxa_req_data, R_GSRC, MADDR_W, x.base);
        set_bits(dut->dxa_req_data, R_BLOAD, 1, x.bload);
        set_bits(dut->dxa_req_data, R_META, 32, 0xC0000000u);
    } else {
        set_bits(dut->dxa_req_data, R_TLOAD, 1, 1);
        set_bits(dut->dxa_req_data, R_META, 32, 0xC0000000u | slot);
        for (int i = 0; i < 5; ++i)
            set_bits(dut->dxa_req_data, R_COORDS + 32 * i, 32, uint32_t(x.c[i]));
        for (int i = 0; i < 3; ++i)
            set_bits(dut->dxa_req_data, R_OFFS + 16 * i, 16, x.offs[i]);
    }
    set_bits(dut->dxa_req_data, R_PF, 1, x.prefetch);
    set_bits(dut->dxa_req_data, R_SMEM, LMEM_LOG, x.smem);
    set_bits(dut->dxa_req_data, R_BLK, LMEM_LOG, x.blk_base);
    set_bits(dut->dxa_req_data, R_MBAR, LMEM_LOG - 3, x.mbar);
    set_bits(dut->dxa_req_data, R_GEN, 4, x.gen);
    set_bits(dut->dxa_req_data, R_SLOT, NW_W, x.slot);
    set_bits(dut->dxa_req_data, R_ROOM, LMEM_LOG + 1, x.room ? x.room : LMEM_BYTES - x.smem);
    if (x.mc_mask) {
        set_bits(dut->dxa_req_data, R_MCCL, 1, 1);
        set_bits(dut->dxa_req_data, R_ROOM, LMEM_LOG + 1, MC_STRIDE - (x.smem % MC_STRIDE));
        set_bits(dut->dxa_req_data, R_MCMASK, MAXC, x.mc_mask);
        for (int r = 0; r < MAXC; ++r) {
            // Rank r: slot r of this core, generation 8 + r.
            uint64_t peer = uint64_t(8 + r) | (uint64_t(r) << 4);
            set_bits(dut->dxa_req_data, R_PEERS + r * PEER_W, PEER_W, peer);
        }
    }
    dut->dxa_req_valid = 1;
}

struct GRsp {
    uint64_t line;
    uint64_t tag;
    uint64_t due;
};

// Runs the transfers back to back (or one at a time with `serial`) and
// checks the local-memory image, the completions and the reads.
static bool run_tensor(const char* name, std::vector<Xfer> xs, int ready_pct, int max_lat, bool serial = false) {
    std::vector<uint8_t> mem(LMEM_BYTES), ref(LMEM_BYTES);
    for (int i = 0; i < LMEM_BYTES; ++i)
        mem[i] = ref[i] = pattern(i);
    struct Done { uint32_t slot, gen, mbar, tx; };
    std::vector<Done> dones;
    uint64_t exp_reads = 0;
    for (auto& x : xs) {
        if (x.tmop || x.kmu)
            continue;
        if (!x.bload && !x.bytes && !x.rank0) {
            const char* bad = dxa_ref::check_tensor(ref_desc(x), x.smem - x.blk_base, x.prefetch);
            if (bad) {
                printf("%s: vector %s breaks a rule: %s\n", name, x.name.c_str(), bad);
                return false;
            }
        }
        exp_reads += expect_reads(x);
        if (x.prefetch)
            continue;
        std::vector<uint32_t> rcv;
        if (x.mc_mask) {
            for (int r = 0; r < MAXC; ++r)
                if (x.mc_mask & (1u << r))
                    rcv.push_back(r * MC_STRIDE);
        } else {
            rcv.push_back(x.blk_base);
        }
        uint32_t tx = 0;
        for (auto base : rcv) {
            uint32_t rel0 = x.smem - x.blk_base;
            tx = walk(x, [&](uint32_t off, bool in, uint64_t a) {
                for (uint32_t b = 0; b < (1u << x.esz); ++b) {
                    uint32_t r = dxa_ref::swizzle(rel0 + off + b, x.swz);
                    ref[base + r] = in ? gmem_byte(a + b) : dxa_ref::fill_byte(x.cfill, r);
                }
            });
        }
        for (size_t i = 0; i < rcv.size(); ++i) {
            uint32_t slot = x.mc_mask ? 0 : x.slot, gen = x.mc_mask ? 0 : x.gen, mbar = x.mbar;
            if (x.mc_mask) {
                int r = 0, k = int(i);
                for (; r < MAXC; ++r)
                    if ((x.mc_mask >> r) & 1) { if (k == 0) break; --k; }
                slot = r; gen = 8 + r; mbar = (r * MC_STRIDE) / 8 + x.mbar;
            }
            dones.push_back({slot, gen, mbar, tx});
        }
    }
    std::vector<uint32_t> nrcv;   // completions per transfer
    for (auto& x : xs)
        nrcv.push_back((x.prefetch || x.tmop || x.kmu) ? 0 : (x.mc_mask ? __builtin_popcount(x.mc_mask) : 1));

    for (size_t i = 0; i < xs.size(); ++i)
        if (!xs[i].bload && xs[i].bytes == 0 && !xs[i].use_map && !xs[i].tmop && !xs[i].kmu)
            program_desc(uint32_t(i % 16), xs[i]);
    if (g_after_program)
        g_after_program();
    for (int i = 0; i < 4; ++i)
        clock_cycle();

    std::vector<GRsp> rsps;
    size_t next = 0, lasts = 0;
    uint64_t reads = 0, writes = 0, cyc = 0, idle = 0;
    bool ok = true, after_last = false;
    size_t cur = 0;            // the transfer whose writes are expected
    uint32_t cur_lasts = 0;
    bool launch_due = false;
    auto issue_next = [&]() {
        while (next < xs.size() && xs[next].kmu) {
            if (xs[next].in_loop)
                launch_due = true;
            else
                dcr_write(VX_DCR_KMU_STATE_BEGIN, 0);
            ++next;
        }
        while (next < xs.size()) {
            load_xfer(uint32_t(next % 16), xs[next]);
            load_tm(xs[next]);
            tm_store(xs[next]);
            return;
        }
        dut->dxa_req_valid = 0;
    };
    uint64_t map_reads = 0, last_map_read = 0, first_eng_read = ~0ull, map_rsps = 0;
    std::vector<uint64_t> map_pending;   // map lines asked and not answered
    g_dcr_held = g_dcr_taken = 0;
    issue_next();
    while (cyc < 400000) {
        // A random response whose latency has passed, any order.
        int pick = -1;
        std::vector<int> ready;
        for (size_t i = 0; i < rsps.size(); ++i) {
            if (rsps[i].due > cyc)
                continue;
            if (g_map_gate && g_map_gate(map_rsps) && tm_line(rsps[i].line, nullptr))
                continue;
            bool later = false;   // a later line of the same map still unanswered
            if (g_rsp_rev && tm_line(rsps[i].line, nullptr))
                for (auto& o : rsps)
                    later |= (o.line > rsps[i].line) && tm_line(o.line, nullptr);
            if (!later)
                ready.push_back(int(i));
        }
        if (!ready.empty() && (rnd() % 100) < 80)
            pick = ready[rnd() % ready.size()];
        dut->gmem_rsp_valid = (pick >= 0);
        bool map_rsp = false;
        if (pick >= 0) {
            const auto& r = rsps[pick];
            uint8_t mb[256];
            bool is_map = tm_line(r.line, mb);
            map_rsp = is_map;
            for (int w = 0; w < CL / 4; ++w) {
                uint32_t v = 0;
                for (int b = 0; b < 4; ++b)
                    v |= uint32_t(is_map ? mb[w * 4 + b] : gmem_byte(r.line * CL + w * 4 + b)) << (8 * b);
                dut->gmem_rsp_data[w] = v;
            }
            port_set(dut->gmem_rsp_tag, 0, 64, r.tag);
        }
        dut->gmem_req_ready = (int(rnd() % 100) < ready_pct) ? 1 : 0;
        dut->lmem_req_ready = (int(rnd() % 100) < ready_pct) ? 1 : 0;
        uint32_t dcr_addr = 0, dcr_data = 0;
        bool dcr = g_dcr_hook && g_dcr_hook(map_rsps, map_rsp, dcr_addr, dcr_data);
        if (launch_due && !dcr) {
            dcr = true;
            launch_due = false;
            dcr_addr = VX_DCR_KMU_STATE_BEGIN;
        }
        if (dcr) {
            dut->dcr_req_valid = 1;
            dut->dcr_req_rw    = 1;
            dut->dcr_req_addr  = dcr_addr;
            dut->dcr_req_data  = dcr_data;
        }
        dut->eval();
        bool req_taken = dut->dxa_req_valid && dut->dxa_req_ready;
        if (dcr && map_rsp && dcr_addr >= VX_DCR_DXA_DESC_BASE) {
            if (dut->gmem_rsp_ready & 1) ++g_dcr_taken; else ++g_dcr_held;
        }
        if ((dut->gmem_req_valid & 1) && (dut->gmem_req_ready & 1)) {
            uint64_t line = port_bits(dut->gmem_req_addr, 0, MADDR_W - clog2(CL));
            uint64_t tag  = port_bits(dut->gmem_req_tag, 0, 64);
            if (tm_line(line, nullptr)) {
                ++map_reads;
                last_map_read = cyc;
            } else {
                ++reads;
                if (first_eng_read == ~0ull)
                    first_eng_read = cyc;
            }
            rsps.push_back({line, tag, cyc + 2 + rnd() % (max_lat - 1)});
        }
        if (pick >= 0 && (dut->gmem_rsp_ready & 1)) {
            map_rsps += map_rsp;
            rsps.erase(rsps.begin() + pick);
        }
        if ((dut->lmem_req_valid & 1) && (dut->lmem_req_ready & 1)) {
            uint32_t addr = dut->lmem_req_addr;
            if (!(dut->lmem_req_rw & 1)) {
                printf("%s: unexpected local-memory read\n", name);
                ok = false;
            }
            ++writes;
            while (cur < xs.size() && nrcv[cur] == 0) ++cur;   // prefetches write nothing
            if (cur >= xs.size()) {
                printf("%s: write after the last transfer, addr 0x%x\n", name, addr);
                ok = false;
            }
            for (int b = 0; b < WORD_BYTES; ++b) {
                if (port_bit(dut->lmem_req_byteen, b)) {
                    uint32_t w = dut->lmem_req_data[b / 4];
                    mem[addr * WORD_BYTES + b] = uint8_t(w >> (8 * (b % 4)));
                }
            }
            uint64_t attr = dut->lmem_req_attr;
            bool last = field(attr, A_LAST, 1);
            if (after_last && !last) {
                printf("%s: transfer %zu writes after its completion beat\n", name, cur);
                ok = false;
            }
            if (last) {
                if (lasts >= dones.size()) {
                    printf("%s: extra completion beat\n", name);
                    ok = false;
                } else {
                    const auto& d = dones[lasts];
                    if (!field(attr, A_MODE, 1) || field(attr, A_PAY, PAY_W) != d.mbar
                     || field(attr, A_TX, LMEM_LOG + 1) != d.tx || field(attr, A_SLOT, NW_W) != d.slot
                     || field(attr, A_GEN, 4) != d.gen || field(attr, A_REMOTE, 1)) {
                        printf("%s: completion %zu attr: mode=%lu mbar=0x%lx tx=%lu slot=%lu gen=%lu,"
                               " expected mbar=0x%x tx=%u slot=%u gen=%u\n",
                               name, lasts, (unsigned long)field(attr, A_MODE, 1),
                               (unsigned long)field(attr, A_PAY, PAY_W), (unsigned long)field(attr, A_TX, LMEM_LOG + 1),
                               (unsigned long)field(attr, A_SLOT, NW_W), (unsigned long)field(attr, A_GEN, 4),
                               d.mbar, d.tx, d.slot, d.gen);
                        ok = false;
                    }
                }
                ++lasts;
                after_last = true;
                if (++cur_lasts == nrcv[cur]) {
                    ++cur;
                    cur_lasts = 0;
                    after_last = false;
                }
            }
        }
        clock_cycle();
        if (dcr) {
            dut->dcr_req_valid = 0;
            dut->dcr_req_rw    = 0;
        }
        ++cyc;
        if (req_taken) {
            ++next;
            if (serial)
                dut->dxa_req_valid = 0;
            else
                issue_next();
        }
        bool quiet = (next >= xs.size()) && !launch_due && rsps.empty() && !dut->busy && (lasts == dones.size());
        if (serial && !dut->dxa_req_valid && next < xs.size() && rsps.empty() && !dut->busy)
            issue_next();
        idle = quiet ? idle + 1 : 0;
        if (idle > 64)
            break;
    }
    dut->dxa_req_valid = 0;
    dut->gmem_rsp_valid = 0;
    dut->lmem_req_ready = 1;
    dut->gmem_req_ready = 1;
    if (lasts != dones.size()) {
        printf("%s: %zu of %zu completions after %lu cycles (step %zu of %zu issued, %zu responses due, busy %d)\n",
               name, lasts, dones.size(), (unsigned long)cyc, next, xs.size(), rsps.size(), int(dut->busy));
        ok = false;
    }
    if (reads != exp_reads) {
        printf("%s: %lu line reads, expected %lu\n", name, (unsigned long)reads, (unsigned long)exp_reads);
        ok = false;
    }
    g_map_reads = map_reads;
    if (g_maps_first && map_reads && first_eng_read <= last_map_read) {
        printf("%s: an engine read at cycle %lu before the last map read at %lu\n", name,
               (unsigned long)first_eng_read, (unsigned long)last_map_read);
        ok = false;
    }
    int bad = 0;
    for (int i = 0; i < LMEM_BYTES; ++i) {
        if (mem[i] != ref[i]) {
            if (bad < 8)
                printf("%s: byte 0x%x = 0x%02x, expected 0x%02x\n", name, i, mem[i], ref[i]);
            ++bad;
            ok = false;
        }
    }
    if (bad)
        printf("%s: %d bytes differ\n", name, bad);
    if (ok)
        printf("%s: %zu transfers, %lu reads, %lu writes in %lu cycles OK\n", name, xs.size(),
               (unsigned long)reads, (unsigned long)writes, (unsigned long)cyc);
    return ok;
}

static Xfer tile2d(const char* name, uint32_t esz, uint32_t size0, uint32_t size1, uint32_t tile0, uint32_t tile1,
                   int32_t c0, int32_t c1, uint64_t base, uint32_t smem) {
    Xfer x;
    x.name = name;
    x.rank = 2; x.esz = esz; x.base = base;
    x.size[0] = size0; x.size[1] = size1;
    x.stride[0] = ((size0 << esz) + 15) & ~15u;
    x.tile[0] = tile0; x.tile[1] = tile1;
    x.c[0] = c0; x.c[1] = c1;
    x.smem = smem; x.mbar = 0x31;
    return x;
}

static bool run_tensor_all() {
    static_assert(R_END <= int(sizeof(((VVX_dxa_core_top*)nullptr)->dxa_req_data)) * 8,
                  "request layout wider than the port");
    const uint32_t q = LMEM_BYTES / 4;
    const uint64_t G = 0x100000;
    bool ok = true;
    // In bounds, then a box past size0 and a halo at -1.
    ok = run_tensor("tile-inb", {tile2d("inb", 2, 64, 32, 16, 8, 4, 3, G + 0x40, q)}, 100, 8, true) && ok;
    ok = run_tensor("edge", {tile2d("edge", 2, 10, 8, 16, 4, 0, 2, G + 0x40, q)}, 100, 8, true) && ok;
    ok = run_tensor("halo", {tile2d("halo", 2, 16, 16, 8, 4, -1, -1, G, q + 0x100)}, 100, 8, true) && ok;
    {
        Xfer a = tile2d("past", 1, 40, 12, 24, 6, 30, 9, G + 0x13, q);      // both past the end
        Xfer b = tile2d("neg", 0, 37, 9, 50, 5, -20, -3, G + 0x7, q + 0x400); // lead, ib and trail, every row
        b.cfill = 0x7fff7fffu;
        Xfer c = tile2d("allout", 3, 8, 8, 4, 2, 9, 0, G, q + 0x800);         // whole rows filled
        c.cfill = 0x7fffffffu;
        Xfer d = tile2d("oddfill", 0, 37, 5, 21, 4, 3, 3, G + 0x5, q + 0xc13);  // rows past size1, bytes off-line
        d.cfill = 0x44332211u;
        ok = run_tensor("oob-rows-odd", {d}, 70, 20, true) && ok;
        ok = run_tensor("oob-2d", {a, b, c}, 70, 20, true) && ok;
        ok = run_tensor("oob-2d-stream", {a, b, c}, 70, 20) && ok;
    }
    {
        // 5D, both sides of every dim.
        Xfer x;
        x.name = "oob5"; x.rank = 5; x.esz = 1; x.base = G + 0x2000;
        uint32_t sz[5] = {11, 5, 4, 3, 3}, tl[5] = {16, 4, 3, 2, 2};
        int32_t cc[5] = {-3, -1, 2, -1, 2};
        uint32_t st = 0;
        for (int d = 0; d < 5; ++d) { x.size[d] = sz[d]; x.tile[d] = tl[d]; x.c[d] = cc[d]; }
        st = 32;
        for (int d = 0; d < 4; ++d) { x.stride[d] = st; st *= sz[d + 1] + 1; }
        x.smem = q; x.mbar = 0x40; x.cfill = 0x11223344;
        ok = run_tensor("oob5", {x}, 80, 12, true) && ok;
        // Element strides 1..8 in dims 1..4, with fill.
        Xfer y = x;
        y.name = "es"; y.c[0] = 0; y.c[1] = 0; y.c[2] = 1; y.c[3] = 0; y.c[4] = -1;
        uint32_t sz2[5] = {16, 20, 30, 9, 12}, tl2[5] = {16, 7, 5, 8, 6}, es2[5] = {1, 3, 2, 8, 5};
        st = 32;
        for (int d = 0; d < 5; ++d) { y.size[d] = sz2[d]; y.tile[d] = tl2[d]; y.es[d] = es2[d]; }
        for (int d = 0; d < 4; ++d) { y.stride[d] = st; st *= sz2[d + 1]; }
        y.base = G + 0x40000;
        ok = run_tensor("es", {y}, 80, 12, true) && ok;
    }
    // Every swizzle x element size x inner width up to the span, with fill;
    // the destination block's base is not 128-byte aligned.
    {
        std::vector<Xfer> v;
        const uint32_t blk = (WORD_BYTES > 64) ? 0x1080 : 0x1040;
        for (uint32_t swz = 1; swz <= 3; ++swz) {
            uint32_t span = 16u << swz;
            for (uint32_t esz = 0; esz <= 3; ++esz) {
                for (uint32_t inner : {16u, span / 2, span}) {
                    if (inner < 16 || inner > span) continue;
                    uint32_t t0 = inner >> esz;
                    if (t0 == 0) continue;
                    char nm[48];
                    snprintf(nm, sizeof(nm), "swz%u-e%u-w%u", span, 1u << esz, inner);
                    bool oob = ((swz + esz + inner) & 1);
                    Xfer x = tile2d("", esz, oob ? t0 - 1 : 4 * t0, 40, t0, 12, oob ? -1 : int32_t(t0), oob ? -2 : 3,
                                    G + 0x80000 + (v.size() << 12), blk + 0x200);
                    x.name = nm; x.swz = swz; x.blk_base = blk; x.cfill = 0xa5a5a5a5u;
                    v.push_back(x);
                }
            }
        }
        for (auto& x : v)
            ok = run_tensor(x.name.c_str(), {x}, 90, 10, true) && ok;
    }
    // Interleave 16B (stride 1 and 2) and 32B (dim-0 stride 2, 32B swizzle), rank 3, with dim-0 fill.
    {
        Xfer x;
        x.name = "il16"; x.rank = 3; x.esz = 1; x.il = 1; x.base = G + 0xc0000;
        x.size[0] = 8; x.size[1] = 6; x.size[2] = 4;
        x.stride[0] = 16; x.stride[1] = 16 * 6;
        x.tile[0] = 8; x.tile[1] = 4; x.tile[2] = 2; x.c[1] = 1; x.c[2] = 1;
        x.smem = q; x.mbar = 0x12;
        ok = run_tensor("il16", {x}, 90, 10, true) && ok;
        Xfer y = x;
        y.name = "il16-es2"; y.es[0] = 2; y.c[0] = -3; y.size[0] = 7; y.tile[0] = 8;
        ok = run_tensor("il16-es2", {y}, 90, 10, true) && ok;
        Xfer z = x;
        z.name = "il32"; z.il = 2; z.swz = 1; z.es[0] = 2; z.size[0] = 16; z.tile[0] = 16;
        z.stride[0] = 32; z.stride[1] = 32 * 6; z.c[0] = 0;
        ok = run_tensor("il32", {z}, 90, 10, true) && ok;
        // A gather whose rows cross lines, with odd strides.
        Xfer w = tile2d("", 0, 300, 4, 200, 3, -7, 2, G + 0xd0000 + 0x31, q);   // row 3 of 2..4 past size1
        w.rank = 3; w.il = 1; w.size[2] = 2; w.tile[2] = 2; w.stride[1] = w.stride[0] * 4; w.cfill = 0x0f1e2d3cu;
        for (uint32_t es0 : {3u, 5u, 7u}) {
            char nm[32];
            snprintf(nm, sizeof(nm), "gather-es%u", es0);
            w.name = nm; w.es[0] = es0;
            ok = run_tensor(nm, {w}, 90, 10, true) && ok;
        }
    }
    // im2col 3D/4D/5D, pixels wrapping the box, padded corners, offsets, strides.
    {
        Xfer x;
        x.name = "i2c3"; x.rank = 3; x.im2col = 1; x.esz = 1; x.base = G + 0x100000;
        x.size[0] = 12; x.size[1] = 9; x.size[2] = 3;            // C, W, N
        x.stride[0] = 32; x.stride[1] = 32 * 9;
        x.cpp = 16; x.ppc = 20; x.lower[0] = -1; x.upper[0] = 1;
        x.c[0] = -2; x.c[1] = -1; x.c[2] = 0; x.offs[0] = 1;
        x.smem = q; x.mbar = 0x22; x.cfill = 0x5a6b7c8du;
        ok = run_tensor("i2c3", {x}, 90, 12, true) && ok;
        Xfer y;
        y.name = "i2c4"; y.rank = 4; y.im2col = 1; y.esz = 2; y.base = G + 0x140000;
        y.size[0] = 8; y.size[1] = 5; y.size[2] = 4; y.size[3] = 2;   // C, W, H, N
        y.stride[0] = 32; y.stride[1] = 32 * 5; y.stride[2] = 32 * 5 * 4;
        y.cpp = 8; y.ppc = 30; y.lower[0] = -1; y.lower[1] = -1; y.upper[0] = 0; y.upper[1] = 1;
        y.es[1] = 2; y.c[0] = 0; y.c[1] = -1; y.c[2] = -1; y.c[3] = 0; y.offs[0] = 1; y.offs[1] = 0;
        y.smem = q + 0x800; y.mbar = 0x23;
        ok = run_tensor("i2c4", {y}, 90, 12, true) && ok;
        Xfer z;
        z.name = "i2c5"; z.rank = 5; z.im2col = 1; z.esz = 0; z.base = G + 0x180000 + 3;
        z.size[0] = 20; z.size[1] = 4; z.size[2] = 3; z.size[3] = 3; z.size[4] = 2;  // C, W, H, D, N
        z.stride[0] = 24; z.stride[1] = 24 * 4; z.stride[2] = 24 * 12; z.stride[3] = 24 * 36;
        z.cpp = 24; z.ppc = 40; z.lower[0] = -1; z.lower[1] = 0; z.lower[2] = -1;
        z.upper[0] = 1; z.upper[1] = 0; z.upper[2] = 0; z.es[1] = 1; z.es[2] = 2; z.es[3] = 1;
        z.c[0] = 0; z.c[1] = -1; z.c[2] = 0; z.c[3] = -1; z.c[4] = 0; z.offs[0] = 0; z.offs[1] = 1; z.offs[2] = 1;
        z.smem = q + 0x1000; z.mbar = 0x24;
        ok = run_tensor("i2c5", {z}, 90, 12, true) && ok;
        Xfer s = x;
        s.name = "i2c3-swz128"; s.swz = 3; s.cpp = 64; s.size[0] = 70; s.stride[0] = 160; s.stride[1] = 160 * 9;
        s.esz = 1; s.ppc = 12;
        ok = run_tensor("i2c3-swz128", {s}, 90, 12, true) && ok;
    }
    // Global-to-local bulk copies, and prefetches of a box and of a range.
    {
        std::vector<Xfer> v;
        for (uint32_t bytes : {16u, 48u, 208u, 2064u, 4096u}) {
            Xfer x;
            x.name = "bload"; x.bload = true; x.esz = 3; x.bytes = bytes;
            x.base = G + 0x200000 + v.size() * 0x4000 + 16 * v.size();
            x.smem = q + 16 * (v.size() + 1); x.mbar = 0x50 + v.size();
            v.push_back(x);
        }
        ok = run_tensor("bload", v, 70, 16) && ok;
        Xfer t = tile2d("pf", 2, 50, 10, 32, 8, -4, 5, G + 0x240000, 0);
        t.prefetch = true;
        Xfer b;
        b.prefetch = true; b.esz = 3; b.bytes = 1040; b.base = G + 0x260000 + 0x30;
        Xfer l = tile2d("after", 2, 50, 10, 32, 8, -4, 5, G + 0x240000, q);
        ok = run_tensor("prefetch", {t, b}, 80, 10, true) && ok;
        ok = run_tensor("prefetch-then-load", {t, l, b}, 80, 10) && ok;
    }
    // Cluster multicast to ranks 0 and 1 with 128B swizzle and fill.
    {
        Xfer x = tile2d("mc", 2, 30, 20, 32, 6, -2, 17, G + 0x280000, 0x100);
        x.swz = 3; x.mc_mask = 0x3; x.mbar = 0x18; x.blk_base = 0;
        if (MAXC >= 2)
            ok = run_tensor("mc-swz128", {x}, 90, 10, true) && ok;
        x.mc_mask = 0xa; x.c[0] = 3; x.c[1] = -1;
        if (MAXC >= 4)
            ok = run_tensor("mc-0xa-swz128", {x}, 80, 10, true) && ok;
        // A global bulk copy multicast.
        Xfer b;
        b.bload = true; b.esz = 3; b.bytes = 1056; b.base = G + 0x2a0010; b.smem = 0x40; b.mbar = 0x1f8 / 8;
        b.mc_mask = 0x3;
        if (MAXC >= 2)
            ok = run_tensor("bload-mc", {b}, 80, 10, true) && ok;
    }
    // Large sizes, coordinates and offsets: products and bounds past 32 bits.
    {
        Xfer h;
        h.name = "bigsize0"; h.rank = 1; h.esz = 0; h.base = G + 0x300000;
        h.size[0] = 0xFFFFFFF0u; h.tile[0] = 32; h.c[0] = -16;
        h.smem = q + 0x40; h.mbar = 0x33;
        ok = run_tensor("bigsize0", {h}, 90, 10, true) && ok;
        Xfer b = tile2d("bigoff", 2, 64, 0x10000, 16, 2, 0, 0x9000, G, q + 0x80);
        b.stride[0] = 0x10000;
        ok = run_tensor("bigoff", {b}, 90, 10, true) && ok;
        Xfer n = tile2d("bigsize1", 2, 64, 0xFFFFFFFFu, 16, 4, 0, -2, G + 0x10, q + 0x100);
        ok = run_tensor("bigsize1", {n}, 90, 10, true) && ok;
        Xfer w = tile2d("xwrap", 1, 40, 0xF0000000u, 16, 4, -3, 0x7FFFFFFE, G + 0x20, q + 0x200);
        w.rank = 3; w.size[2] = 3; w.tile[2] = 2; w.c[2] = 1; w.stride[1] = 0x40000000u;
        ok = run_tensor("xwrap", {w}, 90, 10, true) && ok;
    }
    // A bulk prefetch of the whole local memory (the largest size the unit passes on).
    {
        Xfer b;
        b.prefetch = true; b.esz = 3; b.bytes = LMEM_BYTES; b.base = G + 0x380000;
        ok = run_tensor("bpf-lmem", {b}, 100, 4, true) && ok;
    }
    // Prefetches with fill: entirely out, rows out, both sides; in streams with loads.
    {
        Xfer pa = tile2d("pf-allout", 2, 20, 10, 16, 4, 40, 2, G + 0x320000, 0);
        pa.prefetch = true;
        Xfer pb = tile2d("pf-rowsout", 2, 20, 10, 16, 4, 0, 12, G + 0x320000, 0);
        pb.prefetch = true;
        Xfer pd = tile2d("pf-lastout", 2, 20, 10, 16, 4, 0, 8, G + 0x320000, 0);  // last rows out
        pd.prefetch = true;
        Xfer pc = tile2d("pf", 2, 50, 10, 32, 8, -4, 5, G + 0x340000, 0);
        pc.prefetch = true;
        Xfer l1 = tile2d("l1", 2, 50, 10, 32, 8, -4, 5, G + 0x340000, q);
        Xfer l2 = tile2d("l2", 2, 20, 10, 16, 4, 40, 2, G + 0x320000, q + 0x1000);
        l2.cfill = 0x12345678;
        Xfer bp;
        bp.prefetch = true; bp.esz = 3; bp.bytes = 528; bp.base = G + 0x360010;
        ok = run_tensor("pf-allout", {pa}, 100, 4, true) && ok;
        ok = run_tensor("pf-lastout", {pd}, 80, 10, true) && ok;
        ok = run_tensor("pf-stream", {pa, pb, pc, l1, pa, l2, pb, bp, pc, pd}, 70, 20) && ok;
        ok = run_tensor("pf-stream2", {pa, pa, pb, l2, l2, bp, bp, pa}, 100, 3) && ok;
    }
    if (MAXC >= 2) {
        Xfer w = tile2d("", 0, 300, 4, 200, 3, -7, 2, G + 0xd0000 + 0x31, 0x100);
        w.rank = 3; w.il = 1; w.size[2] = 2; w.tile[2] = 2; w.stride[1] = w.stride[0] * 4; w.cfill = 0x0f1e2d3cu;
        w.es[0] = 3; w.name = "gather-mc"; w.mc_mask = 0x3; w.blk_base = 0; w.mbar = 0x18;
        ok = run_tensor("gather-mc", {w}, 80, 10, true) && ok;
        Xfer w2 = w; w2.name = "gather-mc-swz"; w2.il = 2; w2.swz = 1; w2.es[0] = 2; w2.tile[0] = 32;
        w2.size[0] = 21; w2.c[0] = -5; w2.stride[0] = 32; w2.stride[1] = 128; w2.smem = 0x400;
        ok = run_tensor("gather-mc-swz", {w2, w}, 70, 20) && ok;
    }
    {
        Xfer y;
        y.name = "i2c4-nout"; y.rank = 4; y.im2col = 1; y.esz = 2; y.base = G + 0x140000;
        y.size[0] = 8; y.size[1] = 5; y.size[2] = 4; y.size[3] = 2;
        y.stride[0] = 32; y.stride[1] = 32 * 5; y.stride[2] = 32 * 5 * 4;
        y.cpp = 8; y.ppc = 30; y.lower[0] = -1; y.lower[1] = -1; y.upper[0] = 0; y.upper[1] = 1;
        y.es[1] = 2; y.es[2] = 3; y.c[0] = 0; y.c[1] = -1; y.c[2] = -1; y.c[3] = 1; y.offs[0] = 2; y.offs[1] = 1;
        y.smem = q + 0x800; y.mbar = 0x23; y.cfill = 0xdeadbeefu;
        ok = run_tensor("i2c4-nout", {y}, 90, 12, true) && ok;
        Xfer y2 = y; y2.name = "i2c4-nall"; y2.c[3] = 5;
        Xfer y3 = y; y3.name = "i2c4-c0"; y3.c[0] = 6; y3.cpp = 4; y3.c[3] = 0; y3.ppc = 60;
        Xfer y4 = y2; y4.name = "i2c4-pf-nall"; y4.prefetch = true; y4.smem = 0;
        ok = run_tensor("i2c-stream", {y, y2, y4, y3, y}, 70, 20) && ok;
    }
    {
        Xfer z;
        z.name = "il32-oob"; z.rank = 3; z.esz = 1; z.il = 2; z.swz = 1; z.es[0] = 2; z.base = G + 0x360000;
        z.size[0] = 13; z.size[1] = 6; z.size[2] = 4; z.stride[0] = 32; z.stride[1] = 32 * 6;
        z.tile[0] = 16; z.tile[1] = 4; z.tile[2] = 3; z.c[0] = -3; z.c[1] = 4; z.c[2] = 2;
        z.smem = q; z.blk_base = q; z.mbar = 0x12; z.cfill = 0x7fff7fffu;
        Xfer z2 = z; z2.name = "il16-swz64"; z2.il = 1; z2.swz = 2; z2.es[0] = 3; z2.c[0] = 5; z2.tile[0] = 8;
        z2.smem = q + 0x400;
        ok = run_tensor("il-swz", {z, z2, z}, 70, 20) && ok;
    }
    // The swizzle moves the last, partial 128-byte row's chunks to its end: a
    // 192-byte box at block offset 0x300 spans 256 bytes, which fit exactly.
    {
        Xfer f = tile2d("swz-fp", 2, 16, 3, 16, 3, 0, 0, G + 0x3a0000, q + 0x300);
        f.swz = 3; f.blk_base = q; f.room = 256;
        ok = run_tensor("swz-fp", {f}, 90, 10, true) && ok;
    }
    return ok;
}

// ---- Stores and reductions (funct3 5) -------------------------------------

#ifdef VX_CFG_EXT_FPAMO_ENABLE
static constexpr int AOW = 5;
#else
static constexpr int AOW = 4;
#endif
#ifdef VX_CFG_EXT_ZACAS_ENABLE
static constexpr int ZCW = VX_CFG_XLEN;
#else
static constexpr int ZCW = 0;
#endif
static constexpr int HART_W   = up1(clog2(VX_CFG_NUM_CLUSTERS * NCC) + clog2(VX_CFG_NUM_WARPS) + clog2(VX_CFG_NUM_THREADS));
static constexpr int AMO_BITS = 1 + AOW + 1 + ZCW + HART_W;
static constexpr int ATTR_FENCE = 4 + AMO_BITS;
// The store fields sit above the tensor-load ones: bg_seq, bg, then dxa_st_t up.
static constexpr int R_BGSEQ    = R_TLOAD + 1;
static constexpr int R_BG       = R_BGSEQ + 3;
static constexpr int R_ST_HART  = R_BG + 1;
static constexpr int R_ST_TSIZE = R_ST_HART + HART_W;
static constexpr int R_ST_W8    = R_ST_TSIZE + 2;
static constexpr int R_ST_UNS   = R_ST_W8 + 1;
static constexpr int R_ST_OP    = R_ST_UNS + 1;
static constexpr int R_ST_KIND  = R_ST_OP + AOW;
static constexpr int R_ST_EN    = R_ST_KIND + 3;
static constexpr int R_SEND     = R_ST_EN + 1;
// dlmem_req_pkt_t from its last field up.
static constexpr int PEND_W   = up1(clog2(VX_CFG_DLMEM_PENDING_SIZE));
static constexpr int P_TAG    = 0;
static constexpr int P_MBADDR = P_TAG + max2(PEND_W, NW_W);
static constexpr int P_MBV    = P_MBADDR + LMEM_LOG - 3;
static constexpr int P_CMP    = P_MBV + 1;
static constexpr int P_HART   = P_CMP + ZCW;
static constexpr int P_UNS    = P_HART + HART_W;
static constexpr int P_AOP    = P_UNS + 1;
static constexpr int P_DATA   = P_AOP + AOW;
static constexpr int P_BYTEEN = P_DATA + 64;
static constexpr int P_ADDR   = P_BYTEEN + 8;
static constexpr int P_OP     = P_ADDR + LMEM_LOG - 3;

// A store-family transfer: x's box and descriptor, or a bulk range (x.bytes
// from x.smem to x.base), as the engine receives it.
struct St {
    Xfer x;
    uint32_t kind = dxa_ref::ST_BULK, op = 0, type = 0;
    uint32_t dcore = 0, dslot = 2, wid = 1, seq = 0;
};

static St st_bulk(uint32_t kind, uint32_t src, uint64_t dst, uint32_t bytes, uint32_t op = 0, uint32_t type = 0) {
    St s;
    s.kind = kind; s.op = op; s.type = type;
    s.x.esz = 3; s.x.bytes = bytes; s.x.base = dst; s.x.smem = src; s.x.mbar = 0;
    return s;
}

static void load_st(uint32_t slot, const St& s) {
    const Xfer& x = s.x;
    for (size_t i = 0; i < sizeof(dut->dxa_req_data) / sizeof(dut->dxa_req_data[0]); ++i)
        dut->dxa_req_data[i] = 0;
    bool tensor = dxa_ref::is_tensor(s.kind);
    if (tensor) {
        set_bits(dut->dxa_req_data, R_META, 32, 0xC0000000u | slot);
        for (int i = 0; i < 5; ++i)
            set_bits(dut->dxa_req_data, R_COORDS + 32 * i, 32, uint32_t(x.c[i]));
    } else {
        set_bits(dut->dxa_req_data, R_META, 32, 0xC0000000u);
        set_bits(dut->dxa_req_data, R_SIZE, LMEM_LOG + 1, x.bytes);
        set_bits(dut->dxa_req_data, R_GSRC, MADDR_W, x.base);
    }
    set_bits(dut->dxa_req_data, R_SMEM, LMEM_LOG, x.smem);
    set_bits(dut->dxa_req_data, R_BLK, LMEM_LOG, x.blk_base);
    set_bits(dut->dxa_req_data, R_ROOM, LMEM_LOG + 1, x.room ? x.room : LMEM_BYTES - x.smem);
    set_bits(dut->dxa_req_data, R_WID, NW_W, s.wid);
    set_bits(dut->dxa_req_data, R_MBAR, LMEM_LOG - 3, x.mbar);
    set_bits(dut->dxa_req_data, R_SLOT, NW_W, s.dslot);
    set_bits(dut->dxa_req_data, R_DCORE, CORE_W, s.dcore);
    bool bg = dxa_ref::is_bulk_group(s.kind);
    set_bits(dut->dxa_req_data, R_BG, 1, bg);
    set_bits(dut->dxa_req_data, R_BGSEQ, 3, s.seq);
    set_bits(dut->dxa_req_data, R_ST_EN, 1, 1);
    set_bits(dut->dxa_req_data, R_ST_KIND, 3, s.kind);
    if (dxa_ref::is_reduce(s.kind)) {
        auto a = dxa_ref::red_amo(s.op, s.type);
        uint32_t tb = dxa_ref::type_bytes(s.type);
        set_bits(dut->dxa_req_data, R_ST_OP, AOW, a.amo_op);
        set_bits(dut->dxa_req_data, R_ST_UNS, 1, a.is_unsigned);
        set_bits(dut->dxa_req_data, R_ST_W8, 1, tb == 8);
        set_bits(dut->dxa_req_data, R_ST_TSIZE, 2, tb == 8 ? 3 : (tb == 2 ? 1 : 2));
        set_bits(dut->dxa_req_data, R_ST_HART, HART_W, uint64_t(s.wid) << clog2(VX_CFG_NUM_THREADS));
    }
    dut->dxa_req_valid = 1;
}

// The last-level cache's atomic on a 4- or 8-byte word.
static uint64_t llc_amo(uint32_t op, bool uns, uint32_t width, uint64_t old, uint64_t rhs) {
    const uint64_t m = width == 8 ? ~0ull : 0xffffffffull;
    old &= m; rhs &= m;
    const int64_t so = width == 8 ? int64_t(old) : int64_t(int32_t(uint32_t(old)));
    const int64_t sr = width == 8 ? int64_t(rhs) : int64_t(int32_t(uint32_t(rhs)));
    switch (op) {
    case 0x02: return (old + rhs) & m;
    case 0x04: return old ^ rhs;
    case 0x05: return old | rhs;
    case 0x06: return old & rhs;
    case 0x07: return uns ? (old < rhs ? old : rhs) : (so < sr ? old : rhs);
    case 0x08: return uns ? (old > rhs ? old : rhs) : (so > sr ? old : rhs);
    default:   return amo_ref::apply(op, width == 8 ? amo_ref::W_D : amo_ref::W_W, old, rhs, amo_ref::SITE_LLC) & m;
    }
}

// Runs the transfers back to back and checks global memory byte for byte
// (with the LLC atomics), a cluster reduction's destination block, the
// request counts, and every bulk_group operation's completion events.
static bool run_store(const char* name, std::vector<St> ss, int ready_pct, int max_lat) {
    if (const char* only = std::getenv("DXA_ONLY")) {
        if (!strstr(name, only))
            return true;
    }
    rng_state = 1;
    for (const char* c = name; *c; ++c)
        rng_state = rng_state * 131 + uint8_t(*c);
    std::vector<uint8_t> lmem(LMEM_BYTES), dmem(LMEM_BYTES), dexp(LMEM_BYTES);
    for (int i = 0; i < LMEM_BYTES; ++i) {
        lmem[i] = pattern(i);
        dmem[i] = dexp[i] = pattern(i * 7 + 3);
    }
    std::map<uint64_t, uint8_t> gmem, gexp;
    auto gget = [](std::map<uint64_t, uint8_t>& m, uint64_t a) -> uint8_t& {
        auto it = m.find(a);
        if (it == m.end())
            it = m.emplace(a, gmem_byte(a)).first;
        return it->second;
    };
    uint64_t exp_wr = 0, exp_amo = 0, exp_pkt = 0, exp_rd = 0, st_lines = 0, st_fences = 0;
    for (auto& s : ss) {
        uint64_t wr0 = exp_wr;
        Xfer& x = s.x;
        bool tensor = dxa_ref::is_tensor(s.kind);
        auto d = ref_desc(x);
        dxa_ref::Req q;
        for (int i = 0; i < 5; ++i) q.coord[i] = x.c[i];
        if (tensor) {
            const char* bad = dxa_ref::check_tensor_store(d, q, x.smem - x.blk_base);
            if (bad) {
                printf("%s: vector %s breaks a rule: %s\n", name, x.name.c_str(), bad);
                return false;
            }
        }
        // Tokens: the in-bounds elements of a row grouped by global line;
        // each reads its local-memory words (one per element if gathered).
        uint32_t e = 1u << x.esz, cnt = 0;
        uint32_t per_row = x.bytes ? x.bytes / e : (x.im2col ? x.cpp : box_n(x, 0));
        bool gath = tensor && !x.im2col && x.es[0] != 1;
        int64_t tok_line = -1;
        uint32_t tok_s = 0, tok_end = 0, tok_n = 0;
        uint32_t src0 = tensor ? x.smem - x.blk_base : x.smem;
        auto flush = [&]() {
            if (tok_line < 0) return;
            ++exp_wr;
            if (gath) {
                exp_rd += tok_n;
            } else if (x.swz) {
                std::set<uint32_t> words;
                for (uint32_t b = tok_s; b < tok_end; ++b)
                    words.insert((x.blk_base + dxa_ref::swizzle(b, x.swz)) / WORD_BYTES);
                exp_rd += words.size();
            } else {
                exp_rd += (x.blk_base + tok_end - 1) / WORD_BYTES - (x.blk_base + tok_s) / WORD_BYTES + 1;
            }
            tok_line = -1;
        };
        walk(x, [&](uint32_t off, bool in, uint64_t a) {
            if (in) {
                int64_t ln = int64_t(a / CL);
                if (ln != tok_line) {
                    flush();
                    tok_line = ln; tok_s = src0 + off; tok_n = 0;
                }
                tok_end = src0 + off + e;
                ++tok_n;
            } else {
                flush();
            }
            if (++cnt == per_row) {
                flush();
                cnt = 0;
            }
        });
        flush();
        // The golden image.
        const uint8_t* blk = lmem.data() + x.blk_base;
        if (s.kind == dxa_ref::ST_TENSOR) {
            dxa_ref::for_each_store_byte(d, q, src0, [&](uint32_t r, uint64_t ga) { gget(gexp, ga) = blk[r]; });
        } else if (s.kind == dxa_ref::ST_BULK) {
            for (uint32_t i = 0; i < x.bytes; ++i)
                gget(gexp, x.base + i) = lmem[x.smem + i];
        } else if (s.kind == dxa_ref::ST_RED_CLUSTER) {
            dxa_ref::reduce_bytes(s.op, s.type, dexp.data() + x.base, lmem.data() + x.smem, x.bytes);
            exp_pkt += x.bytes / dxa_ref::type_bytes(s.type);
        } else {
            // Global reductions: the golden per element, each element's bytes
            // contiguous in the map; one request per word holding one.
            uint32_t E = dxa_ref::type_bytes(s.type), W = dxa_ref::red_word_bytes(s.type);
            std::set<uint64_t> words;
            auto red_elem = [&](uint64_t ga, const uint8_t* sp) {
                uint8_t g[8];
                for (uint32_t b = 0; b < E; ++b) g[b] = gget(gexp, ga + b);
                uint64_t r = dxa_ref::reduce_elem(s.op, s.type, dxa_ref::load_le(g, E), dxa_ref::load_le(sp, E));
                for (uint32_t b = 0; b < E; ++b) gget(gexp, ga + b) = uint8_t(r >> (8 * b));
                words.insert(ga / W);
            };
            if (s.kind == dxa_ref::ST_RED_GLOBAL) {
                for (uint32_t i = 0; i < x.bytes; i += E)
                    red_elem(x.base + i, lmem.data() + x.smem + i);
            } else {
                dxa_ref::for_each_element(d, q, [&](uint32_t off, bool in, uint64_t ga) {
                    if (!in) return;
                    uint8_t sb[8];
                    for (uint32_t b = 0; b < E; ++b) sb[b] = blk[dxa_ref::swizzle(src0 + off + b, x.swz)];
                    red_elem(ga, sb);
                });
            }
            exp_amo += words.size();
        }
        if (s.kind == dxa_ref::ST_BULK || s.kind == dxa_ref::ST_TENSOR) {
            st_lines += exp_wr - wr0;
            if (exp_wr != wr0) ++st_fences;
        }
    }
    for (size_t i = 0; i < ss.size(); ++i)
        if (dxa_ref::is_tensor(ss[i].kind))
            program_desc(uint32_t(i % 16), ss[i].x);
    for (int i = 0; i < 4; ++i)
        clock_cycle();

    struct Rsp { uint64_t tag, due; };
    std::vector<Rsp> grsps;
    std::deque<RdRsp> lrsps;
    size_t next = 0;
    uint64_t lreads = 0, gwrites = 0, gamos = 0, gfences = 0, pkts = 0, cyc = 0, idle = 0;
    uint64_t last_perf = 0;       // last fence/AMO response given
    std::vector<std::pair<uint32_t, uint32_t>> evts;   // (bits, cycle)
    size_t bg_ops = 0;
    for (auto& s : ss) bg_ops += dxa_ref::is_bulk_group(s.kind);
    const St* cl = nullptr;
    for (auto& s : ss)
        if (s.kind == dxa_ref::ST_RED_CLUSTER && !cl) cl = &s;
    bool ok = true;
    auto issue_next = [&]() {
        if (next < ss.size()) load_st(uint32_t(next % 16), ss[next]);
        else dut->dxa_req_valid = 0;
    };
    issue_next();
    while (cyc < 400000) {
        int pick = -1;
        std::vector<int> ready;
        for (size_t i = 0; i < grsps.size(); ++i)
            if (grsps[i].due <= cyc) ready.push_back(int(i));
        if (!ready.empty() && (rnd() % 100) < 80)
            pick = ready[rnd() % ready.size()];
        dut->gmem_rsp_valid = (pick >= 0);
        if (pick >= 0)
            port_set(dut->gmem_rsp_tag, 0, 64, grsps[pick].tag);
        bool lrsp = !lrsps.empty() && lrsps.front().due <= cyc;
        dut->lmem_rsp_valid = lrsp;
        if (lrsp) {
            const auto& r = lrsps.front();
            for (int w = 0; w < WORD_BYTES / 4; ++w) {
                uint32_t v = 0;
                for (int b = 0; b < 4; ++b)
                    v |= uint32_t(lmem[r.addr * WORD_BYTES + w * 4 + b]) << (8 * b);
                dut->lmem_rsp_data[w] = v;
            }
            dut->lmem_rsp_tag = r.tag;
        }
        dut->gmem_req_ready = (int(rnd() % 100) < ready_pct) ? 1 : 0;
        dut->lmem_req_ready = (int(rnd() % 100) < ready_pct) ? 1 : 0;
        dut->row_out_ready  = (int(rnd() % 100) < ready_pct) ? 1 : 0;
        dut->eval();
        bool req_taken = dut->dxa_req_valid && dut->dxa_req_ready;
        if (lrsp && !(dut->lmem_rsp_ready & 1)) {
            printf("%s: read response refused\n", name);
            ok = false;
        }
        if ((dut->lmem_req_valid & 1) && (dut->lmem_req_ready & 1)) {
            if (dut->lmem_req_rw & 1) {
                printf("%s: unexpected local-memory write\n", name);
                ok = false;
            } else {
                ++lreads;
                uint64_t prev = lrsps.empty() ? cyc : lrsps.back().due;
                uint64_t due = cyc + 1 + rnd() % 4;
                lrsps.push_back({uint32_t(dut->lmem_req_addr), (uint64_t)dut->lmem_req_tag, due > prev ? due : prev});
            }
        }
        if (lrsp)
            lrsps.pop_front();
        if ((dut->gmem_req_valid & 1) && (dut->gmem_req_ready & 1)) {
            uint64_t line = port_bits(dut->gmem_req_addr, 0, MADDR_W - clog2(CL));
            uint64_t tag  = port_bits(dut->gmem_req_tag, 0, 64);
            bool flush = port_bits(dut->gmem_req_attr, 0, 1);
            bool fence = port_bits(dut->gmem_req_attr, ATTR_FENCE, 1);
            bool amo   = port_bits(dut->gmem_req_attr, 4, 1);
            std::vector<int> be;
            for (int b = 0; b < CL; ++b)
                if (port_bit(dut->gmem_req_byteen, b)) be.push_back(b);
            auto dbyte = [&](int b) { return uint8_t(dut->gmem_req_data[b / 4] >> (8 * (b % 4))); };
            if (dut->gmem_req_rw & 1) {
                ++gwrites;
                if (flush || amo) { printf("%s: a write with fence/atomic attributes\n", name); ok = false; }
                for (int b : be) gget(gmem, line * CL + b) = dbyte(b);
            } else if (flush || fence) {
                ++gfences;
                if (!(flush && fence)) { printf("%s: fence without is_flush and is_fence\n", name); ok = false; }
                grsps.push_back({tag, cyc + 2 + rnd() % (max_lat - 1)});
            } else if (amo) {
                ++gamos;
                uint32_t op  = uint32_t(port_bits(dut->gmem_req_attr, 5, AOW));
                bool uns     = port_bits(dut->gmem_req_attr, 5 + AOW, 1);
                uint32_t w   = uint32_t(be.size());
                bool half_ok = (w == 2) && (op >= 0x14 && op <= 0x19);
                if ((w != 4 && w != 8 && !half_ok) || (be[0] % w) || be.back() != be[0] + int(w) - 1) {
                    printf("%s: atomic with byte enables of %u bytes at %d\n", name, w, be.empty() ? -1 : be[0]);
                    ok = false;
                } else {
                    uint64_t a = line * CL + be[0], old = 0, rhs = 0;
                    for (uint32_t b = 0; b < w; ++b) {
                        old |= uint64_t(gget(gmem, a + b)) << (8 * b);
                        rhs |= uint64_t(dbyte(be[0] + b)) << (8 * b);
                    }
                    // A 16-bit pair's half: the packed op on its lane only.
                    uint64_t r = (w == 2) ? (llc_amo(op, uns, 4, old, rhs) & 0xffff) : llc_amo(op, uns, w, old, rhs);
                    for (uint32_t b = 0; b < w; ++b) gget(gmem, a + b) = uint8_t(r >> (8 * b));
                }
                grsps.push_back({tag, cyc + 2 + rnd() % (max_lat - 1)});
            } else {
                printf("%s: unexpected global read\n", name);
                ok = false;
            }
        }
        if (pick >= 0 && (dut->gmem_rsp_ready & 1)) {
            grsps.erase(grsps.begin() + pick);
            last_perf = cyc;
        }
        if (dut->row_out_valid && dut->row_out_ready) {
            ++pkts;
            auto pb = [&](int lsb, int w) { return port_bits(dut->row_out_data, lsb, w); };
            uint32_t be = uint32_t(pb(P_BYTEEN, 8)), op = uint32_t(pb(P_AOP, AOW));
            uint64_t addr = pb(P_ADDR, LMEM_LOG - 3) * 8, data = pb(P_DATA, 64);
            if (pb(P_OP, 2) != 3 || !pb(P_MBV, 1) || !cl || pb(P_MBADDR, LMEM_LOG - 3) != cl->x.mbar
             || uint32_t(dut->row_out_dst) != cl->dcore) {
                printf("%s: packet op %lu mbar_valid %lu mbar 0x%lx dst %u\n", name, (unsigned long)pb(P_OP, 2),
                       (unsigned long)pb(P_MBV, 1), (unsigned long)pb(P_MBADDR, LMEM_LOG - 3), uint32_t(dut->row_out_dst));
                ok = false;
            }
            uint32_t w = __builtin_popcount(be), lo = __builtin_ctz(be ? be : 1);
            if ((w != 4 && w != 8) || (lo % w)) { printf("%s: packet byteen 0x%x\n", name, be); ok = false; }
            else {
                uint64_t old = 0;
                for (uint32_t b = 0; b < w; ++b) old |= uint64_t(dmem[addr + lo + b]) << (8 * b);
                uint64_t r = llc_amo(op, pb(P_UNS, 1), w, old, data >> (8 * lo));
                for (uint32_t b = 0; b < w; ++b) dmem[addr + lo + b] = uint8_t(r >> (8 * b));
            }
        }
        if (dut->bg_compl_valid & 1)
            evts.push_back({uint32_t(dut->bg_compl), uint32_t(cyc)});
        clock_cycle();
        ++cyc;
        if (req_taken) { ++next; issue_next(); }
        // The engine's busy covers its setup only: wait for every expected request and event.
        size_t wr_evts = 0;
        for (auto& e : evts) wr_evts += e.first & 1;
        bool quiet = (next >= ss.size()) && grsps.empty() && lrsps.empty() && !dut->busy
                  && pkts >= exp_pkt && gwrites >= st_lines && gfences >= st_fences && gamos >= exp_amo
                  && wr_evts >= bg_ops;
        idle = quiet ? idle + 1 : 0;
        if (idle > 64) break;
    }
    dut->dxa_req_valid = 0; dut->gmem_rsp_valid = 0; dut->lmem_rsp_valid = 0;
    dut->lmem_req_ready = 1; dut->gmem_req_ready = 1; dut->row_out_ready = 1;

    // Requests: lines written by stores, words by global reductions, elements by cluster ones.
    if (gwrites != st_lines) { printf("%s: %lu line writes, expected %lu\n", name, (unsigned long)gwrites, (unsigned long)st_lines); ok = false; }
    if (gfences != st_fences) { printf("%s: %lu fences, expected %lu\n", name, (unsigned long)gfences, (unsigned long)st_fences); ok = false; }
    if (gamos != exp_amo) { printf("%s: %lu atomics, expected %lu\n", name, (unsigned long)gamos, (unsigned long)exp_amo); ok = false; }
    if (pkts != exp_pkt) { printf("%s: %lu packets, expected %lu\n", name, (unsigned long)pkts, (unsigned long)exp_pkt); ok = false; }
    if (lreads != exp_rd) { printf("%s: %lu local-memory reads, expected %lu\n", name, (unsigned long)lreads, (unsigned long)exp_rd); ok = false; }
    // Completion events: one read_done and one write_done per bulk_group op, in order.
    {
        size_t k = 0;
        for (auto& s : ss) {
            if (!dxa_ref::is_bulk_group(s.kind)) continue;
            bool rd = false, wr = false;
            while (k < evts.size() && !(rd && wr)) {
                uint32_t v = evts[k].first;
                uint32_t wid = v >> 5, seq = (v >> 2) & 7;
                if (wid != s.wid || seq != s.seq) {
                    printf("%s: event wid %u seq %u, expected wid %u seq %u\n", name, wid, seq, s.wid, s.seq);
                    ok = false;
                }
                if (((v >> 1) & 1) && rd) { printf("%s: second read_done\n", name); ok = false; }
                if ((v & 1) && wr) { printf("%s: second write_done\n", name); ok = false; }
                if ((v & 1) && !rd && !((v >> 1) & 1)) { printf("%s: write_done before read_done\n", name); ok = false; }
                rd |= (v >> 1) & 1; wr |= v & 1;
                ++k;
            }
            if (!rd || !wr) { printf("%s: op of seq %u: read_done %d write_done %d\n", name, s.seq, rd, wr); ok = false; }
        }
        for (; k < evts.size(); ++k) {
            printf("%s: extra event 0x%x\n", name, evts[k].first);
            ok = false;
        }
        // The last write_done comes after the last fence or atomic response.
        for (auto it = evts.rbegin(); it != evts.rend(); ++it) {
            if (it->first & 1) {
                if ((gfences || gamos) && it->second <= last_perf) {
                    printf("%s: write_done at %u, not after the last response at %lu\n", name, it->second, (unsigned long)last_perf);
                    ok = false;
                }
                break;
            }
        }
    }
    int bad = 0;
    std::set<uint64_t> keys;
    for (auto& kv : gmem) keys.insert(kv.first);
    for (auto& kv : gexp) keys.insert(kv.first);
    for (uint64_t a : keys) {
        uint8_t v = gget(gmem, a), w = gget(gexp, a);
        if (v != w) {
            if (bad < 8) printf("%s: global byte 0x%lx = 0x%02x, expected 0x%02x\n", name, (unsigned long)a, v, w);
            ++bad;
        }
    }
    for (int i = 0; i < LMEM_BYTES; ++i) {
        if (dmem[i] != dexp[i]) {
            if (bad < 8) printf("%s: block byte 0x%x = 0x%02x, expected 0x%02x\n", name, i, dmem[i], dexp[i]);
            ++bad;
        }
    }
    if (bad) { printf("%s: %d bytes differ\n", name, bad); ok = false; }
    if (ok)
        printf("%s: %zu transfers, %lu reads, %lu writes, %lu atomics, %lu packets in %lu cycles OK\n", name, ss.size(),
               (unsigned long)lreads, (unsigned long)gwrites, (unsigned long)gamos, (unsigned long)pkts, (unsigned long)cyc);
    return ok;
}

static St st_tensor(const Xfer& x, uint32_t kind = dxa_ref::ST_TENSOR, uint32_t op = 0, uint32_t type = 0) {
    St s;
    s.x = x; s.kind = kind; s.op = op; s.type = type;
    return s;
}

static bool run_store_all() {
    static_assert(R_SEND <= int(sizeof(((VVX_dxa_core_top*)nullptr)->dxa_req_data)) * 8,
                  "request layout wider than the port");
    const uint32_t q = LMEM_BYTES / 4;
    const uint64_t G = 0x400000;
    bool ok = true;
    // Bulk stores, aligned and not to a line, 16 B .. 4 KB.
    {
        std::vector<St> v;
        uint32_t seq = 0;
        for (uint32_t bytes : {16u, 48u, 208u, 2064u, 4096u}) {
            St s = st_bulk(dxa_ref::ST_BULK, q + 16 * (v.size() + 1), G + v.size() * 0x4000 + 16 * v.size(), bytes);
            s.seq = seq++ & 7;
            v.push_back(s);
        }
        ok = run_store("bst", {v[0]}, 100, 8) && ok;
        ok = run_store("bst-stream", v, 70, 16) && ok;
    }
    // Tensor stores: in bounds, past the end, a halo (rows out), 5D, strides.
    {
        auto T = [&](const char* nm, uint32_t esz, uint32_t s0, uint32_t s1, uint32_t t0, uint32_t t1, int32_t c0, int32_t c1,
                     uint64_t base, uint32_t smem) {
            return st_tensor(tile2d(nm, esz, s0, s1, t0, t1, c0, c1, base, smem));
        };
        ok = run_store("tst-inb", {T("inb", 2, 64, 32, 16, 8, 4, 3, G + 0x40, q)}, 100, 8) && ok;
        ok = run_store("tst-edge", {T("edge", 2, 10, 8, 16, 4, 0, 6, G + 0x40, q)}, 90, 8) && ok;
        ok = run_store("tst-past", {T("past", 1, 40, 12, 24, 6, 30, 9, G + 0x13, q), T("odd", 0, 37, 5, 21, 4, 3, 3, G + 0x5, q + 0xc13)}, 70, 20) && ok;
        Xfer x;
        x.name = "oob5"; x.rank = 5; x.esz = 1; x.base = G + 0x2000;
        uint32_t sz[5] = {11, 5, 4, 3, 3}, tl[5] = {16, 4, 3, 2, 2};
        int32_t cc[5] = {3, 1, 2, 1, 2};
        for (int d = 0; d < 5; ++d) { x.size[d] = sz[d]; x.tile[d] = tl[d]; x.c[d] = cc[d]; }
        uint32_t st = 32;
        for (int d = 0; d < 4; ++d) { x.stride[d] = st; st *= sz[d + 1] + 1; }
        x.smem = q;
        ok = run_store("tst-5d", {st_tensor(x)}, 80, 12) && ok;
        Xfer y = x;
        y.name = "es"; y.c[0] = 0; y.c[1] = 0; y.c[2] = 1; y.c[3] = 0; y.c[4] = 0;
        uint32_t sz2[5] = {16, 20, 30, 9, 12}, tl2[5] = {16, 7, 5, 8, 6}, es2[5] = {1, 3, 2, 8, 5};
        st = 32;
        for (int d = 0; d < 5; ++d) { y.size[d] = sz2[d]; y.tile[d] = tl2[d]; y.es[d] = es2[d]; }
        for (int d = 0; d < 4; ++d) { y.stride[d] = st; st *= sz2[d + 1]; }
        y.base = G + 0x40000;
        ok = run_store("tst-es", {st_tensor(y)}, 80, 12) && ok;
    }
    // Every swizzle x element size, an out-of-bounds edge on odd vectors.
    {
        const uint32_t blk = (WORD_BYTES > 64) ? 0x1080 : 0x1040;
        int n = 0;
        for (uint32_t swz = 1; swz <= 3; ++swz) {
            uint32_t span = 16u << swz;
            for (uint32_t esz = 0; esz <= 3; ++esz) {
                uint32_t t0 = span >> esz;
                bool oob = (swz + esz) & 1;
                char nm[48];
                snprintf(nm, sizeof(nm), "tst-swz%u-e%u", span, 1u << esz);
                Xfer x = tile2d(nm, esz, oob ? t0 - 1 : 4 * t0, 40, t0, 12, oob ? 1 : int32_t(t0), oob ? 32 : 3,
                                G + 0x80000 + (uint64_t(n++) << 12), blk + 0x200);
                x.swz = swz; x.blk_base = blk;
                ok = run_store(nm, {st_tensor(x)}, 90, 10) && ok;
            }
        }
    }
    // Interleave and dim-0 strides (gathered tokens), im2col.
    {
        Xfer x;
        x.name = "il16"; x.rank = 3; x.esz = 1; x.il = 1; x.base = G + 0xc0000;
        x.size[0] = 8; x.size[1] = 6; x.size[2] = 4;
        x.stride[0] = 16; x.stride[1] = 16 * 6;
        x.tile[0] = 8; x.tile[1] = 4; x.tile[2] = 2; x.c[1] = 1; x.c[2] = 1;
        x.smem = q;
        ok = run_store("tst-il16", {st_tensor(x)}, 90, 10) && ok;
        Xfer y = x;
        y.name = "il16-es2"; y.es[0] = 2; y.c[0] = 3; y.size[0] = 7; y.tile[0] = 8;
        ok = run_store("tst-il16-es2", {st_tensor(y)}, 90, 10) && ok;
        Xfer z = x;
        z.name = "il32"; z.il = 2; z.swz = 1; z.es[0] = 2; z.size[0] = 16; z.tile[0] = 16;
        z.stride[0] = 32; z.stride[1] = 32 * 6; z.c[0] = 0;
        ok = run_store("tst-il32", {st_tensor(z)}, 90, 10) && ok;
        Xfer w = tile2d("", 0, 300, 4, 200, 3, 7, 2, G + 0xd0000 + 0x31, q);
        w.rank = 3; w.il = 1; w.size[2] = 2; w.tile[2] = 2; w.stride[1] = w.stride[0] * 4;
        for (uint32_t es0 : {3u, 5u, 7u}) {
            char nm[32];
            snprintf(nm, sizeof(nm), "tst-gather-es%u", es0);
            w.name = nm; w.es[0] = es0;
            ok = run_store(nm, {st_tensor(w)}, 90, 10) && ok;
        }
        Xfer i3;
        i3.name = "i2c3"; i3.rank = 3; i3.im2col = 1; i3.esz = 1; i3.base = G + 0x100000;
        i3.size[0] = 12; i3.size[1] = 9; i3.size[2] = 3;
        i3.stride[0] = 32; i3.stride[1] = 32 * 9;
        i3.cpp = 16; i3.ppc = 20; i3.lower[0] = -1; i3.upper[0] = 1;
        i3.c[0] = -2; i3.c[1] = -1; i3.c[2] = 0;
        i3.smem = q;
        ok = run_store("tst-i2c3", {st_tensor(i3)}, 90, 12) && ok;
        Xfer i5;
        i5.name = "i2c5"; i5.rank = 5; i5.im2col = 1; i5.esz = 0; i5.base = G + 0x180000 + 3;
        i5.size[0] = 20; i5.size[1] = 4; i5.size[2] = 3; i5.size[3] = 3; i5.size[4] = 2;
        i5.stride[0] = 24; i5.stride[1] = 24 * 4; i5.stride[2] = 24 * 12; i5.stride[3] = 24 * 36;
        i5.cpp = 24; i5.ppc = 40; i5.lower[0] = -1; i5.lower[1] = 0; i5.lower[2] = -1;
        i5.upper[0] = 1; i5.upper[1] = 0; i5.upper[2] = 0;
        i5.c[0] = 0; i5.c[1] = -1; i5.c[2] = 0; i5.c[3] = -1; i5.c[4] = 0;
        i5.smem = q + 0x1000;
        ok = run_store("tst-i2c5", {st_tensor(i5)}, 90, 12) && ok;
    }
#ifdef VX_CFG_EXT_FPAMO_ENABLE
    const bool fp = true;
#else
    const bool fp = false;
#endif
    // Every op x type of a global bulk reduction, two shapes each.
    {
        int n = 0;
        for (uint32_t op = 0; op < 8; ++op) {
            for (uint32_t ty = 0; ty < 8; ++ty) {
                if (!dxa_ref::red_legal(dxa_ref::ST_RED_GLOBAL, op, ty)) continue;
                if (dxa_ref::red_needs_fpamo(op, ty) && !fp) continue;
                char nm[48];
                snprintf(nm, sizeof(nm), "rg-%s-%s", dxa_ref::red_op_name(op), dxa_ref::red_type_name(ty, op));
                St a = st_bulk(dxa_ref::ST_RED_GLOBAL, q + 0x30, G + 0x200000 + n * 0x1000 + 0x10, 208, op, ty);
                St b = st_bulk(dxa_ref::ST_RED_GLOBAL, q + 0x400, G + 0x200000 + n * 0x1000 + 0x800, 64, op, ty);
                b.seq = 1;
                ok = run_store(nm, {a, b}, 80, 12) && ok;
                ++n;
            }
        }
    }
    // Tensor reductions with out-of-bounds edges and odd starts (16-bit pairs split).
    {
        struct { uint32_t op, ty, esz; } cs[] = {{0, 0, 2}, {1, 1, 2}, {2, 3, 3}, {5, 2, 3}, {0, 4, 2}, {0, 5, 3},
                                                  {0, 6, 1}, {1, 7, 1}, {2, 6, 1}, {3, 0, 2}};
        int n = 0;
        for (auto& c : cs) {
            if (dxa_ref::red_needs_fpamo(c.op, c.ty) && !fp) continue;
            char nm[48];
            snprintf(nm, sizeof(nm), "rt-%s-%s", dxa_ref::red_op_name(c.op), dxa_ref::red_type_name(c.ty, c.op));
            Xfer x = tile2d(nm, c.esz, 37, 9, 16, 4, 3 + (n & 1), 6, G + 0x300000 + (uint64_t(n) << 13), q + 0x100);
            if (c.esz == 1 && (n & 2)) { x.swz = 2; x.blk_base = q; x.tile[0] = 32; x.size[0] = 70; x.c[0] = 41; }
            ok = run_store(nm, {st_tensor(x, dxa_ref::ST_RED_TENSOR, c.op, c.ty)}, 80, 12) && ok;
            ++n;
        }
    }
    // Every op x type of a cluster reduction, into a block of another core.
    {
        for (uint32_t op = 0; op < 8; ++op) {
            for (uint32_t ty = 0; ty < 8; ++ty) {
                if (!dxa_ref::red_legal(dxa_ref::ST_RED_CLUSTER, op, ty)) continue;
                if (dxa_ref::red_needs_fpamo(op, ty) && !fp) continue;
                char nm[48];
                snprintf(nm, sizeof(nm), "rc-%s-%s", dxa_ref::red_op_name(op), dxa_ref::red_type_name(ty, op));
                St a = st_bulk(dxa_ref::ST_RED_CLUSTER, q + 0x40, 2 * q + 0x90, 208, op, ty);
                a.dcore = (NCC > 1) ? 1 : 0; a.x.mbar = (2 * q + 0x8) / 8;
                ok = run_store(nm, {a}, 80, 10) && ok;
            }
        }
    }
    // A stream of every kind, then loads after stores.
    {
        St a = st_bulk(dxa_ref::ST_BULK, q + 0x20, G + 0x500010, 336);
        St b = st_tensor(tile2d("m", 2, 30, 20, 32, 6, 2, 17, G + 0x510000, q + 0x400));
        b.seq = 1;
        St c = st_bulk(dxa_ref::ST_RED_CLUSTER, q + 0x800, 2 * q + 0x40, 64, 0, 0);
        c.x.mbar = (2 * q) / 8;
        St d = st_bulk(dxa_ref::ST_RED_GLOBAL, q + 0x900, G + 0x520020, 96, 0, 2);
        d.seq = 2;
        ok = run_store("st-stream", {a, b, c, d, a}, 60, 20) && ok;
        Xfer l = tile2d("after", 2, 50, 10, 32, 8, -4, 5, G + 0x240000, q);
        ok = run_tensor("load-after-store", {l}, 80, 10, true) && ok;
    }
    return ok;
}

// ---- Tensor maps in global memory ----------------------------------------

// dxa_tm_req_t, above the store fields: rop, rt, addr, op, en.
static constexpr int TM_TAG_W  = MADDR_W - 6;
static constexpr int R_TM_ROP  = R_SEND;
static constexpr int R_TM_RT   = R_TM_ROP + 4;
static constexpr int R_TM_ADDR = R_TM_RT + 1;
static constexpr int R_TM_OP   = R_TM_ADDR + TM_TAG_W;
static constexpr int R_TM_EN   = R_TM_OP + 2;
static constexpr int R_TM_END  = R_TM_EN + 1;
static constexpr int TM_ENTRIES = 8;
static constexpr uint64_t TMAP_BASE = 0x40000000ull;

// The 128-byte map in memory: the slot words, then words 22-23, then junk.
static std::array<uint32_t, 32> map_words(const Xfer& x) {
    std::array<uint32_t, 32> w{};
    w[VX_DCR_DXA_DESC_BASE_LO_OFF] = uint32_t(x.base);
    w[VX_DCR_DXA_DESC_BASE_HI_OFF] = uint32_t(x.base >> 32);
    for (int i = 0; i < 5; ++i)
        w[VX_DCR_DXA_DESC_SIZE0_OFF + i] = x.size[i];
    for (int i = 0; i < 4; ++i)
        w[VX_DCR_DXA_DESC_STRIDE0_OFF + i] = x.stride[i];
    w[VX_DCR_DXA_DESC_META_OFF] = (x.rank0 ? 0 : x.rank) | (x.esz << 3) | (x.swz << 7) | (x.il << 9) | (x.im2col << 13);
    for (int i = 0; i < 5; ++i)
        w[VX_DCR_DXA_DESC_ESTRIDE0_OFF + i] = x.es[i];
    w[VX_DCR_DXA_DESC_TILESIZE01_OFF] = (x.tile[1] << 16) | x.tile[0];
    w[VX_DCR_DXA_DESC_TILESIZE23_OFF] = (x.tile[3] << 16) | x.tile[2];
    w[VX_DCR_DXA_DESC_TILESIZE4_OFF]  = x.tile[4];
    w[VX_DCR_DXA_DESC_CFILL_OFF]      = x.cfill;
    w[VX_DCR_DXA_DESC_SMEM_STRIDE_OFF] = 0;
    w[22] = x.w22;
    w[23] = x.etype;
    for (int i = 24; i < 32; ++i)
        w[i] = 0xdead0000u | uint32_t(i);
    return w;
}

static std::map<uint64_t, std::array<uint32_t, 32>> g_maps;
// Maps whose memory holds stale words (g_stale) until the vector's launch writes
// the new ones (g_new, kept from the steps).
static std::map<uint64_t, std::array<uint32_t, 32>> g_stale, g_new;

static void tm_store(const Xfer& x) {
    if (x.use_map && !g_new.count(x.map))
        g_maps[x.map] = map_words(x);
}

// A line of the map region and its bytes (bytes outside a map read as plain memory).
static bool tm_line(uint64_t line, uint8_t* bytes) {
    uint64_t a = line * CL;
    if (a + CL <= TMAP_BASE || a >= TMAP_BASE + 0x100000)
        return false;
    if (bytes) {
        for (int i = 0; i < CL; ++i) {
            uint64_t b = a + i;
            bytes[i] = gmem_byte(b);
            for (auto& m : g_maps)
                if (b >= m.first && b < m.first + 128)
                    bytes[i] = uint8_t(m.second[(b - m.first) / 4] >> (8 * ((b - m.first) % 4)));
        }
    }
    return true;
}

static void load_tm(const Xfer& x) {
    if (!x.use_map && !x.tmop)
        return;
    set_bits(dut->dxa_req_data, R_TM_EN, 1, 1);
    set_bits(dut->dxa_req_data, R_TM_OP, 2, uint32_t(x.tmop));
    set_bits(dut->dxa_req_data, R_TM_ADDR, TM_TAG_W, x.map >> 6);
    if (x.tmop) {
        set_bits(dut->dxa_req_data, R_TLOAD, 1, 0);
        set_bits(dut->dxa_req_data, R_PF, 1, 0);
        for (int i = 0; i < 5; ++i)
            set_bits(dut->dxa_req_data, R_COORDS + 32 * i, 32, 0);
    }
    set_bits(dut->dxa_req_data, R_META, 32, 0xC0000000u);   // slot bits 0
}

// The descriptor cache: 8 entries, the lowest invalid one or a round-robin victim.
struct TmModel {
    bool v[TM_ENTRIES] = {};
    uint64_t tag[TM_ENTRIES] = {};
    int rr = 0;
    uint64_t misses = 0, hits = 0, lines = 0;
    bool find(uint64_t a) const {
        for (int e = 0; e < TM_ENTRIES; ++e)
            if (v[e] && tag[e] == a) return true;
        return false;
    }
    void launch() {
        for (auto& b : v) b = false;
        rr = 0;
    }
    void step(const Xfer& x) {
        if (x.kmu) {
            launch();
            return;
        }
        if (!x.use_map && !x.tmop)
            return;
        if (x.tmop == 2) {
            for (int e = 0; e < TM_ENTRIES; ++e)
                if (v[e] && tag[e] == x.map) v[e] = false;
            return;
        }
        if (find(x.map)) { ++hits; return; }
        ++misses;
        lines += ((x.map % CL) + 127) / CL + 1;
        if (x.kill) {
            // The killed fetch fills nothing; a transfer misses again, a prefetch pops.
            launch();
            if (x.tmop)
                return;
            ++misses;
            lines += ((x.map % CL) + 127) / CL + 1;
        }
        int e = -1;
        for (int i = 0; i < TM_ENTRIES && e < 0; ++i)
            if (!v[i]) e = i;
        if (e < 0) { e = rr; rr = (rr + 1) % TM_ENTRIES; }
        v[e] = true; tag[e] = x.map;
    }
};

static uint64_t tm_addr(int k) { return TMAP_BASE + 0x40 + uint64_t(k) * 0xc0; }

static Xfer tm_use(const char* name, int k, int32_t c0, int32_t c1, uint64_t base, uint32_t smem) {
    Xfer x = tile2d(name, 2, 64, 32, 16, 8, c0, c1, base, smem);
    x.use_map = true;
    x.map = tm_addr(k);
    return x;
}

static Xfer tm_op(int op, int k) {
    Xfer x;
    x.name = op == 1 ? "prefetch" : "inval";
    x.tmop = op;
    x.map = tm_addr(k);
    return x;
}

static Xfer tm_kmu() {
    Xfer x;
    x.name = "launch";
    x.kmu = true;
    return x;
}

// Runs the steps, then checks the map line reads against the cache model.
// A map's memory holds the descriptor of the step that names it, from its
// first use on (a prefetch reads the map of its next use).
static bool run_tmap(const char* name, const std::vector<Xfer>& xs, int ready_pct, int max_lat,
                     bool serial = true, int rev = 0, bool maps_first = false) {
    if (R_TM_END > int(sizeof(dut->dxa_req_data)) * 8) {
        printf("%s: the request has no tensor-map fields\n", name);
        return false;
    }
    TmModel m;
    for (auto& x : xs)
        m.step(x);
    g_maps.clear();
    for (auto& x : xs)
        if (x.use_map && !g_maps.count(x.map))
            g_maps[x.map] = map_words(x);
    g_new.clear();
    for (auto& st : g_stale) {
        g_new[st.first] = g_maps[st.first];
        g_maps[st.first] = st.second;
    }
#ifdef PERF_ENABLE
    uint64_t h0 = dut->perf_tmap_hits, m0 = dut->perf_tmap_misses;
#endif
    g_rsp_rev = rev;
    g_maps_first = maps_first;
    bool ok = run_tensor(name, xs, ready_pct, max_lat, serial);
    g_rsp_rev = 0;
    g_maps_first = false;
    if (g_map_reads != m.lines) {
        printf("%s: %lu map line reads, expected %lu (%lu misses)\n", name, (unsigned long)g_map_reads,
               (unsigned long)m.lines, (unsigned long)m.misses);
        ok = false;
    }
#ifdef PERF_ENABLE
    if (dut->perf_tmap_hits - h0 != m.hits || dut->perf_tmap_misses - m0 != m.misses) {
        printf("%s: perf hits=%lu misses=%lu, expected %lu %lu\n", name, (unsigned long)(dut->perf_tmap_hits - h0),
               (unsigned long)(dut->perf_tmap_misses - m0), (unsigned long)m.hits, (unsigned long)m.misses);
        ok = false;
    }
#endif
    if (ok)
        printf("%s: %lu map misses, %lu hits OK\n", name, (unsigned long)m.misses, (unsigned long)m.hits);
    return ok;
}

static bool run_tmap_all() {
    const uint32_t q = LMEM_BYTES / 4;
    const uint64_t G = 0x100000;
    bool ok = true;
    // A miss reads the map's two lines, then the same map hits; the lines
    // answered in any order, then last first.
    for (int rev = 0; rev < 2; ++rev) {
        Xfer a = tm_use("a", 0, 4, 3, G + 0x40, q);
        Xfer b = tm_use("b", 0, -3, 30, G + 0x40, q + 0x200);
        Xfer c = tm_use("c", 1, 8, 2, G + 0x1000, q + 0x400);
        ok = run_tmap(rev ? "tm-miss-rev" : "tm-miss-hit", {tm_kmu(), a, b, c, a}, 100, 12, true, rev) && ok;
    }
    // Nine maps evict entry 0, map 0 again evicts entry 1, map 2 still hits.
    {
        std::vector<Xfer> xs = {tm_kmu()};
        for (int k = 0; k < 9; ++k)
            xs.push_back(tm_use("ev", k, k, k % 4, G + 0x2000 * k, q + 0x200 * (k % 4)));
        xs.push_back(tm_use("ev0", 0, 1, 1, G, q));
        xs.push_back(tm_use("ev2", 2, 2, 2, G + 0x4000, q + 0x200));
        xs.push_back(tm_use("ev1", 1, 3, 3, G + 0x2000, q + 0x400));
        ok = run_tmap("tm-evict", xs, 70, 20) && ok;
    }
    // An invalidate drops the entry, so a rewritten map is read again; one of
    // a map not cached changes nothing, nor does one of another map.
    {
        Xfer a = tm_use("inv-a", 3, 4, 3, G + 0x40, q);
        Xfer a2 = tm_use("inv-a2", 3, 1, 5, G + 0x8000, q + 0x200);
        Xfer b = tm_use("inv-b", 4, 0, 0, G + 0x3000, q + 0x400);
        ok = run_tmap("tm-inval", {tm_kmu(), a, b, tm_op(2, 3), a2, tm_op(2, 5), tm_op(2, 4), a2}, 80, 12) && ok;
    }
    // A prefetch fills the entry without a transfer; a prefetch of a cached
    // map only pops.
    {
        Xfer a = tm_use("pf-a", 6, 2, 2, G + 0x9000, q);
        ok = run_tmap("tm-prefetch", {tm_kmu(), tm_op(1, 6), a, tm_op(1, 6), a}, 80, 12) && ok;
    }
    // A launch (KMU DCR write) drops every map; a slot request keeps them.
    {
        Xfer a = tm_use("ln-a", 7, 4, 3, G + 0x40, q);
        Xfer a2 = tm_use("ln-a2", 7, 5, 1, G + 0xa000, q + 0x200);
        Xfer s = tile2d("ln-slot", 2, 40, 12, 16, 4, 1, 1, G + 0x5000, q + 0x400);
        ok = run_tmap("tm-launch", {tm_kmu(), a, s, a, tm_kmu(), a2}, 80, 12) && ok;
    }
    // A slot request behind a missing map waits for the fill; requests stream.
    {
        Xfer a = tm_use("bh-a", 8, 4, 3, G + 0x40, q);
        Xfer s = tile2d("bh-slot", 2, 50, 10, 32, 8, -4, 5, G + 0x240000, q + 0x400);
        Xfer b = tm_use("bh-b", 8, 9, 0, G + 0x40, q + 0x800);
        ok = run_tmap("tm-behind", {tm_kmu(), a, s, b}, 100, 30, false, 1, true) && ok;
    }
    // A launch during a map fetch, after its first line is answered from an
    // older map: the fetch fills nothing and the transfer reads the map again
    // (a prefetch only pops), so the old words are never used; the map then hits.
    for (int pf = 0; pf < 2; ++pf) {
        const char* name = pf ? "tm-kill-pf" : "tm-kill";
        Xfer a = tm_use("kl-a", 9, -3, 2, G + 0x40, q);
        Xfer a2 = tm_use("kl-a2", 9, 6, 1, G + 0x40, q + 0x400);
        Xfer old = tm_use("kl-old", 9, 0, 0, G + 0x6000, q);
        old.tile[0] = 8;                  // differs from a in both map lines
        old.cfill = 0x5a5a5a5au;
        g_stale[a.map] = map_words(old);
        bool fired = false;
        // The launch goes between the two line responses: the second waits for it.
        g_map_gate = [&](uint64_t done) { return !fired && done == 1; };
        g_dcr_hook = [&](uint64_t done, bool map_rsp, uint32_t& addr, uint32_t& data) {
            if (fired || done != 1 || map_rsp)
                return false;
            fired = true;
            g_maps[a.map] = g_new[a.map];
            g_new.erase(a.map);
            addr = VX_DCR_KMU_STATE_BEGIN;
            data = 0;
            return true;
        };
        Xfer first = pf ? tm_op(1, 9) : a;
        first.kill = true;
        std::vector<Xfer> xs = {tm_kmu(), first};
        if (pf)
            xs.push_back(a);
        xs.push_back(a2);
        ok = run_tmap(name, xs, 80, 12) && ok;
        if (!fired) {
            printf("%s: no launch during the fetch\n", name);
            ok = false;
        }
        g_dcr_hook = nullptr;
        g_map_gate = nullptr;
        g_stale.clear();
        g_new.clear();
    }
    // A launch restarts the victim order: nine maps (the ninth evicts entry 0),
    // a launch, nine more (the ninth evicts entry 0 again); the other seven hit.
    {
        std::vector<Xfer> xs = {tm_kmu()};
        for (int k = 0; k < 9; ++k)
            xs.push_back(tm_use("rr", k, k, k % 4, G + 0x2000 * k, q + 0x200 * (k % 4)));
        xs.push_back(tm_kmu());
        xs.back().in_loop = true;
        for (int k = 10; k < 19; ++k)
            xs.push_back(tm_use("rr2", k, k % 7, k % 4, G + 0x2000 * (k % 9), q + 0x200 * (k % 4)));
        for (int k = 11; k < 18; ++k)
            xs.push_back(tm_use("rr-hit", k, k % 5, k % 3, G + 0x2000 * (k % 9), q + 0x200 * (k % 4)));
        ok = run_tmap("tm-rr-launch", xs, 70, 20) && ok;
    }
    // A slot DCR write in the cycle a map line is answered: the line is held
    // that cycle and written after, and the DCR write lands too (it repairs the
    // descriptor of the slot request queued behind the map).
    {
        Xfer a = tm_use("dw-a", 12, 4, 3, G + 0x40, q);
        Xfer s = tile2d("dw-slot", 2, 50, 10, 32, 8, -4, 5, G + 0x240000, q + 0x400);
        s.cfill = 0x11223344u;
        Xfer b = tm_use("dw-b", 12, 9, 0, G + 0x40, q + 0x800);
        const uint32_t d = VX_DCR_DXA_DESC_BASE + 2 * VX_DCR_DXA_DESC_STRIDE;   // s is step 2
        const uint32_t fix[2][2] = {{d + VX_DCR_DXA_DESC_BASE_LO_OFF, uint32_t(s.base)},
                                    {d + VX_DCR_DXA_DESC_CFILL_OFF, s.cfill}};
        g_after_program = [&]() {
            dcr_write(fix[0][0], fix[0][1] + 0x1000);
            dcr_write(fix[1][0], 0);
        };
        int n = 0;
        g_dcr_hook = [&](uint64_t, bool map_rsp, uint32_t& addr, uint32_t& data) {
            if (!map_rsp || n >= 2)
                return false;
            addr = fix[n][0];
            data = fix[n][1];
            ++n;
            return true;
        };
        bool r = run_tmap("tm-dcr-fill", {tm_kmu(), a, s, b}, 100, 30, false);
        if (n != 2 || g_dcr_held != 2 || g_dcr_taken) {
            printf("tm-dcr-fill: %d slot DCR writes on a map response, %lu held, %lu taken\n", n,
                   (unsigned long)g_dcr_held, (unsigned long)g_dcr_taken);
            r = false;
        } else if (r) {
            printf("tm-dcr-fill: 2 map responses held a cycle by a slot DCR write OK\n");
        }
        ok = r && ok;
        g_dcr_hook = nullptr;
        g_after_program = nullptr;
    }
    return ok;
}

// Negative vectors: each must stop the simulation with the given assert.
static void run_negative(const char* which) {
    const uint32_t q = LMEM_BYTES / 4;
    const uint64_t G = 0x100000;
    Xfer f = tile2d("neg", 2, 16, 3, 16, 3, 0, 0, G, q + 0x300);
    f.blk_base = q;
    if (!strcmp(which, "box")) {
        f.room = 191;                    // DXA box exceeds the destination block (dense bytes)
    } else if (!strcmp(which, "swzbox")) {
        f.swz = 3; f.room = 255;         // the same, by the swizzled footprint only
    } else if (!strcmp(which, "tmrank") || !strcmp(which, "tmwide")) {
        // DXA tensor map rank must be 1-5; DXA tensor stride needs 32 bits
        Xfer m = tm_use(which, 0, 0, 0, G, q);
        m.rank0 = !strcmp(which, "tmrank");
        m.w22 = m.rank0 ? 0 : 0x100;
        run_tmap(which, {tm_kmu(), m}, 100, 4);
        return;
    } else {
        printf("unknown negative vector %s\n", which);
        return;
    }
    run_tensor(which, {f}, 100, 4, true);
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
    dut->row_out_ready  = 1;
#endif

    reset_dut();

    // ---- run a few idle cycles -------------------------------------------
    for (int i = 0; i < MAX_SIM_CYCLES; i++)
        clock_cycle();

    bool ok = true;
#if !defined(VX_CFG_EXT_MBAR_ENABLE)
    ok = run_multicast();
#else
    dut->slot_stride = MC_STRIDE;
    if (const char* neg = std::getenv("DXA_NEGATIVE")) {
        run_negative(neg);
        ok = false;
    } else {
        dut->slot_stride = 0x400;
        ok = run_bulk_all();
        dut->slot_stride = MC_STRIDE;
        ok = run_tensor_all() && ok;
        ok = run_store_all() && ok;
        ok = run_tmap_all() && ok;
    }
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
