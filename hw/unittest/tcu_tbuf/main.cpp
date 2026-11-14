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

// VX_tcu_tbuf testbench for operands read through H100 wgmma descriptors.
//
// Four blocks replay WGMMA micro-op sequences against a random local memory
// behind a random-latency, randomly back-pressured read port. Descriptors,
// slot bases and memory contents are random; every operand word a block is
// handed when its micro-op fires is checked against the element addresses of
// the descriptor golden.
//
// Phases: each phase runs one CTA (its slot base) on all blocks, either as a
// warpgroup (same instructions, blocks drifting apart), as two pairs on
// different descriptors, or with independent instructions per block. Between
// phases the memory may be rewritten under unchanged descriptors, so a
// buffer that skipped its refill would hand out stale words.

#include "VVX_tcu_tbuf_top.h"
#include <verilated.h>
#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "VX_config.h"
#include "tcu_desc_ref.h"

using namespace vortex;

static uint64_t timestamp = 0;
double sc_time_stamp() { return timestamp; }

static_assert(VX_CFG_XLEN == 64, "tcu_tbuf bench: XLEN 64");
static_assert(VX_CFG_NUM_THREADS == 32, "tcu_tbuf bench: 32 threads");
static_assert(VX_CFG_NUM_TCU_BLOCKS == 4, "tcu_tbuf bench: 4 blocks");

constexpr int NB          = VX_CFG_NUM_TCU_BLOCKS;
constexpr int LMEM_LOG    = VX_CFG_LMEM_LOG_SIZE;
constexpr uint32_t LMEM_SIZE = 1u << LMEM_LOG;
constexpr int ROW_BYTES   = VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8);
constexpr int ROW_WORDS   = ROW_BYTES / 4;
constexpr int TC_M = 8, TC_N = 4, FEDP_K = 4;
constexpr int A_LANES = TC_M * FEDP_K;    // rs1 words per block
constexpr int RS2_LANES = 32;             // rs2 words per block
constexpr int B_BLOCK_WORDS = FEDP_K * TC_N;

// ---------------------------------------------------------------------------
// Bit helpers over Verilator signals (integral or VlWide).
// ---------------------------------------------------------------------------

template <typename T>
static void setf(T& s, int lsb, int w, uint64_t v) {
    for (int i = 0; i < w; ++i) {
        T bit = T(1) << (lsb + i);
        if ((v >> i) & 1) s |= bit; else s &= ~bit;
    }
}
template <std::size_t N>
static void setf(VlWide<N>& s, int lsb, int w, uint64_t v) {
    for (int i = 0; i < w; ++i) {
        int p = lsb + i;
        uint32_t bit = 1u << (p % 32);
        if ((v >> i) & 1) s[p / 32] |= bit; else s[p / 32] &= ~bit;
    }
}
template <typename T>
static uint64_t getf(const T& s, int lsb, int w) {
    return (uint64_t(s) >> lsb) & ((w >= 64) ? ~0ull : ((1ull << w) - 1));
}
template <std::size_t N>
static uint64_t getf(const VlWide<N>& s, int lsb, int w) {
    uint64_t v = 0;
    for (int i = 0; i < w; ++i) {
        int p = lsb + i;
        v |= uint64_t((s[p / 32] >> (p % 32)) & 1) << i;
    }
    return v;
}
template <std::size_t N>
static uint32_t getw(const VlWide<N>& s, int idx) { return s[idx]; }

// ---------------------------------------------------------------------------
// Micro-op sequence and column map of one WGMMA.
// ---------------------------------------------------------------------------

struct Uop { int m, n, k; };

// The micro-op order of a WGMMA: ctr walks the NRC accumulator registers
// k-step by k-step; an f16 D visits each register once per column parity.
static std::vector<Uop> make_uops(int cd_nregs, bool f16_out) {
    int lg  = 3 + cd_nregs;
    int nrc = 1 << lg;
    int cnt = f16_out ? 4 * nrc : 2 * nrc;
    std::vector<Uop> u;
    for (int ctr = 0; ctr < cnt; ++ctr) {
        int r = ctr % nrc, h, j, e, k;
        if (f16_out) {
            h = r & 1; j = r >> 1; e = (ctr >> lg) & 1; k = ctr >> (lg + 1);
        } else {
            e = r & 1; h = (r >> 1) & 1; j = r >> 2; k = ctr >> lg;
        }
        u.push_back({h, 2 * j + e, k});
    }
    return u;
}

// Column of B that lane column q of micro-op column step n computes.
static int b_col(int step_n, int q) {
    int j = step_n >> 1, e = step_n & 1;
    return 8 * j + 2 * q + e;
}


// ---------------------------------------------------------------------------
// Test state
// ---------------------------------------------------------------------------

struct Instr {
    uint64_t da, db;
    bool e16;       // 16-bit elements (else 8-bit)
    bool f16_out;
    int cd_nregs;
    std::vector<Uop> uops;
};

struct Block {
    std::vector<Instr> prog;
    size_t ii = 0, ui = 0;
    int gap = 0;
    bool done() const { return ii >= prog.size(); }
};

static std::mt19937_64 rng;
static uint32_t rnd(uint32_t n) { return uint32_t(rng() % n); }

static std::vector<uint8_t> lmem(LMEM_SIZE);

static void fill_lmem() {
    for (auto& b : lmem) b = uint8_t(rng());
}

// Negative controls: each check also evaluates deliberately wrong goldens,
// which must disagree with the hardware somewhere in the run, so a check
// that compares nothing or reads a constant cannot pass.
enum { NEG_NONE = 0, NEG_NO_SWIZZLE, NEG_NO_BASE, NEG_MAJOR, NEG_ROW, NEG_COUNT };
static const char* const neg_name[NEG_COUNT] = {"", "swizzle dropped", "slot base dropped",
                                                "major flipped", "row/column off by one"};
static uint64_t neg_seen[NEG_COUNT];
static uint64_t neg_caught[NEG_COUNT];

static uint32_t elem_lmem(uint64_t desc, uint32_t base, int esize, int mn, int k, int neg = NEG_NONE) {
    bool mn_major = (desc >> 52) & 1;
    if (neg == NEG_MAJOR) mn_major = !mn_major;
    if (neg == NEG_ROW) mn ^= 1;
    uint32_t rel;
    if (neg == NEG_NO_SWIZZLE)
        rel = tcu_desc_ref::linear_addr(tcu_desc_ref::decode(desc), mn_major, esize, mn, k);
    else
        rel = tcu_desc_ref::operand_addr(desc, mn_major, esize, mn, k);
    if (neg == NEG_NO_BASE) base = 0;
    return (base + rel) & (LMEM_SIZE - 1);
}

static uint32_t golden_word(uint64_t desc, uint32_t base, bool e16, int mn, int kstep, int kw, int neg = NEG_NONE) {
    uint32_t w = 0;
    if (e16) {
        for (int h = 0; h < 2; ++h) {
            uint32_t a = elem_lmem(desc, base, 2, mn, kstep * 8 + 2 * kw + h, neg);
            w |= (uint32_t(lmem[a]) | (uint32_t(lmem[a + 1]) << 8)) << (16 * h);
        }
    } else {
        for (int t = 0; t < 4; ++t) {
            uint32_t a = elem_lmem(desc, base, 1, mn, kstep * 16 + 4 * kw + t, neg);
            w |= uint32_t(lmem[a]) << (8 * t);
        }
    }
    return w;
}

static void check_negatives(uint64_t desc, uint32_t base, bool e16, int mn, int kstep, int kw, uint32_t got) {
    for (int n = 1; n < NEG_COUNT; ++n) {
        if (n == NEG_MAJOR && !e16) continue;
        if (n == NEG_NO_SWIZZLE && (desc >> 62) == 0) continue;
        ++neg_seen[n];
        neg_caught[n] += golden_word(desc, base, e16, mn, kstep, kw, n) != got;
    }
}

static uint16_t rand_field14() { return uint16_t(rng() & 0x3fff); }

// A random descriptor; half the offsets are canonical multi-atom strides,
// the rest anywhere in the 14-bit field.
static uint64_t rand_desc(int layout, bool mn_major) {
    static const int W[4] = {1, 8, 4, 2};
    int w = W[layout];
    uint64_t start = rand_field14();
    uint64_t lbo, sbo;
    if (rnd(2)) {
        lbo = rand_field14();
        sbo = rand_field14();
    } else if (!mn_major) {
        // K-major: SBO steps 8-row groups, LBO (no swizzle) the k chunks.
        sbo = (8 * w * 16 * (1 + rnd(4))) >> 4;
        lbo = layout ? rnd(2) : ((128 * (1 + rnd(32))) >> 4);
    } else {
        // MN-major: LBO steps MN atoms, SBO the 8-k groups (swapped when unswizzled).
        lbo = (8 * w * 16 * (1 + rnd(8))) >> 4;
        sbo = (8 * w * 16 * (1 + rnd(8))) >> 4;
    }
    return start | (lbo << 16) | (sbo << 32) | (uint64_t(mn_major) << 52)
         | (uint64_t(layout) << 62);
}

static Instr rand_instr() {
    Instr in;
    in.e16 = rnd(4) != 0;
    in.f16_out = in.e16 && rnd(2);
    in.cd_nregs = int(rnd(3));
    int la = int(rnd(4)), lb = int(rnd(4));
    bool ma = in.e16 && rnd(2), mb = in.e16 && rnd(2);
    in.da = rand_desc(la, ma);
    in.db = rand_desc(lb, mb);
    in.uops = make_uops(in.cd_nregs, in.f16_out);
    return in;
}

// Coverage: operand (0 = A, 1 = B) x layout x major x esize16.
static std::map<std::string, uint64_t> cover;
static void cover_hit(int op, uint64_t desc, bool e16) {
    char key[32];
    snprintf(key, sizeof(key), "%c-L%d-%s-e%d", op ? 'B' : 'A', int(desc >> 62),
             ((desc >> 52) & 1) ? "mn" : "k", e16 ? 16 : 8);
    cover[key]++;
}

// ---------------------------------------------------------------------------

struct LmemRsp { uint64_t at; uint32_t row; uint64_t tag; };

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    uint64_t seed = 1;
    int phases = 60;
    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "--seed=", 7)) seed = strtoull(argv[i] + 7, nullptr, 0);
        else if (!strncmp(argv[i], "--phases=", 9)) phases = atoi(argv[i] + 9);
    }
    rng.seed(seed);

    auto top = new VVX_tcu_tbuf_top;
#ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    auto tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open("trace.vcd");
#endif
    auto tick = [&](int clk) {
        top->clk = clk;
        top->eval();
#ifdef VCD_OUTPUT
        tfp->dump(timestamp);
#endif
        ++timestamp;
    };

    top->reset = 1;
    top->req_valid = 0;
    top->lmem_req_ready = 0;
    top->lmem_rsp_valid = 0;
    for (int i = 0; i < 4; ++i) { tick(0); tick(1); }
    top->reset = 0;

    fill_lmem();

    uint64_t cycle = 0, fires = 0, words = 0, errors = 0, lmem_reads = 0;
    std::deque<LmemRsp> rspq;
    Block blk[NB];
    Instr prev_inst;
    uint32_t prev_base = 0;
    int prev_cta = 0;
    bool have_prev = false;
    int mode_count[3] = {0, 0, 0};
    int rewrites = 0;

    for (int ph = 0; ph < phases && errors == 0; ++ph) {
        // A CTA: its id and slot base (nonzero, 16-byte aligned; usually 1 KB).
        int cta;
        uint32_t base;
        bool rewrite = have_prev && rnd(3) == 0;
        if (rewrite) {
            cta = prev_cta;
            base = prev_base;
        } else {
            cta = int(rnd(1u << 2));
            base = rnd(4) ? ((1 + rnd((LMEM_SIZE >> 10) - 1)) << 10)
                          : ((1 + rnd((LMEM_SIZE >> 4) - 1)) << 4);
        }
        int mode = rewrite ? 0 : int(rnd(3));
        mode_count[mode]++;
        if (rewrite) {
            // Same instruction on the same descriptors over new contents.
            fill_lmem();
            ++rewrites;
            for (int b = 0; b < NB; ++b) blk[b].prog.assign(1, prev_inst);
        } else if (mode == 0) {
            int n = 1 + int(rnd(3));
            std::vector<Instr> p;
            for (int i = 0; i < n; ++i) p.push_back(rand_instr());
            for (int b = 0; b < NB; ++b) blk[b].prog = p;
            prev_inst = p.back();
        } else if (mode == 1) {
            std::vector<Instr> p0, p1;
            int n = 1 + int(rnd(2));
            for (int i = 0; i < n; ++i) {
                Instr a = rand_instr(), c = a;
                c.db = rand_desc(int(rnd(4)), c.e16 && rnd(2));
                c.da = rand_desc(int(rnd(4)), c.e16 && rnd(2));
                p0.push_back(a);
                p1.push_back(c);
            }
            blk[0].prog = p0; blk[1].prog = p0;
            blk[2].prog = p1; blk[3].prog = p1;
            prev_inst = p0.back();
        } else {
            for (int b = 0; b < NB; ++b) {
                blk[b].prog.clear();
                int n = 1 + int(rnd(2));
                for (int i = 0; i < n; ++i) blk[b].prog.push_back(rand_instr());
            }
            prev_inst = blk[0].prog.back();
        }
        prev_base = base;
        prev_cta = cta;
        have_prev = true;
        for (int b = 0; b < NB; ++b) {
            blk[b].ii = 0;
            blk[b].ui = 0;
            blk[b].gap = int(rnd(4));
        }

        uint64_t last_fire = cycle;
        for (;;) {
            bool all_done = true;
            for (int b = 0; b < NB; ++b) all_done &= blk[b].done();
            if (all_done && rspq.empty()) break;
            if (cycle - last_fire > 20000) {
                printf("FAILED: no micro-op fired for 20000 cycles (phase %d, cycle %lu)\n",
                       ph, (unsigned long)cycle);
                return 1;
            }

            // ---- drive inputs (clock low) ----
            bool valid[NB];
            for (int b = 0; b < NB; ++b) {
                Block& B = blk[b];
                valid[b] = !B.done() && B.gap == 0;
                setf(top->req_valid, b, 1, valid[b]);
                if (B.done()) continue;
                const Instr& in = B.prog[B.ii];
                const Uop& u = in.uops[B.ui];
                setf(top->req_cta_id, b * 2, 2, uint64_t(cta));
                setf(top->req_step_m, b * 4, 4, u.m);
                setf(top->req_step_n, b * 5, 5, u.n);
                setf(top->req_step_k, b * 4, 4, u.k);
                setf(top->req_cd_nregs, b * 2, 2, in.cd_nregs);
                setf(top->req_desc_a, b * 64, 64, in.da);
                setf(top->req_desc_b, b * 64, 64, in.db);
                setf(top->req_is_first_uop, b, 1, B.ui == 0);
                setf(top->req_is_last_uop, b, 1, B.ui + 1 == in.uops.size());
                setf(top->req_slot_base, b * LMEM_LOG, LMEM_LOG, base);
                setf(top->req_esize16, b, 1, in.e16);
            }
            top->lmem_req_ready = rnd(4) != 0;
            bool rsp_drive = !rspq.empty() && rspq.front().at <= cycle;
            top->lmem_rsp_valid = rsp_drive;
            if (rsp_drive) {
                uint32_t row = rspq.front().row;
                for (int w = 0; w < ROW_WORDS; ++w) {
                    uint32_t a = row * ROW_BYTES + 4 * w;
                    top->lmem_rsp_data[w] = uint32_t(lmem[a]) | (uint32_t(lmem[a + 1]) << 8)
                                          | (uint32_t(lmem[a + 2]) << 16) | (uint32_t(lmem[a + 3]) << 24);
                }
                top->lmem_rsp_tag = rspq.front().tag;
            }
            tick(0);

            // ---- sample ----
            if (!top->lmem_rsp_ready) {
                printf("FAILED: response port not ready\n");
                return 1;
            }
            if (rsp_drive) rspq.pop_front();
            if (top->lmem_req_valid && top->lmem_req_ready) {
                if (top->lmem_req_rw) {
                    printf("FAILED: tile buffer wrote local memory\n");
                    return 1;
                }
                uint64_t at = cycle + 1 + rnd(4);
                if (!rspq.empty() && at <= rspq.back().at) at = rspq.back().at + 1;
                rspq.push_back({at, uint32_t(top->lmem_req_addr), uint64_t(top->lmem_req_tag)});
                ++lmem_reads;
            }
            for (int b = 0; b < NB; ++b) {
                Block& B = blk[b];
                if (!valid[b]) {
                    if (B.gap > 0) --B.gap;
                    continue;
                }
                bool rdy = (top->ready >> b) & 1;
                bool downstream = rnd(8) != 0;
                if (!(rdy && downstream)) continue;
                const Instr& in = B.prog[B.ii];
                const Uop& u = in.uops[B.ui];
                // A: rows 16 b + 8 step_m + i of the m64 tile.
                for (int i = 0; i < TC_M; ++i) {
                    for (int kw = 0; kw < FEDP_K; ++kw) {
                        int mn = 16 * b + 8 * u.m + i;
                        uint32_t exp = golden_word(in.da, base, in.e16, mn, u.k, kw);
                        uint32_t got = getw(top->rs1_data, (b * A_LANES) + i * FEDP_K + kw);
                        ++words;
                        check_negatives(in.da, base, in.e16, mn, u.k, kw, got);
                        if (got != exp && errors++ < 10)
                            printf("ERROR: A block %d instr %zu uop %zu (m%d n%d k%d) row %d word %d: got 0x%08x exp 0x%08x desc 0x%016llx base 0x%x e16 %d\n",
                                   b, B.ii, B.ui, u.m, u.n, u.k, mn, kw, got, exp,
                                   (unsigned long long)in.da, base, in.e16);
                    }
                }
                cover_hit(0, in.da, in.e16);
                // B: columns b_col(step_n, q) at the core's offset for step_n.
                int b_off = (u.n & 1) * B_BLOCK_WORDS;
                for (int q = 0; q < TC_N; ++q) {
                    for (int kw = 0; kw < FEDP_K; ++kw) {
                        int mn = b_col(u.n, q);
                        uint32_t exp = golden_word(in.db, base, in.e16, mn, u.k, kw);
                        uint32_t got = getw(top->rs2_data, (b * RS2_LANES) + b_off + q * FEDP_K + kw);
                        ++words;
                        check_negatives(in.db, base, in.e16, mn, u.k, kw, got);
                        if (got != exp && errors++ < 10)
                            printf("ERROR: B block %d instr %zu uop %zu (m%d n%d k%d) col %d word %d: got 0x%08x exp 0x%08x desc 0x%016llx base 0x%x e16 %d\n",
                                   b, B.ii, B.ui, u.m, u.n, u.k, mn, kw, got, exp,
                                   (unsigned long long)in.db, base, in.e16);
                    }
                }
                cover_hit(1, in.db, in.e16);
                ++fires;
                last_fire = cycle;
                if (++B.ui == in.uops.size()) {
                    B.ui = 0;
                    ++B.ii;
                    B.gap = int(rnd(3));
                } else {
                    B.gap = rnd(6) == 0;
                }
            }

            tick(1);
            ++cycle;
        }
    }

    printf("tcu_tbuf: seed %lu, %d phases (warpgroup %d, two descriptor pairs %d, independent %d, rewrites %d), %lu cycles, %lu micro-ops, %lu words, %lu local-memory reads\n",
           (unsigned long)seed, phases, mode_count[0], mode_count[1], mode_count[2], rewrites,
           (unsigned long)cycle, (unsigned long)fires, (unsigned long)words, (unsigned long)lmem_reads);
    int missing = 0;
    for (int op = 0; op < 2; ++op)
        for (int l = 0; l < 4; ++l)
            for (int mj = 0; mj < 2; ++mj)
                for (int e = 0; e < 2; ++e) {
                    if (mj && !e) continue;
                    char key[32];
                    snprintf(key, sizeof(key), "%c-L%d-%s-e%d", op ? 'B' : 'A', l, mj ? "mn" : "k", e ? 16 : 8);
                    if (cover[key] == 0) {
                        printf("coverage: %s not exercised\n", key);
                        ++missing;
                    }
                }
    for (int n = 1; n < NEG_COUNT; ++n) {
        printf("negative control '%s': %lu of %lu words differ\n", neg_name[n],
               (unsigned long)neg_caught[n], (unsigned long)neg_seen[n]);
        if (neg_caught[n] == 0) {
            printf("negative control '%s' was never caught\n", neg_name[n]);
            ++missing;
        }
    }
#ifdef VCD_OUTPUT
    tfp->close();
#endif
    top->final();
    delete top;
    if (errors || missing) {
        printf("FAILED! (%lu mismatches, %d coverage holes)\n", (unsigned long)errors, missing);
        return 1;
    }
    printf("PASSED!\n");
    return 0;
}
