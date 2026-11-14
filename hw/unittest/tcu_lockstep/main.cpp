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

// VX_tcu_lockstep testbench: the CTA gate of the shared WGMMA tile buffers.
//
// Four blocks present WGMMA micro-op sequences of given CTAs; a block fires
// when it presents a micro-op, the gate does not defer it and its downstream
// is ready. Directed cases pin the exact fire cycles of same-cycle starts of
// different CTAs, a block mid-expansion against a lower block that starts,
// and a lower block that is itself deferred; a random phase stresses the
// rest. Every case must make progress and never have two blocks inside
// WGMMAs of different CTAs.

#include "VVX_tcu_lockstep_top.h"
#include <verilated.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static uint64_t timestamp = 0;
double sc_time_stamp() { return timestamp; }

constexpr int NB = 4;
constexpr int NO_PROGRESS = 200;

struct Instr { int cta; int start; int nuops; };

struct Block {
    std::vector<Instr> prog;
    size_t ii = 0;
    int ui = 0;
    std::vector<int> fires;   // fire cycle of every micro-op
    bool in_exp = false;
    int exp_cta = -1;
    bool done() const { return ii >= prog.size(); }
};

static VVX_tcu_lockstep_top* top;

static void tick(int clk) {
    top->clk = clk;
    top->eval();
    ++timestamp;
}

static void reset() {
    top->reset = 1;
    top->is_wgmma = 0;
    top->exec_fire = 0;
    for (int i = 0; i < 4; ++i) { tick(0); tick(1); }
    top->reset = 0;
}

// The CTA id width of the RTL: clog2 of the warp count.
static int ncta_bits() {
    int n = 0;
    while ((1 << n) < VX_CFG_NUM_WARPS) ++n;
    return n < 1 ? 1 : n;
}

// Runs the blocks to completion; ready_pct < 100 stalls fires at random.
// Returns false on a deadlock or an invariant violation.
static bool run(Block* blk, const char* name, int ready_pct, std::mt19937& rng, bool quiet) {
    reset();
    int cycle = 0, last_fire = 0, violations = 0;
    for (;;) {
        bool all = true;
        for (int b = 0; b < NB; ++b) all &= blk[b].done();
        if (all) break;
        if (cycle - last_fire > NO_PROGRESS) {
            printf("%s: DEADLOCK at cycle %d:", name, cycle);
            for (int b = 0; b < NB; ++b)
                if (blk[b].in_exp) printf(" block %d inside a WGMMA of CTA %d;", b, blk[b].exp_cta);
            printf("\n");
            return false;
        }
        uint32_t valid = 0, first = 0, last = 0, cta = 0;
        for (int b = 0; b < NB; ++b) {
            Block& B = blk[b];
            if (B.done()) continue;
            const Instr& in = B.prog[B.ii];
            if (cycle < in.start) continue;
            valid |= 1u << b;
            if (B.ui == 0) first |= 1u << b;
            if (B.ui == in.nuops - 1) last |= 1u << b;
            cta |= uint32_t(in.cta) << (b * ncta_bits());
        }
        top->is_wgmma = valid;
        top->is_first_uop = first;
        top->is_last_uop = last;
        top->new_cta = cta;
        top->exec_fire = 0;
        tick(0);
        uint32_t conflict = top->cta_conflict;
        uint32_t fire = 0;
        for (int b = 0; b < NB; ++b) {
            bool ready = ready_pct >= 100 || int(rng() % 100) < ready_pct;
            if (((valid >> b) & 1) && !((conflict >> b) & 1) && ready)
                fire |= 1u << b;
        }
        top->exec_fire = fire;
        top->eval();
        for (int b = 0; b < NB; ++b) {
            if (!((fire >> b) & 1)) continue;
            Block& B = blk[b];
            const Instr& in = B.prog[B.ii];
            B.fires.push_back(cycle);
            if (B.ui == 0) { B.in_exp = true; B.exp_cta = in.cta; }
            if (B.ui == in.nuops - 1) { B.in_exp = false; B.ui = 0; ++B.ii; }
            else ++B.ui;
            last_fire = cycle;
        }
        for (int i = 0; i < NB; ++i)
            for (int j = i + 1; j < NB; ++j)
                if (blk[i].in_exp && blk[j].in_exp && blk[i].exp_cta != blk[j].exp_cta) {
                    if (violations++ < 3 && !quiet)
                        printf("%s: cycle %d: blocks %d and %d inside WGMMAs of CTAs %d and %d\n",
                               name, cycle, i, j, blk[i].exp_cta, blk[j].exp_cta);
                }
        tick(1);
        ++cycle;
    }
    if (violations) {
        printf("%s: %d cycles with two CTAs inside WGMMAs\n", name, violations);
        return false;
    }
    return true;
}

// Expected first-fire cycle of every block's instructions (-1: no instruction).
static bool expect(Block* blk, const char* name, const std::vector<std::vector<int>>& starts) {
    bool ok = true;
    for (int b = 0; b < NB; ++b) {
        size_t f = 0;
        for (size_t i = 0; i < starts[b].size(); ++i) {
            int got = (f < blk[b].fires.size()) ? blk[b].fires[f] : -1;
            if (got != starts[b][i]) {
                printf("%s: block %d instruction %zu first fired at %d, expected %d\n",
                       name, b, i, got, starts[b][i]);
                ok = false;
            }
            f += blk[b].prog[i].nuops;
        }
    }
    return ok;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i)
        if (!strncmp(argv[i], "--seed=", 7)) seed = strtoull(argv[i] + 7, nullptr, 0);
    std::mt19937 rng(seed);
    top = new VVX_tcu_lockstep_top;
    int failed = 0, passed = 0;
    const int A = 1, B = 2;

    struct Case {
        const char* name;
        std::vector<std::vector<Instr>> prog;
        std::vector<std::vector<int>> starts;
    };
    std::vector<Case> cases = {
        // Two CTAs start in one cycle: the lower block goes first.
        {"same-cycle (A,B)", {{{A, 0, 4}}, {{B, 0, 4}}, {}, {}},
         {{0}, {4}, {}, {}}},
        // (A,B,A): block 1 defers, so it does not defer block 2.
        {"same-cycle (A,B,A)", {{{A, 0, 4}}, {{B, 0, 4}}, {{A, 0, 4}}, {}},
         {{0}, {4}, {0}, {}}},
        // (B,A,A,B): blocks 1 and 2 defer to block 0; block 3 shares its CTA.
        {"same-cycle (B,A,A,B)", {{{B, 0, 4}}, {{A, 0, 4}}, {{A, 0, 4}}, {{B, 0, 4}}},
         {{0}, {4}, {4}, {0}}},
        // One CTA on all blocks starts together.
        {"same-cycle (A,A,A,A)", {{{A, 0, 4}}, {{A, 0, 4}}, {{A, 0, 4}}, {{A, 0, 4}}},
         {{0}, {0}, {0}, {0}}},
        // A higher block mid-expansion is not deferred by a lower block that
        // presents a first micro-op of another CTA; the lower one waits.
        {"mid-expansion vs lower leader", {{{A, 2, 4}}, {}, {{B, 0, 6}}, {}},
         {{6}, {}, {0}, {}}},
        // A lower block deferred by a block mid-expansion is no leader.
        {"deferred lower block", {{{B, 2, 4}}, {{A, 2, 4}}, {}, {{A, 0, 6}}},
         {{6}, {2}, {}, {0}}},
        // Back-to-back WGMMAs of two CTAs alternating on two blocks.
        {"alternating", {{{A, 0, 3}, {B, 0, 3}}, {{B, 0, 3}, {A, 0, 3}}, {}, {}},
         {{0, 3}, {3, 6}, {}, {}}},
    };

    for (auto& c : cases) {
        Block blk[NB];
        for (int b = 0; b < NB; ++b) blk[b].prog = c.prog[b];
        bool ok = run(blk, c.name, 100, rng, false) && expect(blk, c.name, c.starts);
        printf("%s: %s\n", c.name, ok ? "ok" : "FAILED");
        ok ? ++passed : ++failed;
    }

    // Random: up to 3 CTAs, random lengths, gaps and downstream stalls.
    int rnd_fail = 0;
    for (int it = 0; it < 300; ++it) {
        Block blk[NB];
        for (int b = 0; b < NB; ++b) {
            int t = int(rng() % 4);
            int n = 1 + int(rng() % 6);
            for (int i = 0; i < n; ++i) {
                t += int(rng() % 6);
                blk[b].prog.push_back({1 + int(rng() % 3), t, 1 + int(rng() % 8)});
            }
        }
        char name[32];
        snprintf(name, sizeof(name), "random %d", it);
        if (!run(blk, name, 30 + int(rng() % 71), rng, true)) ++rnd_fail;
    }
    printf("random: %d of 300 runs failed\n", rnd_fail);
    rnd_fail ? ++failed : ++passed;

    top->final();
    delete top;
    if (failed) {
        printf("FAILED! (%d of %d checks)\n", failed, failed + passed);
        return 1;
    }
    printf("PASSED!\n");
    return 0;
}
