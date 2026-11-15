// DramSim host test: the DRAM path alone (wrapper + Ramulator), in SM cycles at the build's clock pair.
// Device numbers are computed from the toml knobs (-D flags) and the generated vx_dram_config.h.
// Cases: SEC, Z-RD, Z-WR, T2, T3a, T3b, T4, T5, T6 (print only), T7 (print only), AGE, Q, ORD, CLK, MAP,
// LEG (HBM2 build); SPEED (argument SPEED: wall time of every channel saturated).

#include <dram_sim.h>
#include <vx_dram_config.h>
#include <vx_dram_map.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace vortex;

namespace {

constexpr uint64_t SM_MHZ   = VX_CFG_SM_CLOCK_MHZ;
constexpr uint64_t DRAM_MHZ = VX_CFG_DRAM_CLOCK_MHZ;
constexpr uint32_t LINE     = VX_CFG_DRAM_LINE_BYTES;
constexpr uint32_t ATOM     = VX_CFG_DRAM_ATOM_BYTES;
constexpr uint32_t ATOMS    = LINE / ATOM;
constexpr uint32_t CHANNELS = VX_CFG_DRAM_CHANNELS;
#ifdef VX_CFG_DRAM_MODEL_H100
constexpr bool H100 = true;
constexpr uint32_t SYS_CHANNELS = VX_CFG_DRAM_CHANNELS / VX_CFG_NUM_PARTITIONS;
constexpr uint32_t NUM_SYSTEMS = VX_CFG_NUM_PARTITIONS;
#else
constexpr bool H100 = false;
constexpr uint32_t SYS_CHANNELS = VX_CFG_DRAM_CHANNELS;
constexpr uint32_t NUM_SYSTEMS = 1;
#endif
constexpr uint64_t LINES_PER_CHANNEL = (uint64_t)VX_CFG_GLOBAL_MEM_SIZE / CHANNELS / LINE;

// A request sent in SM cycle t enters Ramulator in the DRAM ticks of cycle t + 1; the response is visible after
// the SM tick whose DRAM tick completed it.
constexpr uint64_t WRAPPER_STAGES = 1;

constexpr uint32_t log2u(uint64_t x) { return x <= 1 ? 0 : 1 + log2u(x >> 1); }

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char* name, const std::string& what) {
  ++g_checks;
  printf("%s: %s: %s\n", ok ? "PASS" : "FAIL", name, what.c_str());
  if (!ok)
    ++g_failures;
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  vsnprintf(buf, sizeof(buf), f, ap);
  va_end(ap);
  return buf;
}

// --- device numbers from the knobs -------------------------------------------------------------------------------

constexpr uint64_t TCK_PS = (2000000 + DRAM_MHZ - 1) / DRAM_MHZ;
constexpr uint64_t nck(uint64_t ps) { return (ps + TCK_PS - 1) / TCK_PS; }
// Ramulator's HBM3 ticks at half CK, i.e. at the DRAM clock.
constexpr uint64_t ck_ticks(uint64_t ck) { return 2 * ck; }

#ifdef VX_CFG_DRAM_MODEL_H100
uint64_t ticks_to_sm(uint64_t ticks) { return (ticks * SM_MHZ + DRAM_MHZ - 1) / DRAM_MHZ; }
double ps_to_sm(double ps) { return ps * SM_MHZ / 1e6; }
double sm_to_ps(double sm) { return sm * 1e6 / SM_MHZ; }

// Visit interval of the controller's service limiter (contract §3 "Visit intervals"), in ps.
double occ_ps(bool write, uint32_t k) {
  double floor_ps = write ? (k == 1 ? VX_CFG_H100_DRAM_VISIT_WR1_PS : VX_CFG_H100_DRAM_VISIT_WR2_PS)
                          : VX_CFG_H100_DRAM_VISIT_RD_PS;
  double rate_mbps = write ? VX_CFG_H100_SLICE_WR_MBPS : VX_CFG_H100_SLICE_RD_MBPS;
  double data_ps = k * double(ATOM) * 1e6 / rate_mbps;
  return double(VX_CFG_DRAM_SERVICE_SCALE_NUM) / VX_CFG_DRAM_SERVICE_SCALE_DEN * std::max(floor_ps, data_ps);
}

// One channel's rate at the exact share (the SLICE_*_MBPS knobs are floored to whole MB/s).
double slice_mbps(bool write) {
  return double(write ? VX_CFG_H100_SLICE_WR_MBPS : VX_CFG_H100_SLICE_RD_MBPS) * VX_CFG_DRAM_SERVICE_SCALE_DEN /
         VX_CFG_DRAM_SERVICE_SCALE_NUM;
}

uint64_t zero_rd_expect(uint32_t sectors) {
  uint64_t t = VX_DRAM_NRCDRD_TICKS + VX_DRAM_NCCDS_TICKS * (sectors - 1) + VX_DRAM_READ_LATENCY_TICKS;
  return ticks_to_sm(t) + VX_CFG_DRAM_MISS_PREMIUM + WRAPPER_STAGES;
}

uint64_t zero_wr_expect() {
  return ticks_to_sm(VX_DRAM_NRCDWR_TICKS + VX_DRAM_WRITE_DONE_TICKS) + WRAPPER_STAGES;
}
#endif

// --- golden map ---------------------------------------------------------------------------------------------------

const uint64_t g_hb[] = VX_DRAM_HB_H100;
const vx_dram_geom_t g_geom = {log2u(VX_CFG_DRAM_ROW_BYTES / LINE), log2u(VX_CFG_DRAM_BANKS), g_hb};

#ifdef VX_CFG_DRAM_MODEL_H100
vx_dram_loc_t map_lc(uint64_t lc) { return vx_dram_map(lc, &g_geom); }

uint32_t mask_of(uint32_t first, uint32_t n) { return ((1u << n) - 1) << first; }
#endif

// --- harness ------------------------------------------------------------------------------------------------------

struct Inflight {
  uint32_t ch;
  bool write;
  uint32_t bytes;
  uint64_t sent;
  uint64_t lc;
  uint32_t mask;
};

struct Done {
  uint64_t tag;
  uint32_t ch;
  bool write;
  uint32_t bytes;
  uint64_t sent;
  uint64_t done;
};

class Harness {
public:
  explicit Harness(bool report_writes = true, uint32_t first_channel = 0)
    : dram_(DramSim::Config{SYS_CHANNELS, first_channel, report_writes}) {
    dram_.reset();
  }

  DramSim& dram() { return dram_; }
  uint64_t now() const { return now_; }

  bool can(uint32_t ch, bool write) {
    return dram_.can_accept(ch, write);
  }

  uint64_t send(uint32_t ch, uint64_t lc, uint32_t mask, bool write) {
    uint64_t tag = next_tag_++;
    uint32_t first = __builtin_ctz(mask);
    DramSim::Req req{ch, lc * LINE + first * ATOM, mask, write, tag};
    dram_.send(req);
    inflight_[tag] = Inflight{ch, write, uint32_t(__builtin_popcount(mask)) * ATOM, now_, lc, mask};
    sent_order_[ch].push_back(tag);
    return tag;
  }

  void tick() {
    dram_.tick();
    ++now_;
    for (uint32_t ch = 0; ch < SYS_CHANNELS; ++ch) {
      while (dram_.rsp_valid(ch)) {
        auto rsp = dram_.rsp_peek(ch);
        dram_.rsp_pop(ch);
        auto it = inflight_.find(rsp.tag);
        if (it == inflight_.end()) {
          ++unknown_rsps_;
          continue;
        }
        if (it->second.write != rsp.write || it->second.ch != ch)
          ++bad_rsps_;
        done_.push_back(Done{rsp.tag, ch, rsp.write, it->second.bytes, it->second.sent, now_});
        inflight_.erase(it);
      }
    }
  }

  void run(uint64_t cycles) {
    for (uint64_t i = 0; i < cycles; ++i)
      tick();
  }

  // Ticks until `tag` completes; returns its latency.
  uint64_t wait(uint64_t tag) {
    while (inflight_.count(tag))
      tick();
    for (auto it = done_.rbegin(); it != done_.rend(); ++it)
      if (it->tag == tag)
        return it->done - it->sent;
    return 0;
  }

  uint64_t latency(uint64_t tag) const {
    for (auto& d : done_)
      if (d.tag == tag)
        return d.done - d.sent;
    return UINT64_MAX;
  }

  std::vector<Done>& done() { return done_; }
  size_t inflight() const { return inflight_.size(); }
  uint64_t unknown_rsps() const { return unknown_rsps_; }
  uint64_t bad_rsps() const { return bad_rsps_; }
  const std::map<uint32_t, std::vector<uint64_t>>& sent_order() const { return sent_order_; }

private:
  DramSim dram_;
  uint64_t now_ = 0;
  uint64_t next_tag_ = 1;
  std::unordered_map<uint64_t, Inflight> inflight_;
  std::vector<Done> done_;
  std::map<uint32_t, std::vector<uint64_t>> sent_order_;
  uint64_t unknown_rsps_ = 0;
  uint64_t bad_rsps_ = 0;
};

std::mt19937_64 g_rng(17);

uint64_t rand_lc() { return g_rng() % LINES_PER_CHANNEL; }

template <typename T>
T median(std::vector<T> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

#ifdef VX_CFG_DRAM_MODEL_H100
// Lines never touched before in this process, so every access opens its row.
std::set<uint64_t> g_used_rows;

uint64_t fresh_lc() {
  for (;;) {
    uint64_t lc = rand_lc();
    auto loc = map_lc(lc);
    uint64_t key = (loc.row << 8) | loc.bank;
    if (g_used_rows.insert(key).second)
      return lc;
  }
}

enum class Rel { CONFLICT, SAMEROW, OTHERBANK };

// A second line in the same channel with the given relation to `a`.
uint64_t partner(uint64_t a, Rel rel) {
  auto la = map_lc(a);
  uint64_t row_lines = uint64_t(1) << (g_geom.lg_lines_per_row + g_geom.lg_banks);
  for (int i = 0; i < 1000000; ++i) {
    uint64_t b;
    if (rel == Rel::SAMEROW)
      b = (a & ~(row_lines - 1)) | (g_rng() & (row_lines - 1));
    else
      b = fresh_lc();
    if (b == a)
      continue;
    auto lb = map_lc(b);
    bool ok = false;
    switch (rel) {
    case Rel::CONFLICT:  ok = lb.bank == la.bank && lb.row != la.row; break;
    case Rel::SAMEROW:   ok = lb.bank == la.bank && lb.row == la.row && lb.col != la.col; break;
    case Rel::OTHERBANK: ok = lb.bank != la.bank; break;
    }
    if (ok) {
      g_used_rows.insert((lb.row << 8) | lb.bank);
      return b;
    }
  }
  fprintf(stderr, "error: no partner line found\n");
  exit(1);
}
#endif

// --- cases --------------------------------------------------------------------------------------------------------

void case_sector() {
  // byteen bit i = byte i of the 64-B port word at addr.
  bool ok = DramSim::sector_mask(0, 64, ~0ull) == 0x3 && DramSim::sector_mask(64, 64, ~0ull) == 0xc
         && DramSim::sector_mask(0, 64, 0xffffffff00000000ull) == 0x2 && DramSim::sector_mask(64, 64, 0xffull) == 0x4
         && DramSim::sector_mask(LINE * 5 + 64, 64, 0x0000ffff00000000ull) == 0x8;
  check(ok, "SEC", "sector masks of 64-B port words (reads all ones, partial writes)");
}

#ifdef VX_CFG_DRAM_MODEL_H100

// The generated device timings against the knobs: ps rounded up to whole CK at the DRAM clock, two ticks per CK.
void case_gen() {
  struct { const char* name; uint64_t got, want; } t[] = {
    {"nRCDRD", VX_DRAM_NRCDRD_TICKS, ck_ticks(nck(VX_CFG_DRAM_T_RCDRD_PS))},
    {"nRCDWR", VX_DRAM_NRCDWR_TICKS, ck_ticks(nck(VX_CFG_DRAM_T_RCDWR_PS))},
    {"nRC", VX_DRAM_NRC_TICKS, ck_ticks(nck(VX_CFG_DRAM_T_RC_PS))},
    {"nRP", VX_DRAM_NRP_TICKS, ck_ticks(nck(VX_CFG_DRAM_T_RP_PS))},
    {"nRAS", VX_DRAM_NRAS_TICKS, ck_ticks(nck(VX_CFG_DRAM_T_RC_PS) - nck(VX_CFG_DRAM_T_RP_PS))},
    {"nCL", VX_DRAM_NCL_TICKS, ck_ticks(nck(VX_CFG_DRAM_T_CL_PS))},
    {"nCWL", VX_DRAM_NCWL_TICKS, ck_ticks(nck(VX_CFG_DRAM_T_CWL_PS))},
    {"nBL", VX_DRAM_NBL_TICKS, ck_ticks(VX_CFG_DRAM_NCK_BL)},
    {"nCCDS", VX_DRAM_NCCDS_TICKS, ck_ticks(VX_CFG_DRAM_NCK_CCDS)},
    {"read latency", VX_DRAM_READ_LATENCY_TICKS, ck_ticks(nck(VX_CFG_DRAM_T_CL_PS) + VX_CFG_DRAM_NCK_BL)},
    {"write done", VX_DRAM_WRITE_DONE_TICKS, ck_ticks(nck(VX_CFG_DRAM_T_CWL_PS) + VX_CFG_DRAM_NCK_BL)},
  };
  for (auto& x : t)
    check(x.got == x.want, "GEN", fmt("%s %lu ticks, knob gives %lu", x.name, x.got, x.want));
  struct { const char* name; uint64_t got; bool write; uint32_t k; } o[] = {
    {"occ rd2", VX_DRAM_OCC_RD2_MILLITICKS, false, 2}, {"occ rd4", VX_DRAM_OCC_RD4_MILLITICKS, false, 4},
    {"occ wr1", VX_DRAM_OCC_WR1_MILLITICKS, true, 1}, {"occ wr2", VX_DRAM_OCC_WR2_MILLITICKS, true, 2},
    {"occ wr4", VX_DRAM_OCC_WR4_MILLITICKS, true, 4},
  };
  for (auto& x : o) {
    double want = occ_ps(x.write, x.k) * DRAM_MHZ / 1000.0;
    check(std::abs(double(x.got) - want) <= 1.0, "GEN", fmt("%s %lu milliticks, knobs give %.1f", x.name, x.got, want));
  }
}

uint64_t zero_latency(uint32_t sectors, bool write, int samples = 63) {
  Harness h;
  std::vector<uint64_t> lat;
  for (int i = 0; i < samples; ++i) {
    auto tag = h.send(0, fresh_lc(), mask_of(0, sectors), write);
    lat.push_back(h.wait(tag));
    h.run(400);
  }
  return median(lat);
}

uint64_t g_zrd64 = 0;
uint64_t g_zrd128 = 0;

void case_zero() {
  g_zrd64 = zero_latency(2, false);
  g_zrd128 = zero_latency(4, false);
  uint64_t e = zero_rd_expect(2);
  check(g_zrd64 + 2 >= e && g_zrd64 <= e + 2, "Z-RD",
        fmt("64-B read at idle %lu SM cycles, expected %lu +-2 (premium %d)", g_zrd64, e, VX_CFG_DRAM_MISS_PREMIUM));
  uint64_t w = zero_latency(1, true);
  uint64_t ew = zero_wr_expect();
  check(w + 2 >= ew && w <= ew + 2, "Z-WR", fmt("32-B write at idle %lu SM cycles, expected %lu +-2", w, ew));
}

// Second completion of a pair sent back to back (one SM cycle apart: one skid entry per direction).
uint64_t pair_second(Harness& h, uint64_t a, uint64_t b) {
  uint64_t t0 = h.now();
  auto ta = h.send(0, a, 0x3, false);
  do {
    h.tick();
  } while (!h.can(0, false));
  uint64_t tb_sent = h.now();
  auto tb = h.send(0, b, 0x3, false);
  h.wait(ta);
  h.wait(tb);
  return std::max(t0 + h.latency(ta), tb_sent + h.latency(tb)) - t0;
}

void case_pairs() {
  Harness h;
  const int trials = 31;
  std::vector<uint64_t> conflict, samerow, other;
  for (int i = 0; i < trials; ++i) {
    for (Rel rel : {Rel::CONFLICT, Rel::SAMEROW, Rel::OTHERBANK}) {
      uint64_t a = fresh_lc();
      uint64_t b = partner(a, rel);
      uint64_t t = pair_second(h, a, b);
      (rel == Rel::CONFLICT ? conflict : rel == Rel::SAMEROW ? samerow : other).push_back(t);
      h.run(600);
    }
  }
  int64_t mc = median(conflict), ms = median(samerow), mo = median(other);
  printf("PERF: pair conflict=%ld samerow=%ld otherbank=%ld (median second completion, SM cycles)\n", mc, ms, mo);
  check(mc - mo >= 60 && mc - mo <= 80, "T2", fmt("conflict - otherbank = %ld, expected 70 +-10", mc - mo));
  check(ms - mo <= 5, "T3a", fmt("samerow - otherbank = %ld, expected <= +5", ms - mo));

  // sequential: the second request is sent after the first returns
  std::vector<int64_t> seq_conf, seq_other;
  for (int i = 0; i < trials; ++i) {
    for (bool conf : {true, false}) {
      uint64_t a = fresh_lc();
      uint64_t b = partner(a, conf ? Rel::CONFLICT : Rel::OTHERBANK);
      h.wait(h.send(0, a, 0x3, false));
      int64_t l = h.wait(h.send(0, b, 0x3, false));
      (conf ? seq_conf : seq_other).push_back(l);
      h.run(600);
    }
  }
  int64_t d = median(seq_conf) - median(seq_other);
  check(d >= -20 && d <= 20, "T3b", fmt("sequential same-bank - other-bank = %ld, expected 0 +-20", d));
}

struct Rate {
  double mbps_per_ch;
  double interval_ps;
  std::vector<double> per_ch;
  uint64_t max_rd_q, max_wr_q;
  uint64_t span;  // SM cycles of channel 0's window
  uint64_t total0;  // channel 0 completions since cycle 0
  uint64_t last0;   // cycle of channel 0's last completion
};

// Saturated stream of `sectors`-sector requests on channels [0, nch): send whenever the channel accepts.
Rate saturate(uint32_t nch, uint32_t sectors, bool write, uint64_t warm = 20000, uint64_t win = 100000) {
  Harness h;
  std::vector<uint64_t> bytes(nch, 0), count(nch, 0);
  uint64_t warm0 = 0;
  bool skid_ok = true;
  for (uint64_t c = 0; c < warm + win; ++c) {
    for (uint32_t ch = 0; ch < nch; ++ch) {
      if (h.can(ch, write)) {
        h.send(ch, rand_lc(), mask_of(0, sectors), write);
        skid_ok &= !h.can(ch, write);
      }
    }
    h.tick();
    if (c + 1 == warm) {
      for (auto& d : h.done())
        warm0 += (d.ch == 0);
      h.done().clear();
    }
  }
  // Between a channel's first and last completion of the window: n completions span n - 1 intervals.
  std::vector<uint64_t> first(nch, UINT64_MAX), last(nch, 0);
  for (auto& d : h.done()) {
    if (d.done < first[d.ch]) {
      first[d.ch] = d.done;
    } else {
      bytes[d.ch] += d.bytes;
      count[d.ch] += 1;
    }
    last[d.ch] = std::max(last[d.ch], d.done);
  }
  Rate r{};
  for (uint32_t ch = 0; ch < nch; ++ch)
    r.per_ch.push_back(bytes[ch] * 1e6 / sm_to_ps(double(last[ch] - first[ch])));
  r.mbps_per_ch = r.per_ch[0];
  r.span = last[0] - first[0];
  r.total0 = warm0 + count[0] + 1;
  r.last0 = last[0];
  r.interval_ps = sm_to_ps(double(r.span)) / double(count[0]);
  for (uint32_t ch = 0; ch < nch; ++ch) {
    auto& s = h.dram().stats(ch);
    r.max_rd_q = std::max<uint64_t>(r.max_rd_q, s.max_rd_queue);
    r.max_wr_q = std::max<uint64_t>(r.max_wr_q, s.max_wr_queue);
  }
  check(skid_ok, "Q", fmt("can_accept false right after every send (%u ch, %s)", nch, write ? "writes" : "reads"));
  check(r.max_rd_q <= VX_CFG_DRAM_QD_RD && r.max_wr_q <= VX_CFG_DRAM_QD_WR, "Q",
        fmt("max queue rd %lu <= %d, wr %lu <= %d", r.max_rd_q, VX_CFG_DRAM_QD_RD, r.max_wr_q, VX_CFG_DRAM_QD_WR));
  return r;
}

void case_rates() {
  for (bool write : {false, true}) {
    double target = slice_mbps(write);
    Rate one = saturate(1, 4, write);
    double ratio = one.mbps_per_ch / target;
    printf("PERF: T4 %s ch=1 rate_mbps=%.0f target=%.0f ratio=%.4f\n", write ? "wr" : "rd", one.mbps_per_ch, target, ratio);
    check(ratio >= 0.95, "T4", fmt("one channel %s %.0f MB/s = %.4f x %.0f", write ? "write" : "read", one.mbps_per_ch,
                                   ratio, target));
    // A window's rate can exceed the server's by its edge visits' latency difference (FR-FCFS reorders the queue), so the
    // ceiling is checked from cycle 0: visit n starts (n - 1) server intervals after the first and completes after it.
    double occ4 = double(write ? VX_DRAM_OCC_WR4_MILLITICKS : VX_DRAM_OCC_RD4_MILLITICKS) * 1000 / DRAM_MHZ;
    double need_ps = double(one.total0 - 1) * occ4;
    check(need_ps <= sm_to_ps(double(one.last0)), "T4",
          fmt("one channel %s: %lu visits by cycle %lu need %.0f ps of visit server, %.0f elapsed", write ? "write" : "read",
              one.total0, one.last0, need_ps, sm_to_ps(double(one.last0))));
    for (uint32_t n = 2; n <= std::min<uint32_t>(8, SYS_CHANNELS); n *= 2) {
      Rate m = saturate(n, 4, write);
      double worst = 0;
      for (double v : m.per_ch)
        worst = std::max(worst, std::abs(v / one.mbps_per_ch - 1));
      printf("PERF: T4 %s ch=%u worst_dev=%.4f\n", write ? "wr" : "rd", n, worst);
      check(worst <= 0.01, "T4", fmt("%u channels: every channel within 1 %% of one (worst %.4f)", n, worst));
    }
  }
  struct { bool write; uint32_t k; const char* name; } t5[] = {{false, 2, "rd64"}, {true, 1, "wr32"}, {true, 2, "wr64"}};
  for (auto& t : t5) {
    Rate r = saturate(1, t.k, t.write);
    double e = occ_ps(t.write, t.k);
    double dev = r.interval_ps / e - 1;
    printf("PERF: T5 %s interval_ps=%.0f occ_ps=%.0f\n", t.name, r.interval_ps, e);
    check(std::abs(dev) <= 0.03, "T5", fmt("%s visit interval %.0f ps vs occ %.0f ps (%.2f %%)", t.name, r.interval_ps, e,
                                            100 * dev));
  }
}

// In-order source of 128-B requests, write slots spread over 8 (hbmsim's OP_MIX rule); one channel.
double mix_rate(uint32_t num, uint32_t den, uint64_t limit, uint64_t warm, uint64_t win, uint64_t* switches) {
  Harness h;
  std::deque<bool> q;
  uint64_t slot = 0, made = 0;
  uint32_t nw = 8 * num / den;
  // Every share sees one address stream, whatever ran before: a ratio of two batches then measures the mix alone.
  g_rng.seed(17);
  uint64_t end = limit ? UINT64_MAX : warm + win;
  for (uint64_t c = 0; c < end; ++c) {
    while (q.size() < 256 && (!limit || made < limit)) {
      uint64_t s = slot++ % 8;
      q.push_back(nw && (s * nw) % 8 < nw);
      ++made;
    }
    if (!q.empty() && h.can(0, q.front())) {
      h.send(0, rand_lc(), 0xf, q.front());
      q.pop_front();
    }
    h.tick();
    if (!limit && c + 1 == warm)
      h.done().clear();
    if (limit && q.empty() && h.inflight() == 0)
      break;
  }
  *switches = h.dram().stats(0).dir_switches;
  uint64_t bytes = 0;
  for (auto& d : h.done())
    bytes += d.bytes;
  double t_ps = sm_to_ps(double(limit ? h.now() : win));
  return bytes * 1e6 / t_ps;
}

void case_t6() {
  const uint32_t shares[][2] = {{0, 8}, {1, 8}, {2, 8}, {4, 8}, {6, 8}, {7, 8}, {8, 8}};
  for (int shallow = 0; shallow < 2; ++shallow) {
    double base = 0;
    for (auto& s : shares) {
      uint64_t sw = 0;
      double r = shallow ? mix_rate(s[0], s[1], 256 * 1024 / LINE, 0, 0, &sw) : mix_rate(s[0], s[1], 0, 20000, 160000, &sw);
      if (base == 0)
        base = r;
      printf("PERF: T6 %s share=%.3f ratio=%.4f rate_mbps=%.0f switches=%lu\n", shallow ? "shallow" : "deep",
             double(s[0]) / s[1], r / base, r, sw);
    }
  }
}

void case_t7() {
  const int loads[] = {45, 63, 78, 94, 100};
  const uint64_t n_req = 20000;
  for (int load : loads) {
    Harness h;
    // one Bernoulli trial per SM cycle: random arrivals, as a slice sees from many SMs
    double p_req = load / 100.0 * slice_mbps(false) / double(SM_MHZ) / LINE;
    std::bernoulli_distribution arrive(p_req);
    std::mt19937_64 rng(23);
    uint64_t made = 0;
    std::deque<uint64_t> q;  // generation cycles
    std::unordered_map<uint64_t, uint64_t> gen_of;
    while (made < n_req || !q.empty() || h.inflight()) {
      if (made < n_req && arrive(rng)) {
        q.push_back(h.now());
        ++made;
      }
      if (!q.empty() && h.can(0, false)) {
        gen_of[h.send(0, rand_lc(), 0xf, false)] = q.front();
        q.pop_front();
      }
      h.tick();
    }
    std::vector<int64_t> lat;
    for (auto& d : h.done())
      if (d.tag > n_req / 10)
        lat.push_back(int64_t(d.done - gen_of[d.tag]));
    int64_t over = median(lat) - int64_t(g_zrd128);
    printf("PERF: T7 load=%d median_over_zero=%ld (SM cycles; zero-load 128-B read %lu)\n", load, over, g_zrd128);
  }
}

void case_age() {
  Harness h;
  uint64_t rd = 0;
  for (uint64_t c = 0; c < 200000; ++c) {
    if (h.can(0, true))
      h.send(0, rand_lc(), 0xf, true);
    if (c == 20000)
      rd = h.send(0, fresh_lc(), 0x3, false);
    h.tick();
    if (rd && h.latency(rd) != UINT64_MAX)
      break;
  }
  uint64_t l = h.latency(rd);
  double e = ps_to_sm(VX_CFG_DRAM_T_AGE_PS + VX_CFG_DRAM_T_TURN_PS + occ_ps(true, 4)) + double(g_zrd64);
  double dev = double(l) / e - 1;
  check(std::abs(dev) <= 0.05, "AGE", fmt("read under a saturating write stream: %lu SM cycles, expected %.0f (%.1f %%)",
                                          l, e, 100 * dev));
  check(h.dram().stats(0).dir_switches > 0, "AGE", fmt("dir_switches %lu > 0", h.dram().stats(0).dir_switches));
}

void case_order() {
  Harness h;
  // a few rows per bank, so row hits queue behind conflicts
  std::vector<uint64_t> pool;
  for (int i = 0; i < 64; ++i)
    pool.push_back(rand_lc());
  uint64_t sent = 0;
  std::set<uint64_t> tags;
  while (sent < 10000) {
    uint32_t ch = g_rng() % SYS_CHANNELS;
    bool write = (g_rng() % 4) == 0;
    uint32_t n = 1u << (g_rng() % 3);  // 1, 2, 4 sectors
    uint32_t first = (g_rng() % (ATOMS / n)) * n;
    uint64_t lc = pool[g_rng() % pool.size()] ^ (g_rng() % 4);
    if (h.can(ch, write)) {
      tags.insert(h.send(ch, lc % LINES_PER_CHANNEL, mask_of(first, n), write));
      ++sent;
    }
    h.tick();
  }
  while (h.inflight())
    h.tick();
  std::map<uint64_t, int> seen;
  for (auto& d : h.done())
    seen[d.tag]++;
  bool once = seen.size() == tags.size() && h.unknown_rsps() == 0 && h.bad_rsps() == 0;
  for (auto& s : seen)
    once &= s.second == 1 && tags.count(s.first);
  check(once, "ORD", fmt("%zu requests, every tag exactly once (%zu seen)", tags.size(), seen.size()));
  uint64_t ooo = 0;
  for (uint32_t ch = 0; ch < SYS_CHANNELS; ++ch)
    ooo += h.dram().stats(ch).ooo_rsps;
  // ooo_rsps must agree with what the test observed: responses popped out of send order.
  uint64_t seen_ooo = 0;
  std::map<uint32_t, std::vector<uint64_t>> got;
  for (auto& d : h.done())
    got[d.ch].push_back(d.tag);
  for (auto& kv : got) {
    auto& order = h.sent_order().at(kv.first);
    std::set<uint64_t> popped;
    size_t head = 0;
    for (auto tag : kv.second) {
      while (head < order.size() && popped.count(order[head]))
        ++head;
      if (order[head] != tag)
        ++seen_ooo;
      popped.insert(tag);
    }
  }
  check(ooo > 0 && seen_ooo > 0, "ORD", fmt("completion order: ooo_rsps %lu, observed %lu out of send order", ooo, seen_ooo));
}

void case_map() {
  Harness h;
  struct Expect {
    uint32_t ch;
    std::vector<vx_dram_fields_t> f;
    uint32_t cmds;
  };
  std::unordered_map<uint64_t, Expect> exp;
  uint64_t bad = 0, cmds = 0;
  // fields: Ramulator's address vector (Channel, PseudoChannel, Sid, BankGroup, Bank, Row, Column)
  h.dram().set_issue_observer([&](uint32_t channel, const char* cmd, const uint32_t fields[7], uint64_t tag) {
    std::string c(cmd);
    if (c != "RD" && c != "RDA" && c != "WR" && c != "WRA")
      return;
    ++cmds;
    auto it = exp.find(tag);
    if (it == exp.end()) {
      ++bad;
      return;
    }
    it->second.cmds++;
    bool hit = false;
    for (auto& f : it->second.f)
      hit |= f.pc == fields[1] && fields[2] == 0 && f.bg == fields[3] && f.ba == fields[4] && f.row == fields[5]
          && f.column == fields[6];
    if (!hit || fields[0] != it->second.ch || channel != it->second.ch) {
      if (bad < 5)
        printf("  MAP mismatch: tag %lu ch %u cmd %s fields %u %u %u %u %u %u %u\n", tag, channel, cmd, fields[0],
               fields[1], fields[2], fields[3], fields[4], fields[5], fields[6]);
      ++bad;
    }
  });
  const uint32_t lg_line = log2u(LINE), lg_ch = log2u(CHANNELS);
  uint64_t sent = 0;
  std::vector<uint64_t> tags;
  while (sent < 10000) {
    uint64_t addr = (g_rng() % (uint64_t(VX_CFG_GLOBAL_MEM_SIZE) / ATOM)) * ATOM;
    uint32_t ch = vx_dram_interim_channel(addr, lg_line, lg_ch);
    if (ch >= SYS_CHANNELS)
      continue;
    uint64_t local = vx_dram_interim_local(addr, lg_line, lg_ch);
    uint64_t lc = local / LINE;
    uint32_t sector = (local % LINE) / ATOM;
    uint32_t n = 1u << (g_rng() % 3);
    uint32_t first = sector & ~(n - 1);
    bool write = g_rng() & 1;
    while (!h.can(ch, write))
      h.tick();
    Expect e{ch, {}, 0};
    auto loc = map_lc(lc);
    for (uint32_t s = first; s < first + n; ++s)
      e.f.push_back(vx_dram_fields(loc, s, log2u(ATOMS), log2u(VX_CFG_DRAM_PCS), log2u(VX_CFG_DRAM_BGS)));
    // the tag the harness assigns next
    uint64_t tag = h.send(ch, lc, mask_of(first, n), write);
    exp[tag] = e;
    tags.push_back(tag);
    ++sent;
    h.tick();
  }
  while (h.inflight())
    h.tick();
  h.run(1000);
  uint64_t short_cmds = 0;
  for (auto t : tags) {
    auto& e = exp[t];
    if (e.cmds != e.f.size())
      ++short_cmds;
  }
  check(bad == 0 && short_cmds == 0 && cmds > 0, "MAP",
        fmt("%lu RD/WR commands, %lu off the golden map, %lu requests with a wrong command count", cmds, bad, short_cmds));
}

#endif  // VX_CFG_DRAM_MODEL_H100

void case_clock() {
  Harness h;
  const uint64_t n = 100;
  h.run(SM_MHZ * n);
  uint64_t t = h.dram().stats(0).dram_ticks;
  check(t == DRAM_MHZ * n, "CLK", fmt("%lu SM ticks -> %lu DRAM ticks, expected %lu", SM_MHZ * n, t, DRAM_MHZ * n));
}

#ifndef VX_CFG_DRAM_MODEL_H100
void case_legacy() {
  {
    Harness h;
    const uint64_t warm = 5000, win = 50000;
    for (uint64_t c = 0; c < warm + win; ++c) {
      for (uint32_t ch = 0; ch < SYS_CHANNELS; ++ch)
        if (h.can(ch, false))
          h.send(ch, rand_lc(), 0x3, false);
      h.tick();
      if (c + 1 == warm)
        h.done().clear();
    }
    uint64_t bytes = 0;
    for (auto& d : h.done())
      bytes += d.bytes;
    double bpc = double(bytes) / win;
    printf("PERF: LEG rd64 bytes_per_cycle=%.2f\n", bpc);
    check(bpc > 16.0, "LEG", fmt("%u-channel 64-B read stream moves %.2f B per SM cycle (> 16)", SYS_CHANNELS, bpc));
  }
  uint64_t l32, l64;
  {
    Harness h;
    l32 = h.wait(h.send(0, 1000, 0x1, false));
  }
  {
    Harness h;
    l64 = h.wait(h.send(0, 1000, 0x3, false));
  }
  check(l64 > l32, "LEG", fmt("a 64-B read completes after its last sector (%lu > 32-B %lu)", l64, l32));
  {
    Harness h;
    std::vector<uint64_t> tags;
    for (int i = 0; i < 64; ++i) {
      while (!h.can(i % SYS_CHANNELS, true))
        h.tick();
      tags.push_back(h.send(i % SYS_CHANNELS, rand_lc(), 0x3, true));
      h.tick();
    }
    while (h.inflight())
      h.tick();
    check(h.done().size() == tags.size(), "LEG", fmt("%zu of %zu writes completed", h.done().size(), tags.size()));
  }
}
#endif

void case_speed() {
  std::vector<std::unique_ptr<Harness>> sys;
  for (uint32_t p = 0; p < NUM_SYSTEMS; ++p)
    sys.emplace_back(new Harness(true, p * SYS_CHANNELS));
  const uint64_t cycles = 10000;
  auto t0 = std::chrono::steady_clock::now();
  for (uint64_t c = 0; c < cycles; ++c) {
    for (auto& h : sys) {
      for (uint32_t ch = 0; ch < SYS_CHANNELS; ++ch)
        if (h->can(ch, false))
          h->send(ch, rand_lc(), 0xf, false);
      h->tick();
      h->done().clear();
    }
  }
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("PERF: SPEED channels=%u wall_s_per_1e5_cycles=%.2f\n", CHANNELS, s * 1e5 / cycles);
}

}  // namespace

int main(int argc, char** argv) {
  printf("dram_sim: model %s, %u channels (%u per memory system), SM %lu MHz, DRAM %lu MHz\n", H100 ? "H100" : "HBM2",
         CHANNELS, SYS_CHANNELS, SM_MHZ, DRAM_MHZ);
  std::string only = argc > 1 ? argv[1] : "";
  if (only == "SPEED") {
    case_speed();
    return 0;
  }
  auto want = [&](const char* name) { return only.empty() || only == name; };
  if (want("SEC")) case_sector();
  if (want("CLK")) case_clock();
#ifdef VX_CFG_DRAM_MODEL_H100
  if (want("GEN")) case_gen();
  case_zero();
  if (want("PAIR")) case_pairs();
  if (want("RATE")) case_rates();
  if (want("T6")) case_t6();
  if (want("T7")) case_t7();
  if (want("AGE")) case_age();
  if (want("ORD")) case_order();
  if (want("MAP")) case_map();
#else
  if (want("LEG")) case_legacy();
#endif
  printf("dram_sim: %d checks, %d failed\n", g_checks, g_failures);
  if (g_failures) {
    printf("FAILED!\n");
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
