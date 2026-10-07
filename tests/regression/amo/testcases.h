#pragma once

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <vortex2.h>
#include "common.h"

void cleanup();

#define RT_CHECK(_expr)                                       \
   do {                                                       \
     int _ret = _expr;                                        \
     if (0 == _ret) break;                                    \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
     cleanup();                                               \
     exit(-1);                                                \
   } while (false)

class ITestCase;

class TestSuite {
public:
  TestSuite(vx_device_h device);
  ~TestSuite();

  ITestCase* get_test(int testid) const { return _tests.at(testid); }
  void add_test(ITestCase* t) { _tests.push_back(t); }
  size_t size() const { return _tests.size(); }
  vx_device_h device() const { return device_; }

  uint32_t num_cores() const { return num_cores_; }
  uint32_t num_warps() const { return num_warps_; }
  uint32_t peer_core() const { return peer_core_; }
  uint32_t peer_warp() const { return peer_warp_; }

private:
  std::vector<ITestCase*> _tests;
  vx_device_h device_;
  uint32_t num_cores_ = 0;
  uint32_t num_warps_ = 0;
  uint32_t peer_core_ = 0;
  uint32_t peer_warp_ = 0;
};

class ITestCase {
public:
  ITestCase(TestSuite* suite, const char* name) : suite_(suite), name_(name) {}
  virtual ~ITestCase() {}

  TestSuite* suite() const { return suite_; }
  const char* name() const { return name_; }

  // Initialize the shared word and per-hart buffer to test-specific
  // initial values. `n` is num_harts (size of per_hart buffer in u32s).
  virtual void setup(uint32_t n, uint32_t* shared, uint32_t* per_hart) = 0;

  // Validate the post-run shared word and per-hart buffer.
  // Returns 0 on pass, error count on fail.
  virtual int verify(uint32_t n, uint32_t iters,
                     const uint32_t* shared, const uint32_t* per_hart) = 0;

  // Threads of the one block to launch, or 0 to fill the device with n harts.
  virtual uint32_t block_threads() const { return 0; }

  // Why the device's geometry cannot run this case, or nullptr to run it.
  virtual const char* skip_reason() const { return nullptr; }

protected:
  TestSuite* suite_;
  const char* const name_;
};

// 0) AMOADD hammer: each hart does +1 iter times. Final = n * iters.
class Test_AMOADD : public ITestCase {
public:
  Test_AMOADD(TestSuite* s) : ITestCase(s, "amoadd") {}
  void setup(uint32_t, uint32_t* shared, uint32_t* /*ph*/) override {
    *shared = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* /*ph*/) override {
    uint32_t expected = n * iters;
    if (*shared != expected) {
      std::cout << "  shared=" << *shared << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 1) AMOOR hammer: each hart sets bit (hart_id % 32) idempotently.
//    Final = OR of all such bits = 0xFFFFFFFF for n>=32 (all bits 0..31 hit),
//    else (1 << n) - 1.
class Test_AMOOR : public ITestCase {
public:
  Test_AMOOR(TestSuite* s) : ITestCase(s, "amoor") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override { *shared = 0; }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    uint32_t expected = 0;
    for (uint32_t h = 0; h < n; ++h) expected |= 1u << (h & 31);
    if (*shared != expected) {
      std::cout << "  shared=0x" << std::hex << *shared
                << " expected=0x" << expected << std::dec << std::endl;
      return 1;
    }
    return 0;
  }
};

// 2) AMOAND hammer: start all-ones, each hart clears bit (hart_id % 32).
class Test_AMOAND : public ITestCase {
public:
  Test_AMOAND(TestSuite* s) : ITestCase(s, "amoand") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    *shared = 0xffffffff;
  }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    uint32_t cleared = 0;
    for (uint32_t h = 0; h < n; ++h) cleared |= 1u << (h & 31);
    uint32_t expected = ~cleared;
    if (*shared != expected) {
      std::cout << "  shared=0x" << std::hex << *shared
                << " expected=0x" << expected << std::dec << std::endl;
      return 1;
    }
    return 0;
  }
};

// 3) AMOXOR hammer: each hart toggles its bit twice per iter. Final = initial.
class Test_AMOXOR : public ITestCase {
public:
  Test_AMOXOR(TestSuite* s) : ITestCase(s, "amoxor") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    *shared = 0xa5a5a5a5;
  }
  int verify(uint32_t /*n*/, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    if (*shared != 0xa5a5a5a5) {
      std::cout << "  shared=0x" << std::hex << *shared
                << " expected=0xa5a5a5a5" << std::dec << std::endl;
      return 1;
    }
    return 0;
  }
};

// 4) AMOMAX hammer (signed): each hart writes its hart_id. Final = n - 1.
class Test_AMOMAX : public ITestCase {
public:
  Test_AMOMAX(TestSuite* s) : ITestCase(s, "amomax") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    // INT32_MIN as initial so any unsigned hart_id wins.
    *shared = 0x80000000u;
  }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    int32_t got = (int32_t)*shared;
    int32_t expected = (int32_t)(n - 1);
    if (got != expected) {
      std::cout << "  shared=" << got << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 5) AMOMINU hammer (unsigned): each hart writes its hart_id (≥ 0).
//    Final = 0 (the lowest hart_id always wins).
class Test_AMOMINU : public ITestCase {
public:
  Test_AMOMINU(TestSuite* s) : ITestCase(s, "amominu") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override {
    *shared = 0xffffffffu;
  }
  int verify(uint32_t /*n*/, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    if (*shared != 0) {
      std::cout << "  shared=0x" << std::hex << *shared
                << " expected=0" << std::dec << std::endl;
      return 1;
    }
    return 0;
  }
};

// 6) AMOSWAP exchange: each hart swaps in its hart_id; per-hart obs[]
//    captures last observed-old value. Validate:
//    - final shared is some hart_id (0..n-1) — some hart wrote it last.
//    - every observed value is either initial sentinel or a valid hart_id.
class Test_AMOSWAP : public ITestCase {
public:
  Test_AMOSWAP(TestSuite* s) : ITestCase(s, "amoswap") {}
  void setup(uint32_t n, uint32_t* shared, uint32_t* per_hart) override {
    *shared = 0xdeadbeefu; // sentinel (not a hart_id)
    for (uint32_t i = 0; i < n; ++i) per_hart[i] = 0xdeadbeefu;
  }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t* per_hart) override {
    int errors = 0;
    if (*shared >= n) {
      std::cout << "  final shared=" << *shared
                << " not a valid hart_id (n=" << n << ")" << std::endl;
      ++errors;
    }
    for (uint32_t h = 0; h < n; ++h) {
      uint32_t obs = per_hart[h];
      bool ok = (obs == 0xdeadbeefu) || (obs < n);
      if (!ok) {
        std::cout << "  hart " << h << " observed garbage=0x" << std::hex
                  << obs << std::dec << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

// 7) LR/SC counter (lock-free): final = n * iters.
class Test_LRSC_COUNTER : public ITestCase {
public:
  Test_LRSC_COUNTER(TestSuite* s) : ITestCase(s, "lrsc_counter") {}
  void setup(uint32_t n, uint32_t* shared, uint32_t* per_hart) override {
    *shared = 0;
    for (uint32_t i = 0; i < n; ++i) per_hart[i] = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* /*ph*/) override {
    uint32_t expected = n * iters;
    if (*shared != expected) {
      std::cout << "  shared=" << *shared << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 8) AMOADD.W.AQRL hammer: same expected as Test_AMOADD. Validates
//    the .aqrl encoding path; under SimX's SC ordering the observable
//    result is identical to the plain variant.
class Test_AMOADD_AQRL : public ITestCase {
public:
  Test_AMOADD_AQRL(TestSuite* s) : ITestCase(s, "amoadd_aqrl") {}
  void setup(uint32_t, uint32_t* shared, uint32_t*) override { *shared = 0; }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t*) override {
    uint32_t expected = n * iters;
    if (*shared != expected) {
      std::cout << "  shared=" << *shared << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 9) LR.W.AQ / SC.W.RL counter: same expected as Test_LRSC_COUNTER.
//    Validates the .aq / .rl encoding paths.
class Test_LRSC_COUNTER_AQRL : public ITestCase {
public:
  Test_LRSC_COUNTER_AQRL(TestSuite* s) : ITestCase(s, "lrsc_counter_aqrl") {}
  void setup(uint32_t /*n*/, uint32_t* shared, uint32_t*) override {
    *shared = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t*) override {
    uint32_t expected = n * iters;
    if (*shared != expected) {
      std::cout << "  shared=" << *shared << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 10) Atomic reduction (CUDA atomicAdd reduction). per_hart[h] = h+1
//     supplied by host; kernel atomicAdds each into shared. Expected
//     final = sum_{h=0..n-1} (h+1) = n*(n+1)/2.
class Test_ATOMIC_REDUCTION : public ITestCase {
public:
  Test_ATOMIC_REDUCTION(TestSuite* s) : ITestCase(s, "atomic_reduction") {}
  void setup(uint32_t n, uint32_t* shared, uint32_t* per_hart) override {
    *shared = 0;
    for (uint32_t i = 0; i < n; ++i) per_hart[i] = i + 1;
  }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t*) override {
    uint32_t expected = n * (n + 1) / 2;
    if (*shared != expected) {
      std::cout << "  shared=" << *shared << " expected=" << expected << std::endl;
      return 1;
    }
    return 0;
  }
};

// 11) Atomic critical section (CUDA atomicCAS spinlock). Only thread 0
//     of each warp participates. Active harts = num_cores * num_warps
//     = num_harts / num_threads. Final counter = active * iters; lock must end at 0.
class Test_ATOMIC_CRITICAL : public ITestCase {
public:
  Test_ATOMIC_CRITICAL(TestSuite* s, uint32_t num_threads_per_warp)
    : ITestCase(s, "atomic_critical")
    , num_threads_(num_threads_per_warp) {}
  // The non-atomic load/store inside the lock-protected critical section
  // needs L1<->L1 coherence, which Vortex does not implement: each core's L1
  // is private, and an AMO commits at the LLC and invalidates only the issuing
  // core's L1. The lock itself works across cores, but the counter access
  // inside it observes stale L1 copies on other cores.
  const char* skip_reason() const override {
    return (suite_->num_cores() > 1) ? "multi-core" : nullptr;
  }
  void setup(uint32_t /*n*/, uint32_t* shared, uint32_t* per_hart) override {
    *shared = 0;
    per_hart[0] = 0;  // lock starts unheld
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* per_hart) override {
    int errors = 0;
    uint32_t active   = n / num_threads_;          // one thread per warp
    uint32_t expected = active * iters;
    if (*shared != expected) {
      std::cout << "  counter=" << *shared << " expected=" << expected
                << " (active_harts=" << active << ")" << std::endl;
      ++errors;
    }
    if (per_hart[0] != 0) {
      std::cout << "  lock not released, lock=" << per_hart[0] << std::endl;
      ++errors;
    }
    return errors;
  }
private:
  uint32_t num_threads_;
};

// 12) Issuer self-consistency: each hart caches its own private word,
//     atomically increments it, and must observe the increment via a
//     later plain load. Final per_hart[hid] == 1 for every hart. Fails
//     when a non-LLC cache does not invalidate the issuer's stale copy
//     on the AMO (post-AMO plain load returns the pre-AMO value).
class Test_SELF_CONSISTENCY : public ITestCase {
public:
  Test_SELF_CONSISTENCY(TestSuite* s) : ITestCase(s, "self_consistency") {}
  void setup(uint32_t n, uint32_t* /*shared*/, uint32_t* per_hart) override {
    for (uint32_t h = 0; h < n; ++h) per_hart[h] = 0;
  }
  int verify(uint32_t n, uint32_t /*iters*/,
             const uint32_t* /*shared*/, const uint32_t* per_hart) override {
    int errors = 0;
    for (uint32_t h = 0; h < n; ++h) {
      if (per_hart[h] != 1) {
        if (errors < 4)
          std::cout << "  hart " << h << " observed " << per_hart[h]
                    << " expected 1 (stale post-AMO load)" << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

// 13) LMEM AMOADD hammer: the same contention as amoadd, on local memory, so
//     the LMEM banks resolve the read-modify-write rather than the dcache.
//     Local memory is per-workgroup, so each group's leader publishes its
//     group's final count and the published values sum to n * iters -- a closed
//     form independent of how the launch was shaped into groups.
class Test_AMOADD_LMEM : public ITestCase {
public:
  Test_AMOADD_LMEM(TestSuite* s) : ITestCase(s, "amoadd_lmem") {}
  void setup(uint32_t n, uint32_t* /*shared*/, uint32_t* per_hart) override {
    for (uint32_t h = 0; h < n; ++h) per_hart[h] = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* /*shared*/, const uint32_t* per_hart) override {
    uint64_t total = 0;
    uint32_t groups = 0;
    for (uint32_t h = 0; h < n; ++h) {
      if (per_hart[h] != 0) { total += per_hart[h]; ++groups; }
    }
    uint64_t expected = (uint64_t)n * iters;
    if (groups == 0) {
      std::cout << "  no group published a count -- the kernel did not run"
                << std::endl;
      return 1;
    }
    if (total != expected) {
      std::cout << "  published total=" << total << " expected=" << expected
                << " across " << groups << " group(s)" << std::endl;
      return 1;
    }
    return 0;
  }
};

// 14) LMEM LR/SC counter: the lock-free retry of test 7 on local memory, so
//     the LMEM banks resolve the reservation. Published per-group counts sum
//     to n * iters, independent of how the launch was shaped into groups.
class Test_LRSC_LMEM : public ITestCase {
public:
  Test_LRSC_LMEM(TestSuite* s) : ITestCase(s, "lrsc_lmem") {}
  void setup(uint32_t n, uint32_t* /*shared*/, uint32_t* per_hart) override {
    for (uint32_t h = 0; h < n; ++h) per_hart[h] = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* /*shared*/, const uint32_t* per_hart) override {
    uint64_t total = 0;
    uint32_t groups = 0;
    for (uint32_t h = 0; h < n; ++h) {
      if (per_hart[h] != 0) { total += per_hart[h]; ++groups; }
    }
    uint64_t expected = (uint64_t)n * iters;
    if (groups == 0) {
      std::cout << "  no group published a count -- no store-conditional succeeded"
                << std::endl;
      return 1;
    }
    if (total != expected) {
      std::cout << "  published total=" << total << " expected=" << expected
                << " across " << groups << " group(s)" << std::endl;
      return 1;
    }
    return 0;
  }
};

// 15) Native compare-and-swap ladder: every hart races for each slot, so each
//     slot has exactly one winner. Summing the per-hart win counts checks the
//     per-lane return value, which a compare-and-swap that returned the wrong
//     old word would get wrong even while leaving the ladder itself plausible.
class Test_CAS_LADDER : public ITestCase {
public:
  Test_CAS_LADDER(TestSuite* s) : ITestCase(s, "cas_ladder") {}
  void setup(uint32_t n, uint32_t* shared, uint32_t* per_hart) override {
    for (uint32_t i = 0; i < n; ++i) per_hart[i] = 0;
    // The ladder lives in the shared region; iters slots, all unclaimed.
    for (uint32_t i = 0; i < CAS_LADDER_SLOTS; ++i) shared[i] = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* shared, const uint32_t* per_hart) override {
    const uint32_t slots = (iters < CAS_LADDER_SLOTS) ? iters : CAS_LADDER_SLOTS;
    uint64_t wins = 0;
    for (uint32_t h = 0; h < n; ++h) wins += per_hart[h];
    if (wins != slots) {
      std::cout << "  wins=" << wins << " expected=" << slots
                << " (each slot must have exactly one winner)" << std::endl;
      return 1;
    }
    // Every slot must hold some hart's claim, and no slot may still be empty.
    for (uint32_t i = 0; i < slots; ++i) {
      if (shared[i] == 0 || shared[i] > n) {
        std::cout << "  slot " << i << " holds " << shared[i]
                  << ", expected a claim in 1.." << n << std::endl;
        return 1;
      }
    }
    return 0;
  }
};

// 16) LMEM compare-and-swap ladder: the cas_ladder race on local memory, so
//     the LMEM banks resolve the swap and the returned old word. Each group
//     has its own ladder with exactly one winner per slot; the leaders' publish
//     marker carries the group count, so the win counts must sum to
//     groups * slots whatever the launch geometry.
class Test_CAS_LMEM : public ITestCase {
public:
  Test_CAS_LMEM(TestSuite* s) : ITestCase(s, "cas_lmem") {}
  void setup(uint32_t n, uint32_t* /*shared*/, uint32_t* per_hart) override {
    for (uint32_t h = 0; h < n; ++h) per_hart[h] = 0;
  }
  int verify(uint32_t n, uint32_t iters,
             const uint32_t* /*shared*/, const uint32_t* per_hart) override {
    const uint32_t slots = (iters < CAS_LMEM_SLOTS) ? iters : CAS_LMEM_SLOTS;
    uint64_t wins = 0;
    uint32_t groups = 0;
    for (uint32_t h = 0; h < n; ++h) {
      uint32_t v = per_hart[h];
      if (v >= CAS_LMEM_GROUP_MARK) { ++groups; v -= CAS_LMEM_GROUP_MARK; }
      wins += v;
    }
    if (groups == 0) {
      std::cout << "  no group published -- the kernel did not run" << std::endl;
      return 1;
    }
    uint64_t expected = (uint64_t)groups * slots;
    if (wins != expected) {
      std::cout << "  wins=" << wins << " expected=" << expected
                << " across " << groups << " group(s)"
                << " (each slot must have exactly one winner)" << std::endl;
      return 1;
    }
    return 0;
  }
};

// 15/16) Store-conditional into another thread's reservation, on global and
//     on local memory (17/18 with Zacas). A single warp runs. A thread below
//     LRSC_FOREIGN_DIST must fail its SC, and its word keep its seed. Every
//     other thread's own SC must succeed, leaving the complement, unless the
//     warp outnumbers the stations; then it may fail and leave the seed.
class Test_LRSC_FOREIGN : public ITestCase {
public:
  Test_LRSC_FOREIGN(TestSuite* s, const char* name, uint32_t num_threads_per_warp,
                    uint32_t stations)
    : ITestCase(s, name)
    , num_threads_(num_threads_per_warp)
    , distinct_(num_threads_per_warp <= stations) {}
  uint32_t block_threads() const override { return num_threads_; }
  const char* skip_reason() const override {
    return (num_threads_ > 1) ? nullptr : "one-thread warp";
  }
  // No thread leaves its word at 0, so a thread that did not run fails the case.
  void setup(uint32_t /*n*/, uint32_t* shared, uint32_t* per_hart) override {
    for (uint32_t t = 0; t < num_threads_; ++t) {
      shared[t * LLC_LINE_WORDS] = 0;
      per_hart[t] = 0;
    }
  }
  int verify(uint32_t /*n*/, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t* per_hart) override {
    int errors = 0;
    for (uint32_t t = 0; t < num_threads_; ++t) {
      const bool     foreign  = (t < LRSC_FOREIGN_DIST);
      const bool     sc_ok    = (per_hart[t] == 0);
      const bool     legal    = foreign ? !sc_ok : (sc_ok || !distinct_);
      const uint32_t word     = shared[t * LLC_LINE_WORDS];
      const uint32_t seed     = LRSC_FOREIGN_SEED + t;
      // The word must match the reported outcome, or the required one.
      const bool     stored   = legal ? sc_ok : !foreign;
      const uint32_t expected = stored ? ~seed : seed;
      if (legal && word == expected) continue;
      if (errors < 8) {
        std::cout << "  thread " << t << ": "
                  << (foreign ? "foreign" : "own") << " sc "
                  << (sc_ok ? "succeeded" : "failed")
                  << " word=0x" << std::hex << word
                  << " expected=0x" << expected << std::dec << std::endl;
      }
      ++errors;
    }
    return errors;
  }
private:
  uint32_t num_threads_;
  bool     distinct_;
};

// 17) Store-conditional by a peer hart (19 with Zacas). The holder, thread 0
//     of warp 0 on core 0, reserves X. The peer reserves Y and
//     store-conditionals X, which must fail and leave X alone: the peer's plain
//     load of X right after it must read the seed. The holder's SC on X then
//     must succeed, so its reservation was live when the peer's SC arrived, and
//     the holder reserves X again. The peer swaps X, breaking that reservation,
//     and runs a control LR/SC on Y, which must succeed. The holder's last SC
//     on X then must fail, leaving the peer's swap in X.
class Test_LRSC_PEER : public ITestCase {
public:
  Test_LRSC_PEER(TestSuite* s) : ITestCase(s, "lrsc_peer") {}
  // The holder and the peer spin on each other, so they need separate warps.
  const char* skip_reason() const override {
    return (suite_->num_cores() > 1 || suite_->num_warps() > 1)
         ? nullptr : "one core of one warp";
  }
  // The kernel seeds X and Y itself; main zeroes the flags and the out lines.
  void setup(uint32_t, uint32_t*, uint32_t*) override {}
  int verify(uint32_t /*n*/, uint32_t /*iters*/,
             const uint32_t* shared, const uint32_t* /*per_hart*/) override {
    const uint32_t  x = shared[LRSC_PEER_X * LLC_LINE_WORDS];
    const uint32_t  y = shared[LRSC_PEER_Y * LLC_LINE_WORDS];
    const uint32_t* a = &shared[LRSC_PEER_A_OUT * LLC_LINE_WORDS];
    const uint32_t* b = &shared[LRSC_PEER_B_OUT * LLC_LINE_WORDS];
    int errors = 0;
    auto fail = [&]() -> std::ostream& {
      if (errors++ == 0)
        std::cout << "  holder core 0 warp 0, peer core " << suite_->peer_core()
                  << " warp " << suite_->peer_warp() << std::endl;
      return std::cout << "  ";
    };
    auto check = [&](bool ok, const char* what, uint32_t got, uint32_t expected) {
      if (!ok)
        fail() << what << "=0x" << std::hex << got << " expected 0x" << expected
               << std::dec << std::endl;
    };
    for (auto [out, role] : {std::pair{a, "holder"}, std::pair{b, "peer"}}) {
      if (out[LRSC_PEER_RAN] != 1)
        fail() << role << " did not run" << std::endl;
      else if (out[LRSC_PEER_TIMEOUT] != 0)
        fail() << role << " gave up after " << out[LRSC_PEER_POLLS]
               << " polls for the other role" << std::endl;
    }
    // The value checks assume both roles ran in step; otherwise they mislead.
    if (errors != 0)
      return errors;
    check(a[LRSC_PEER_A_LR] == LRSC_PEER_X_SEED, "holder's LR of X",
          a[LRSC_PEER_A_LR], LRSC_PEER_X_SEED);
    if (b[LRSC_PEER_B_SC] == 0)
      fail() << "peer's SC on X succeeded on the holder's reservation" << std::endl;
    check(b[LRSC_PEER_B_SEEN] == LRSC_PEER_X_SEED, "X read by the peer after its SC",
          b[LRSC_PEER_B_SEEN], LRSC_PEER_X_SEED);
    if (a[LRSC_PEER_A_LIVE] != 0)
      fail() << "holder's SC on X failed after the peer's SC: its reservation was not live"
             << std::endl;
    check(a[LRSC_PEER_A_LR2] == LRSC_PEER_A_LIVE_VALUE, "holder's second LR of X",
          a[LRSC_PEER_A_LR2], LRSC_PEER_A_LIVE_VALUE);
    check(b[LRSC_PEER_B_PREV] == LRSC_PEER_A_LIVE_VALUE, "X found by the peer's swap",
          b[LRSC_PEER_B_PREV], LRSC_PEER_A_LIVE_VALUE);
    if (b[LRSC_PEER_B_CTRL] != 0)
      fail() << "peer's control SC on Y failed" << std::endl;
    check(y == LRSC_PEER_Y_SEED + 1, "Y", y, LRSC_PEER_Y_SEED + 1);
    if (a[LRSC_PEER_A_SC] == 0)
      fail() << "holder's SC on X succeeded after the peer's swap" << std::endl;
    check(x == LRSC_PEER_B_SWAP_VALUE, "X", x, LRSC_PEER_B_SWAP_VALUE);
    return errors;
  }
};

inline TestSuite::TestSuite(vx_device_h device) : device_(device) {
  uint64_t num_clusters, num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CLUSTERS, &num_clusters));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  num_cores_ = (uint32_t)num_cores;
  num_warps_ = (uint32_t)num_warps;

  // lrsc_peer's peer is the first core of the last cluster: it holds core 0's
  // position within its cluster, so only the cluster part of the global core
  // id tells the two harts apart. One cluster takes core 1, one core warp 1.
  if (num_clusters > 1) {
    peer_core_ = (uint32_t)((num_clusters - 1) * (num_cores / num_clusters));
  } else if (num_cores > 1) {
    peer_core_ = 1;
  } else {
    peer_warp_ = 1;
  }

  add_test(new Test_AMOADD(this));
  add_test(new Test_AMOOR(this));
  add_test(new Test_AMOAND(this));
  add_test(new Test_AMOXOR(this));
  add_test(new Test_AMOMAX(this));
  add_test(new Test_AMOMINU(this));
  add_test(new Test_AMOSWAP(this));
  add_test(new Test_LRSC_COUNTER(this));
  add_test(new Test_AMOADD_AQRL(this));
  add_test(new Test_LRSC_COUNTER_AQRL(this));
  add_test(new Test_ATOMIC_REDUCTION(this));
  add_test(new Test_ATOMIC_CRITICAL(this, (uint32_t)num_threads));
  add_test(new Test_SELF_CONSISTENCY(this));
  add_test(new Test_AMOADD_LMEM(this));
  add_test(new Test_LRSC_LMEM(this));
#if VX_CFG_EXT_ZACAS_ENABLED
  // Native compare-and-swap only exists when Zacas is built in. Without it a
  // compare-exchange lowers to an LR/SC retry loop, which is the construct
  // this instruction exists to replace and which does not survive contention.
  add_test(new Test_CAS_LADDER(this));
  add_test(new Test_CAS_LMEM(this));
#endif
  add_test(new Test_LRSC_FOREIGN(this, "lrsc_foreign", (uint32_t)num_threads,
                                 LRSC_LLC_STATIONS));
  add_test(new Test_LRSC_FOREIGN(this, "lrsc_foreign_lmem", (uint32_t)num_threads,
                                 LRSC_LMEM_STATIONS));
  add_test(new Test_LRSC_PEER(this));
}

inline TestSuite::~TestSuite() {
  for (auto* t : _tests) delete t;
}
