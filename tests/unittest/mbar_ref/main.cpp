// mbar_ref against the H100: the mbarrier scripts of the H100 semantics probe
// (tests/regression/mbar_ops/scripts.h) replayed through
// mbar_ref::apply, every recorded pending count and test_wait result equal to
// the H100's; then the operand flags' edge cases and errors.

#include <mbar_ref.h>
#include <scripts.h>

#include <cinttypes>
#include <cstdio>

namespace mr = vortex::mbar_ref;

namespace {

int g_fail = 0, g_pass = 0;

void expect(const char* what, uint64_t got, uint64_t want) {
  if (got != want) {
    std::printf("FAIL %s: got %#" PRIx64 ", expected %#" PRIx64 "\n", what, got, want);
    ++g_fail;
  } else {
    ++g_pass;
  }
}

void h100_scripts() {
  uint32_t checked = 0;
  for (const auto& sc : mbar_ops::scripts()) {
    std::vector<uint64_t> rec;
    uint64_t word = 0;
    char what[96];
    std::snprintf(what, sizeof what, "%s: mbar_ref errors", sc.name);
    expect(what, mbar_ops::replay(sc, rec, word), 1);
    for (size_t i = 0; i < sc.steps.size(); ++i) {
      const auto& st = sc.steps[i];
      if (st.h100 == mbar_ops::NA)
        continue;
      std::snprintf(what, sizeof what, "%s step %zu (kind %u n %u)", sc.name, i, st.kind, st.n);
      expect(what, (uint64_t)mbar_ops::h100_view(st, rec[i]), (uint64_t)st.h100);
      checked += sc.h100;
    }
    // Every completion an op makes is the one the next wait sees.
    uint64_t w = 0;
    for (const auto& st : sc.steps) {
      if (st.kind == K_INIT) {
        w = mr::init(st.n);
        continue;
      }
      const mr::Result r = mr::apply(mbar_ops::op_code(st), w, mbar_ops::operand(st));
      if (r.writes) {
        std::snprintf(what, sizeof what, "%s: flip matches the phase bit", sc.name);
        expect(what, r.flip, mr::phase(r.new_word) != mr::phase(w));
        w = r.new_word;
      }
    }
  }
  std::printf("mbar_ref: %u H100 values checked\n", checked);
}

void flags() {
  // A drop that completes the phase reloads pending from the new expected count.
  mr::Result r = mr::apply(mr::OP_ARRIVE, mr::pack(1, 4, 0, 0), 1 | mr::FLAG_DROP);
  expect("drop completes: err", r.err, 0);
  expect("drop completes: flip", r.flip, 1);
  expect("drop completes: word", r.new_word, mr::pack(3, 3, 0, 1));
  // The flags never reach the word.
  expect("drop: bits 62:61 zero", r.new_word & mr::FLAGS, 0);

  // arrive_drop.expect_tx: tx += bytes, one arrival dropped.
  r = mr::apply(mr::OP_ARRIVE_TX, mr::init(3), 100 | mr::FLAG_DROP);
  expect("drop_tx: err", r.err, 0);
  expect("drop_tx: word", r.new_word, mr::pack(2, 2, 100, 0));
  r = mr::apply(mr::OP_ARRIVE_TX, mr::pack(1, 3, -100, 0), 100 | mr::FLAG_DROP);
  expect("drop_tx completes: word", r.new_word, mr::pack(2, 2, 0, 1));
  expect("drop_tx completes: flip", r.flip, 1);

  // noComplete leaves the phase open; the relaxed form is the same op.
  r = mr::apply(mr::OP_ARRIVE, mr::init(4), 3 | mr::FLAG_NOCOMPLETE);
  expect("nocomplete open: err", r.err, 0);
  expect("nocomplete open: word", r.new_word, mr::pack(1, 4, 0, 0));
  // Pending reaches 0 while tx is outstanding: not a completion.
  r = mr::apply(mr::OP_ARRIVE, mr::pack(1, 4, 8, 0), 1 | mr::FLAG_NOCOMPLETE);
  expect("nocomplete tx pending: err", r.err, 0);
  expect("nocomplete tx pending: word", r.new_word, mr::pack(0, 4, 8, 0));
  r = mr::apply(mr::OP_ARRIVE, mr::init(4), 1 | mr::FLAG_DROP | mr::FLAG_NOCOMPLETE);
  expect("drop nocomplete: word", r.new_word, mr::pack(3, 3, 0, 0));

  // The new errors; the model still applies the normal rule.
  r = mr::apply(mr::OP_ARRIVE, mr::init(2), 2 | mr::FLAG_NOCOMPLETE);
  expect("ERR_NOCOMPLETE", r.err, mr::ERR_NOCOMPLETE);
  expect("ERR_NOCOMPLETE: word", r.new_word, mr::pack(2, 2, 0, 1));
  r = mr::apply(mr::OP_ARRIVE_TX, mr::init(1), 0 | mr::FLAG_NOCOMPLETE);
  expect("ERR_NOCOMPLETE arrive_tx", r.err, mr::ERR_NOCOMPLETE);
  r = mr::apply(mr::OP_ARRIVE, mr::init(2), 1 | mr::FLAG_DROP | mr::FLAG_NOCOMPLETE);
  expect("drop nocomplete, last but one", r.err, 0);
  r = mr::apply(mr::OP_ARRIVE, mr::pack(1, 2, 0, 0), 1 | mr::FLAG_DROP | mr::FLAG_NOCOMPLETE);
  expect("ERR_NOCOMPLETE by a drop", r.err, mr::ERR_NOCOMPLETE);
  expect("ERR_NOCOMPLETE by a drop: word", r.new_word, mr::pack(1, 1, 0, 1));

  r = mr::apply(mr::OP_ARRIVE, mr::init(2), 2 | mr::FLAG_DROP);
  expect("ERR_DROPZERO", r.err, mr::ERR_DROPZERO);
  expect("ERR_DROPZERO: word", r.new_word, mr::pack(0, 0, 0, 1));
  r = mr::apply(mr::OP_ARRIVE, mr::pack(3, 2, 0, 0), 2 | mr::FLAG_DROP);
  expect("ERR_DROPZERO, pending left", r.err, mr::ERR_DROPZERO);
  r = mr::apply(mr::OP_ARRIVE_TX, mr::init(1), 4 | mr::FLAG_DROP);
  expect("ERR_DROPZERO arrive_tx", r.err, mr::ERR_DROPZERO);
  r = mr::apply(mr::OP_ARRIVE, mr::init(2), 3 | mr::FLAG_DROP);
  expect("drop past pending", r.err, mr::ERR_UNDERFLOW | mr::ERR_DROPZERO);
  r = mr::apply(mr::OP_ARRIVE, mr::init(1), 1 | mr::FLAG_DROP | mr::FLAG_NOCOMPLETE);
  expect("both errors", r.err, mr::ERR_DROPZERO | mr::ERR_NOCOMPLETE);

  // Bits 62:61 are flags only on the arrives: elsewhere they are bytes out of range.
  r = mr::apply(mr::OP_EXPECT_TX, mr::init(1), 8 | mr::FLAG_DROP);
  expect("expect_tx bit 62", r.err, mr::ERR_TX_RANGE);
  r = mr::apply(mr::OP_COMPLETE_TX, mr::init(1), 8 | mr::FLAG_NOCOMPLETE);
  expect("complete_tx bit 61", r.err, mr::ERR_TX_RANGE);
  // Other high bits of an arrive keep today's checks.
  r = mr::apply(mr::OP_ARRIVE, mr::init(4), (uint64_t(1) << 60) | 1 | mr::FLAG_DROP);
  expect("arrive bit 60", r.err & mr::ERR_UNDERFLOW, mr::ERR_UNDERFLOW);
  r = mr::apply(mr::OP_ARRIVE_TX, mr::init(4), (uint64_t(1) << 21) | mr::FLAG_NOCOMPLETE);
  expect("arrive_tx bit 21", r.err, mr::ERR_TX_RANGE);
  // Waits ignore the flags' bits of the token (phase in bit 63).
  r = mr::apply(mr::OP_TEST_WAIT, mr::pack(1, 1, 0, 1), mr::FLAGS);
  expect("test_wait token with bits 62:61", r.ret_word, 1);
}

}  // namespace

int main() {
  h100_scripts();
  flags();
  std::printf("mbar_ref: %d passed, %d failed\n", g_pass, g_fail);
  if (g_fail) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
