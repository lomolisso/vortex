#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define UR_MAX_LANES 32
#define UR_SENTINEL  0x5a5a5a5a5a5a5a5aull
#define UR_CHAIN     16

enum { MODE_ALU = 0, MODE_OPERAND = 1, MODE_LATENCY = 2 };

// ALU mode case codes: the uniform_ref ops (U-ALU 0-7, U-CMP 8-10, R2UR+UR2R 11), then:
enum { CASE_URZ = 12, CASE_S2UR = 13 };
#define UR_NUM_CODES 14
// Words per lane and case: S2UR mode writes {s2ur, csr} per selector at [2 sel, 2 sel + 1].
#define S2UR_WORDS 56

// One case over one warp: lane i is active when tmask bit i is set and holds a[i], b[i].
typedef struct {
  uint32_t code;
  uint32_t pad;
  uint64_t tmask;
  uint64_t a[UR_MAX_LANES];
  uint64_t b[UR_MAX_LANES];
} ur_case_t;

// Operand mode: per lane, for each checked instruction, {GPR form, UR form}.
#define OPD_MAX 40

typedef struct {
  uint64_t cases_addr;  // ALU: ur_case_t[ncases]
  uint64_t out_addr;    // ALU: uint64_t[ncases][lanes]; operand: [lanes][OPD_MAX][2] + counters; latency: per warp records
  uint64_t u;           // operand mode: the uniform value
  uint32_t mode;
  uint32_t ncases;
  uint32_t iters;
  uint32_t sel;         // S2UR selectors checked (ALU mode: all of them)
} ur_arg_t;

// Operand mode counters after the lane results: instret of the prefixed and the plain block.
#define OPD_COUNTERS 2
// Latency mode records per warp: {ualu chain, s2ur->consumer chain, s2ur back-to-back, result,
// the same two chains on the vector ALU and the CSR unit (add chain, csrr->add chain), then the
// ualu, s2ur, add and csrr chains again with a body twice as long} cycles.
#define LAT_RECS 10

#endif
