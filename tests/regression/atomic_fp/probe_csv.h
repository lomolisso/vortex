#pragma once

// Reader of the H100 atomics semantics CSV, one measured operation per row
// (columns form, space, op, variant, old_hex, rhs_hex, mem_hex, and optionally
// rhs2_hex, ret_hex, guard_ok; others are ignored), and its expectations under
// amo_ref. Values are hex numbers up to 128 bits, element 0 in the low bits;
// cas.b16 rows carry the comparand in rhs and the new value in rhs2. The golden
// subset is tests/unittest/amo_ref/h100_atomics.csv.gz.

#include <amo_ref.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace probe_csv {

enum Kind : uint32_t {
  K_NONE = 0,
  K_OP,     // amo_ref::apply(op, width) on each `width` word of the value
  K_ADD16,  // amo_ref::add_f16 / add_bf16 (the header loops)
  K_CAS16,  // amo_ref::cas(W_H): mem = old == rhs ? rhs2 : old
};

struct Form {
  Kind kind = K_NONE;
  uint32_t op = 0, width = 0;
  bool bf = false;      // K_ADD16: bf16
  uint32_t bytes = 0;   // the whole operand
};

struct V128 { uint64_t lo = 0, hi = 0; };

struct Row {
  std::string form, op, space, variant;
  Form f;
  V128 old, rhs, rhs2, ret, mem;
  bool has_ret = false, guard_ok = true;
};

inline std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  std::istringstream is(s);
  while (std::getline(is, cur, sep))
    out.push_back(cur);
  if (!s.empty() && s.back() == sep)
    out.push_back("");
  return out;
}

inline std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n\"");
  size_t b = s.find_last_not_of(" \t\r\n\"");
  return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

inline V128 hex128(std::string h) {
  V128 v;
  if (h.size() > 1 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X'))
    h = h.substr(2);
  if (h.size() > 16) {
    v.hi = std::strtoull(h.substr(0, h.size() - 16).c_str(), nullptr, 16);
    h = h.substr(h.size() - 16);
  }
  v.lo = h.empty() ? 0 : std::strtoull(h.c_str(), nullptr, 16);
  return v;
}

// (form, op) of a probe row: f32 f64 f16 bf16 f16x2 bf16x2 u32 b16, and
// vN.<elem> vectors.
inline Form map_form(const std::string& form, const std::string& op) {
  using namespace vortex::amo_ref;
  Form f;
  std::string elem = form;
  uint32_t vec = 1;
  if (form.size() > 3 && form[0] == 'v' && form[2] == '.') {
    vec = form[1] - '0';
    elem = form.substr(3);
  }
  const uint32_t eb = (elem == "f64") ? 8 : (elem == "f16" || elem == "bf16" || elem == "b16") ? 2 : 4;
  f.bytes = eb * vec;
  const bool bf = elem.find("bf16") != std::string::npos;
  auto set = [&](Kind k, uint32_t o, uint32_t w) { f.kind = k; f.op = o; f.width = w; };
  if (op == "add") {
    if (elem == "f32")                        set(K_OP, vec == 1 ? OP_FADD : OP_FADD2, vec == 1 ? W_W : W_D);
    else if (elem == "f64")                   set(K_OP, OP_FADD, W_D);
    else if ((elem == "f16" || elem == "bf16") && vec == 1) { set(K_ADD16, 0, W_H); f.bf = bf; }
    else                                      set(K_OP, bf ? OP_FADD2_BF : OP_FADD2, W_W);
  } else if ((op == "min" || op == "max") && vec > 1) {
    const bool mx = (op == "max");
    set(K_OP, bf ? (mx ? OP_FMAX2_BF : OP_FMIN2_BF) : (mx ? OP_FMAX2 : OP_FMIN2), W_W);
  } else if (op == "inc" || op == "dec") {
    set(K_OP, op == "inc" ? OP_INC : OP_DEC, W_W);
  } else if (op == "cas" && elem == "b16") {
    set(K_CAS16, OP_CAS, W_H);
  }
  // A vector of odd 32-bit f32 count would not split into f32x2 pairs.
  if (f.kind == K_OP && f.op == OP_FADD2 && f.width == W_D && f.bytes % 8)
    f.kind = K_NONE;
  return f;
}

// The amo_ref site of a row's space. Only a global address completes the
// H100's float hardware op; shared and distributed shared memory, in every
// spelling, run the CAS-loop fallback.
inline vortex::amo_ref::Site row_site(const Row& r) {
  return (r.space == "global" || r.space == "generic_global") ? vortex::amo_ref::SITE_LLC
                                                               : vortex::amo_ref::SITE_LOOP;
}

// Rows that measure no atomic: nvcc 12.9 lowers a packed f16/bf16 add to
// distributed shared memory into a plain load, integer add and store.
inline bool excluded(const Row& r) {
  return r.op == "add" && (r.form == "f16x2" || r.form == "bf16x2")
      && (r.space == "shared_cluster" || r.space == "generic_dsmem");
}

// The memory after the op under amo_ref, over the row's operand bytes.
inline V128 expect_mem(const Row& r, vortex::amo_ref::Site site) {
  using namespace vortex::amo_ref;
  V128 m = r.old;
  switch (r.f.kind) {
  case K_ADD16:
    m.lo = r.f.bf ? add_bf16((uint16_t)r.old.lo, (uint16_t)r.rhs.lo) : add_f16((uint16_t)r.old.lo, (uint16_t)r.rhs.lo);
    m.hi = 0;
    return m;
  case K_CAS16:
    m.lo = cas(W_H, r.old.lo, r.rhs.lo, r.rhs2.lo);
    m.hi = 0;
    return m;
  case K_OP: {
    const uint32_t wb = 1u << r.f.width;
    uint8_t o[16] = {}, b[16] = {}, z[16] = {};
    auto put = [](uint8_t* d, const V128& v) {
      for (int i = 0; i < 8; ++i) { d[i] = (uint8_t)(v.lo >> (8 * i)); d[8 + i] = (uint8_t)(v.hi >> (8 * i)); }
    };
    put(o, r.old);
    put(b, r.rhs);
    for (uint32_t i = 0; i + wb <= r.f.bytes && i < 16; i += wb) {
      uint64_t x = 0, y = 0;
      for (uint32_t k = 0; k < wb; ++k) { x |= (uint64_t)o[i + k] << (8 * k); y |= (uint64_t)b[i + k] << (8 * k); }
      const uint64_t n = apply(r.f.op, r.f.width, x, y, site);
      for (uint32_t k = 0; k < wb; ++k) z[i + k] = (uint8_t)(n >> (8 * k));
    }
    V128 v;
    for (int i = 0; i < 8; ++i) { v.lo |= (uint64_t)z[i] << (8 * i); v.hi |= (uint64_t)z[8 + i] << (8 * i); }
    return v;
  }
  default:
    return m;
  }
}

inline bool load(const std::string& path, std::vector<Row>& rows, std::string& msg) {
  FILE* fp = nullptr;
  const bool gz = path.size() > 3 && path.compare(path.size() - 3, 3, ".gz") == 0;
  if (gz) {
    FILE* probe = std::fopen(path.c_str(), "r");
    if (!probe) { msg = "absent: " + path; return false; }
    std::fclose(probe);
    fp = popen(("gzip -dc '" + path + "'").c_str(), "r");
  } else {
    fp = std::fopen(path.c_str(), "r");
  }
  if (!fp) { msg = "absent: " + path; return false; }
  std::map<std::string, int> col;
  char buf[8192];
  bool header = true;
  while (std::fgets(buf, sizeof(buf), fp)) {
    std::string line = buf;
    if (trim(line).empty())
      continue;
    auto f = split(trim(line), ',');
    for (auto& x : f) x = trim(x);
    if (header) {
      for (size_t i = 0; i < f.size(); ++i) col[f[i]] = (int)i;
      header = false;
      for (auto need : {"form", "space", "op", "old_hex", "rhs_hex", "mem_hex"}) {
        if (!col.count(need)) {
          msg = std::string("header lacks ") + need + ": " + trim(line);
          gz ? pclose(fp) : std::fclose(fp);
          return false;
        }
      }
      continue;
    }
    auto get = [&](const char* n) {
      auto it = col.find(n);
      return (it != col.end() && it->second < (int)f.size()) ? f[it->second] : std::string();
    };
    Row r;
    r.form    = get("form");
    r.op      = get("op");
    r.space   = get("space");
    r.variant = get("variant");
    r.f       = map_form(r.form, r.op);
    r.old     = hex128(get("old_hex"));
    r.rhs     = hex128(get("rhs_hex"));
    r.rhs2    = hex128(get("rhs2_hex"));
    r.mem     = hex128(get("mem_hex"));
    r.has_ret = !get("ret_hex").empty();
    r.ret     = hex128(get("ret_hex"));
    r.guard_ok = get("guard_ok") != "0";
    rows.push_back(r);
  }
  gz ? pclose(fp) : std::fclose(fp);
  if (header) { msg = "empty: " + path; return false; }
  return true;
}

// `path` itself, or its .gz; "" if neither exists.
inline std::string locate(const std::string& path) {
  for (auto p : {path, path + ".gz"}) {
    FILE* fp = std::fopen(p.c_str(), "r");
    if (fp) { std::fclose(fp); return p; }
  }
  return "";
}

}  // namespace probe_csv
