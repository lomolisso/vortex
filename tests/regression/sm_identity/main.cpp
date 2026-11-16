// sm_identity: one whole-SM block per SM reads %smid and %nsmid, the blocks
// meeting at an atomic counter so they are all resident at once. The host
// checks that %smid is a bijection onto the SMs and %nsmid their number, that
// the caps report the build's level counts and CTA caps, and that vx_hier.h
// decodes every %smid into a distinct (partition, GPC, CPC, TPC slot, SM)
// that encodes back to it, as the numbering rule gives it: GPCs
// partition-major, and inside a GPC TPC-slot-major, CPC-minor.
//
//   -v       print every SM's position

#include <vortex2.h>
#include <VX_types.h>
#include <vx_hier.h>
#include "common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <tuple>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

namespace {

constexpr uint32_t kRendezvousBound = 1000000;

const char* kernel_file = "kernel.vxbin";
bool        verbose     = false;

struct Caps {
  uint64_t num_cores = 0, num_clusters = 0, num_warps = 0, num_threads = 0, isa = 0;
  uint64_t partitions = 0, gpcs_per_partition = 0, cpcs_per_gpc = 0, tpcs_per_cpc = 0, sms_per_tpc = 0;
  uint64_t max_ctas = 0, max_ctas_cluster = 0;
};

int errors = 0;

void expect_eq(const char* what, uint64_t got, uint64_t want) {
  if (got != want) {
    if (errors < 32)
      std::printf("*** %s = %llu, expected %llu\n", what, (unsigned long long)got, (unsigned long long)want);
    ++errors;
  }
}

}  // namespace

int main(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "vh")) != -1) {
    if (c == 'v') {
      verbose = true;
    } else {
      std::printf("Usage: [-v] [-h]\n");
      return (c == 'h') ? 0 : -1;
    }
  }

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  Caps caps;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,          &caps.num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CLUSTERS,       &caps.num_clusters));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,          &caps.num_warps));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,        &caps.num_threads));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,          &caps.isa));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_PARTITIONS,     &caps.partitions));
  CHECK(vx_device_query(dev, VX_CAPS_GPCS_PER_PARTITION, &caps.gpcs_per_partition));
  CHECK(vx_device_query(dev, VX_CAPS_CPCS_PER_GPC,       &caps.cpcs_per_gpc));
  CHECK(vx_device_query(dev, VX_CAPS_TPCS_PER_CPC,       &caps.tpcs_per_cpc));
  CHECK(vx_device_query(dev, VX_CAPS_SMS_PER_TPC,        &caps.sms_per_tpc));
  CHECK(vx_device_query(dev, VX_CAPS_MAX_CTAS_PER_SM,    &caps.max_ctas));
  CHECK(vx_device_query(dev, VX_CAPS_MAX_CTAS_PER_SM_CLUSTER, &caps.max_ctas_cluster));
  const uint32_t n = (uint32_t)caps.num_cores;
  std::printf("sm_identity: %u SMs: %u partitions x %u GPCs x %u CPCs x %u TPCs x %u SMs;"
              " %u CTAs per SM, %u of a cluster launch\n", n, (uint32_t)caps.partitions,
              (uint32_t)caps.gpcs_per_partition, (uint32_t)caps.cpcs_per_gpc, (uint32_t)caps.tpcs_per_cpc,
              (uint32_t)caps.sms_per_tpc, (uint32_t)caps.max_ctas, (uint32_t)caps.max_ctas_cluster);

  expect_eq("caps NUM_CORES (SMs)", n, VX_CFG_NUM_SMS);
  expect_eq("caps NUM_CLUSTERS (GPCs)", caps.num_clusters, VX_CFG_NUM_CLUSTERS);
  expect_eq("caps NUM_PARTITIONS", caps.partitions, VX_CFG_NUM_PARTITIONS);
  expect_eq("caps CPCS_PER_GPC", caps.cpcs_per_gpc, VX_CFG_CPCS_PER_GPC);
  expect_eq("caps SMS_PER_TPC", caps.sms_per_tpc, VX_CFG_SOCKET_SIZE);
  expect_eq("caps partitions x GPCs per partition", caps.partitions * caps.gpcs_per_partition,
            caps.num_clusters);
  expect_eq("caps CPCs x TPCs x SMs per TPC", caps.cpcs_per_gpc * caps.tpcs_per_cpc * caps.sms_per_tpc,
            caps.num_cores / caps.num_clusters);
  expect_eq("caps MAX_CTAS_PER_SM", caps.max_ctas, VX_CFG_MAX_CTAS_PER_SM);
  expect_eq("caps MAX_CTAS_PER_SM_CLUSTER", caps.max_ctas_cluster, VX_CFG_MAX_CTAS_PER_SM_CLUSTER);
  if (errors) {
    std::printf("Found %d errors!\nFAILED!\n", errors);
    return 1;
  }

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_get_kernel(mod, "sm_id", &kernel));

  const uint64_t rec_bytes = (uint64_t)n * REC_WORDS * sizeof(uint32_t);
  std::vector<uint32_t> recs((size_t)n * REC_WORDS, 0);
  uint32_t arrive = 0;
  vx_buffer_h rec_buf = nullptr, arrive_buf = nullptr;
  id_arg_t arg = {};
  CHECK(vx_buffer_create(dev, rec_bytes, VX_MEM_READ_WRITE, &rec_buf));
  CHECK(vx_buffer_create(dev, sizeof(arrive), VX_MEM_READ_WRITE, &arrive_buf));
  CHECK(vx_enqueue_write(q, rec_buf, 0, recs.data(), rec_bytes, 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, arrive_buf, 0, &arrive, sizeof(arrive), 0, nullptr, nullptr));
  CHECK(vx_buffer_address(rec_buf, &arg.rec_addr));
  CHECK(vx_buffer_address(arrive_buf, &arg.arrive_addr));
  arg.blocks = n;
  arg.bound  = kRendezvousBound;

  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = n;
  li.grid_dim[1]  = 1;
  li.grid_dim[2]  = 1;
  li.block_dim[0] = (uint32_t)(caps.num_warps * caps.num_threads);
  li.block_dim[1] = 1;
  li.block_dim[2] = 1;
  vx_event_h ev = nullptr, rd = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &ev));
  CHECK(vx_enqueue_read(q, recs.data(), rec_buf, 0, rec_bytes, 1, &ev, &rd));
  CHECK(vx_event_wait_value(rd, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(rd);
  vx_event_release(ev);
  CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));

  const vortex::hier::Shape shape = {(uint32_t)caps.partitions, (uint32_t)caps.gpcs_per_partition,
                                     (uint32_t)caps.cpcs_per_gpc, (uint32_t)caps.tpcs_per_cpc,
                                     (uint32_t)caps.sms_per_tpc};
  expect_eq("vx_hier num_sms", shape.num_sms(), n);
  std::set<uint32_t> smids;
  std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> positions;
  const uint32_t per_gpc = n / (uint32_t)caps.num_clusters;
  for (uint32_t b = 0; b < n; ++b) {
    const uint32_t* rec = &recs[(size_t)b * REC_WORDS];
    char what[64];
    std::snprintf(what, sizeof(what), "block %u ran", b);
    expect_eq(what, rec[REC_RUNS], 1);
    if (rec[REC_RUNS] != 1)
      continue;
    std::snprintf(what, sizeof(what), "block %u: %%nsmid", b);
    expect_eq(what, rec[REC_NSMID], n);
    if (rec[REC_POLLS] == 0) {
      std::printf("*** block %u: the rendezvous timed out (not every SM held a block at once)\n", b);
      ++errors;
    }
    const uint32_t sm = rec[REC_SMID];
    if (sm >= n || !smids.insert(sm).second) {
      std::printf("*** block %u: %%smid %u is out of range or already seen\n", b, sm);
      ++errors;
      continue;
    }
    const vortex::hier::Pos p = vortex::hier::decode(shape, sm);
    std::snprintf(what, sizeof(what), "SM %u: encode(decode(%%smid))", sm);
    expect_eq(what, vortex::hier::encode(shape, p), sm);
    // The numbering rule, written out on its own.
    const uint32_t gpc = sm / per_gpc, i = sm % per_gpc;
    const uint32_t s2 = shape.sms_per_tpc, cc = shape.cpcs_per_gpc;
    std::snprintf(what, sizeof(what), "SM %u: partition", sm);
    expect_eq(what, p.partition, gpc / shape.gpcs_per_partition);
    std::snprintf(what, sizeof(what), "SM %u: GPC in its partition", sm);
    expect_eq(what, p.gpc, gpc % shape.gpcs_per_partition);
    std::snprintf(what, sizeof(what), "SM %u: CPC", sm);
    expect_eq(what, p.cpc, (i / s2) % cc);
    std::snprintf(what, sizeof(what), "SM %u: TPC slot", sm);
    expect_eq(what, p.tpc_slot, i / (s2 * cc));
    std::snprintf(what, sizeof(what), "SM %u: SM in its TPC", sm);
    expect_eq(what, p.parity, i % s2);
    if (p.partition >= shape.partitions || p.gpc >= shape.gpcs_per_partition || p.cpc >= shape.cpcs_per_gpc
        || p.tpc_slot >= shape.tpcs_per_cpc || p.parity >= shape.sms_per_tpc
        || !positions.insert({p.partition, p.gpc, p.cpc, p.tpc_slot, p.parity}).second) {
      std::printf("*** SM %u: position (%u, %u, %u, %u, %u) out of range or taken twice\n", sm,
                  p.partition, p.gpc, p.cpc, p.tpc_slot, p.parity);
      ++errors;
    }
    if (verbose)
      std::printf("SM %u: partition %u, GPC %u, CPC %u, TPC slot %u, SM %u\n", sm, p.partition, p.gpc,
                  p.cpc, p.tpc_slot, p.parity);
  }
  expect_eq("distinct %smid values", smids.size(), n);
  expect_eq("distinct positions", positions.size(), n);

  vx_buffer_release(arrive_buf);
  vx_buffer_release(rec_buf);
  vx_kernel_release(kernel);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) {
    std::printf("Found %d errors!\nFAILED!\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
