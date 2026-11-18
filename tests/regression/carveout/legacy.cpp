// The chase as a legacy image for the legacy vx_start path: main() on each core's first thread, the
// argument in MSCRATCH, no KMU entry point.

#include <vx_intrinsics.h>

#include "common.h"
#include "chase.h"

int main() {
  auto arg = reinterpret_cast<const kernel_arg_t*>(csr_read(VX_CSR_MSCRATCH));
  if (vx_core_id() == 0)
    cv_chase(arg, nullptr);
  return 0;
}
