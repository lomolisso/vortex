// The 29 DSMEM collectives, one kernel each, built for the CLUSTER_N,
// MSG_ELEMS and ELEM_T of the build. A collective whose constraints the
// build violates exports an empty kernel; the host never launches it.

#include <vx_cuda.h>
#include "common.h"
#include "dsmem_primitives.h"

#include "coll/allgather__bruck__mpich.h"
#include "coll/allgather__direct_pull__generic.h"
#include "coll/allgather__direct_replicated_push__credit.h"
#include "coll/allgather__direct_shift__clusterfusion.h"
#include "coll/allgather__gather_bcast_binomial__mpich.h"
#include "coll/allgather__recursive_doubling__clusterfusion.h"
#include "coll/allgather__ring__nccl.h"
#include "coll/allreduce__2d_mesh__generic.h"
#include "coll/allreduce__atomic_all_push__generic.h"
#include "coll/allreduce__direct_replicated_push__credit.h"
#include "coll/allreduce__one_shot_pull__trtllm.h"
#include "coll/allreduce__pull_reduce_scatter_push_allgather__hpcops.h"
#include "coll/allreduce__rabenseifner__mpich.h"
#include "coll/allreduce__recursive_doubling__clusterfusion.h"
#include "coll/allreduce__reduce_bcast_binomial__mpich.h"
#include "coll/allreduce__ring__nccl.h"
#include "coll/allreduce__shift_accumulate_atomic__clusterfusion.h"
#include "coll/allreduce__shift_accumulate_bulk__clusterfusion.h"
#include "coll/allreduce__two_shot_push__trtllm.h"
#include "coll/reducescatter__direct_pull__generic.h"
#include "coll/reducescatter__direct_push__generic.h"
#include "coll/reducescatter__recursive_halving__mpich.h"
#include "coll/reducescatter__ring__nccl.h"
#include "coll/alltoall__bruck__mpich.h"
#include "coll/alltoall__direct_pull__generic.h"
#include "coll/alltoall__direct_push__generic.h"
#include "coll/alltoall__pairwise_exchange__fa4.h"
#include "coll/alltoall__pairwise_xor__mpich.h"
#include "coll/alltoall__shift__generic.h"

// Op is a template parameter so that the run() of an illegal build sits in a
// dependent discarded statement and is never instantiated.
#define DSMEM_COLL_KERNEL(n)                                                               \
  template <class Op>                                                                      \
  __device__ void n##_k(kernel_arg_t* a) {                                                 \
    VX_CUDA_KERNEL_ENTRY(n);                                                               \
    (void)a;                                                                               \
    if constexpr (n::illegal<CLUSTER_N, MSG_ELEMS, ELEM_T>() == nullptr)                    \
      n::run<CLUSTER_N, ELEM_T, MSG_ELEMS, Op>((const ELEM_T*)a->in, (ELEM_T*)a->out,       \
                                               (unsigned long long*)a->ts, (unsigned*)a->smid); \
  }
DSMEM_COLL_ALGS(DSMEM_COLL_KERNEL)
#undef DSMEM_COLL_KERNEL

// The exported kernels carry the collectives' names, which are also their
// namespaces; C linkage keeps the symbol name, this namespace keeps them apart.
namespace dsmem_coll_kernels {
#define DSMEM_COLL_EXPORT(n) VX_CUDA_KERNEL_EXPORT(n, n##_k<dsmem::Sum>)
DSMEM_COLL_ALGS(DSMEM_COLL_EXPORT)
#undef DSMEM_COLL_EXPORT
} // namespace dsmem_coll_kernels
