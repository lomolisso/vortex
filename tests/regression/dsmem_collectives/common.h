#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// N, M and T of the build, as one H100 binary per (N, M, T).
#ifndef CLUSTER_N
#define CLUSTER_N 4
#endif
#ifndef MSG_ELEMS
#define MSG_ELEMS 1024
#endif
#ifndef ELEM_T
#define ELEM_T float
#endif

// Rank b of the one cluster (N blocks of 256 threads), for k < M:
//   allreduce      out[b*M + k]         = sum over r of in[r*M + k]
//   allgather      out[b*N*M + r*M + k] = in[r*M + k]
//   reducescatter  out[b*M + k]         = sum over r of in[r*N*M + b*M + k]
//   alltoall       out[b*N*M + r*M + k] = in[r*N*M + b*M + k]
// Rank b's stamps are ts[b*32 + i]: i = 0 entry, 1 exit, 2 + r after round r;
// smid[b] is its core.
typedef struct {
  uint64_t in;
  uint64_t out;
  uint64_t ts;
  uint64_t smid;
  uint32_t alg;
} kernel_arg_t;

// The 29 collectives, one kernel each, named <op>__<alg>.
#define DSMEM_COLL_ALGS(X)                         \
  X(allgather__bruck)                              \
  X(allgather__direct_pull)                        \
  X(allgather__direct_replicated_push)             \
  X(allgather__direct_shift)                       \
  X(allgather__gather_bcast_binomial)              \
  X(allgather__recursive_doubling)                 \
  X(allgather__ring)                               \
  X(allreduce__2d_mesh)                            \
  X(allreduce__atomic_all_push)                    \
  X(allreduce__direct_replicated_push)             \
  X(allreduce__one_shot_pull)                      \
  X(allreduce__pull_reduce_scatter_push_allgather) \
  X(allreduce__rabenseifner)                       \
  X(allreduce__recursive_doubling)                 \
  X(allreduce__reduce_bcast_binomial)              \
  X(allreduce__ring)                               \
  X(allreduce__shift_accumulate_atomic)            \
  X(allreduce__shift_accumulate_bulk)              \
  X(allreduce__two_shot_push)                      \
  X(reducescatter__direct_pull)                    \
  X(reducescatter__direct_push)                    \
  X(reducescatter__recursive_halving)              \
  X(reducescatter__ring)                           \
  X(alltoall__bruck)                               \
  X(alltoall__direct_pull)                         \
  X(alltoall__direct_push)                         \
  X(alltoall__pairwise_exchange)                   \
  X(alltoall__pairwise_xor)                        \
  X(alltoall__shift)

#define DSMEM_COLL_NAME(n) #n,
static const char* const kAlgNames[] = {DSMEM_COLL_ALGS(DSMEM_COLL_NAME)};
#undef DSMEM_COLL_NAME

constexpr int kNumAlgs = (int)(sizeof(kAlgNames) / sizeof(kAlgNames[0]));

// Every kernel is __cluster_dims__(CLUSTER_N, 1, 1).
#define DSMEM_COLL_DIMS(n) VX_CUDA_CLUSTER_DIMS(n, CLUSTER_N, 1, 1);
DSMEM_COLL_ALGS(DSMEM_COLL_DIMS)
#undef DSMEM_COLL_DIMS

#endif
