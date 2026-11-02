#include <vx_spawn2.h>
#include <vx_dlmem.h>
#include "common.h"

static inline uint32_t linear_block() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

// One lane per CTA records what the CTA sees of its cluster; with -m it also
// maps one of its local-memory objects into every rank of the cluster.
__kernel void rank_k(rank_arg_t* __UNIFORM__ arg) {
  if (threadIdx.x != 0)
    return;
  const uint32_t b = linear_block();
  auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + b * REC_WORDS;
  auto cluster = vortex::this_cluster();
  rec[REC_BLOCK_X]    = blockIdx.x;
  rec[REC_BLOCK_Y]    = blockIdx.y;
  rec[REC_BLOCK_Z]    = blockIdx.z;
  rec[REC_CTA_ID]     = get_local_group_id();
  rec[REC_RANK]       = vx_cluster_block_rank();
  rec[REC_GROUP_RANK] = cluster.block_rank();
  rec[REC_SPAWN_RANK] = get_cluster_rank();
  rec[REC_NUM_BLOCKS] = vx_cluster_num_blocks();
  if (arg->map_addr) {
    auto map = reinterpret_cast<uint64_t*>(arg->map_addr) + (uint64_t)b * arg->num_blocks;
    auto obj = reinterpret_cast<uint64_t*>(
        static_cast<char*>(__local_mem()) + map_offset(b, arg->lmem_bytes));
    for (uint32_t r = 0; r < arg->num_blocks; ++r)
      map[r] = (uint64_t)(uintptr_t)cluster.map_shared_rank(obj, r);
  }
}

// Every thread of the CTA writes its share of the whole local-memory slice,
// then, after the barrier, checks the share another warp wrote.
__kernel void lmem_k(lmem_arg_t* __UNIFORM__ arg) {
  const uint32_t tid = threadIdx.x;
  const uint32_t nthreads = blockDim.x;
  const uint32_t b = linear_block();
  const uint32_t n = arg->lmem_bytes / sizeof(uint64_t);
  auto slice = reinterpret_cast<volatile uint64_t*>(__local_mem());
  for (uint32_t i = tid; i < n; i += nthreads)
    slice[i] = lmem_pattern(b, i);
  __syncthreads();
  uint32_t errors = 0, first = 0xffffffffu;
  for (uint32_t i = tid; i < n; i += nthreads) {
    const uint32_t j = n - 1 - i;
    if (slice[j] != lmem_pattern(b, j)) {
      if (errors == 0)
        first = j;
      ++errors;
    }
  }
  auto err = reinterpret_cast<uint32_t*>(arg->err_addr) + 2 * (b * nthreads + tid);
  err[0] = errors;
  err[1] = first;
  if (tid == 0)
    reinterpret_cast<uint64_t*>(arg->base_addr)[b] = (uint64_t)(uintptr_t)slice;
}

static void ids_record(ids_arg_t* arg, uint32_t b, uint32_t rank, uint32_t k, uint64_t* obj,
                       uint32_t cl_trank, uint64_t t0);

// Every PTX cluster identity register and address op, recorded for the host model.
__kernel void ids_k(ids_arg_t* __UNIFORM__ arg) {
  const uint64_t t0 = vx_globaltimer();
  const uint32_t b = linear_block();
  auto cluster = vortex::this_cluster();
  const uint32_t rank = vx_cluster_block_rank();
  const uint32_t k = cluster.num_blocks();
  auto obj = reinterpret_cast<uint64_t*>(static_cast<char*>(__local_mem()) + IDS_OBJ_OFFSET);
  if (threadIdx.x == 0)
    *obj = ids_tag(b);
  const uint32_t cl_trank = cluster.thread_rank();
  cluster.sync();
  if (threadIdx.x == 0)
    ids_record(arg, b, rank, k, obj, cl_trank, t0);
  // No block may leave while a peer still reads its object.
  cluster.sync();
}

static void ids_record(ids_arg_t* arg, uint32_t b, uint32_t rank, uint32_t k, uint64_t* obj,
                       uint32_t cl_trank, uint64_t t0) {
  auto cluster = vortex::this_cluster();
  const uint32_t next = (rank + 1) % k;
  auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + b * IDS_WORDS;
  rec[IDS_BLOCK + 0]      = blockIdx.x;
  rec[IDS_BLOCK + 1]      = blockIdx.y;
  rec[IDS_BLOCK + 2]      = blockIdx.z;
  rec[IDS_CTAID + 0]      = vx_cluster_ctaid_x();
  rec[IDS_CTAID + 1]      = vx_cluster_ctaid_y();
  rec[IDS_CTAID + 2]      = vx_cluster_ctaid_z();
  rec[IDS_NCTAID + 0]     = vx_cluster_nctaid_x();
  rec[IDS_NCTAID + 1]     = vx_cluster_nctaid_y();
  rec[IDS_NCTAID + 2]     = vx_cluster_nctaid_z();
  rec[IDS_CLUSTERID + 0]  = vx_clusterid_x();
  rec[IDS_CLUSTERID + 1]  = vx_clusterid_y();
  rec[IDS_CLUSTERID + 2]  = vx_clusterid_z();
  rec[IDS_NCLUSTERID + 0] = vx_nclusterid_x();
  rec[IDS_NCLUSTERID + 1] = vx_nclusterid_y();
  rec[IDS_NCLUSTERID + 2] = vx_nclusterid_z();
  rec[IDS_EXPLICIT]       = vx_is_explicit_cluster();
  rec[IDS_DYN_SMEM]       = vx_dynamic_smem_size();
  rec[IDS_TOTAL_SMEM]     = vx_total_smem_size();
  rec[IDS_AGGR_SMEM]      = vx_aggr_smem_size();
  rec[IDS_RESERVED]       = vx_reserved_smem_offset();
  rec[IDS_RANK]           = rank;

  const uint32_t a = vx_cvta_shared(obj);
  const uint32_t m = vx_mapa_u32(a, next);
  uint64_t* window = static_cast<uint64_t*>(cluster.map_shared_rank(obj, next));
  uint64_t* generic = static_cast<uint64_t*>(vx_cvta_cluster_to_generic(m));
  rec[IDS_CVTA]        = a;
  rec[IDS_RANK_LOCAL]  = vx_getctarank(obj);
  rec[IDS_RANK_WINDOW] = vx_getctarank(window);
  rec[IDS_RANK_U32]    = vx_getctarank_u32(m);
  rec[IDS_MAPA]        = m;
  rec[IDS_PEER_VALUE]  = (uint32_t)*reinterpret_cast<volatile uint64_t*>(generic);
  rec[IDS_CG_THREADS]  = cluster.num_threads();
  rec[IDS_CG_TRANK]    = cl_trank;
  rec[IDS_CG_DIMTX]    = cluster.dim_threads().x;
  rec[IDS_CG_QRANK]    = cluster.query_shared_rank(window);

  const vortex::dim3 dims = cluster.dim_blocks();
  const vortex::dim3 idx  = cluster.block_index();
  uint32_t flags = 0;
  flags |= vx_isspacep_shared_cluster(obj) ? IDS_F_SPACEP_LOCAL : 0;
  flags |= vx_isspacep_shared_cluster(window) ? IDS_F_SPACEP_WINDOW : 0;
  flags |= vx_isspacep_shared_cluster(rec) ? IDS_F_SPACEP_GLOBAL : 0;
  flags |= vx_isspacep_shared(obj) ? IDS_F_CTA_LOCAL : 0;
  flags |= vx_isspacep_shared(window) ? IDS_F_CTA_WINDOW : 0;
  flags |= (generic == window) ? IDS_F_GENERIC : 0;
  flags |= (vx_mapa(window, rank) == cluster.map_shared_rank(obj, rank)) ? IDS_F_MAPA_WINDOW : 0;
  flags |= (dims.x == vx_cluster_nctaid_x() && dims.y == vx_cluster_nctaid_y()
         && dims.z == vx_cluster_nctaid_z() && idx.x == vx_cluster_ctaid_x()
         && idx.y == vx_cluster_ctaid_y() && idx.z == vx_cluster_ctaid_z()) ? IDS_F_CG_DIMS : 0;
  flags |= (vx_globaltimer() > t0) ? IDS_F_TIMER : 0;
  flags |= cluster.is_valid() ? IDS_F_VALID : 0;
  rec[IDS_FLAGS] = flags;
}
