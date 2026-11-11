// The six CREDIT cluster kernels the H100 ran, each at its H100 template
// arguments <kBlockThreads = 256, kMaxClusterSize = 8>, exported by name.

#include <vx_cuda.h>
#include "common.h"
#include "layernorm.h"
#include "weighted_var.h"
#include "pearson.h"
#include "softmax.h"
#include "lars.h"
#include "rowwise.h"

using credit::kBlockThreads;
using credit::kMaxClusterSize;

// The exported names are also the kernels' namespaces: C linkage keeps the
// symbol name, this namespace keeps the two apart.
namespace credit_kernels {
VX_CUDA_KERNEL_EXPORT(layernorm, layernorm::cluster_bwd_staged_kernel<kBlockThreads, kMaxClusterSize>)
VX_CUDA_KERNEL_EXPORT(weighted_var, weighted_var::cluster_bwd_staged_kernel<kBlockThreads, kMaxClusterSize>)
VX_CUDA_KERNEL_EXPORT(pearson, pearson::cluster_bwd_staged_kernel<kBlockThreads, kMaxClusterSize>)
VX_CUDA_KERNEL_EXPORT(softmax, softmax::cluster_softmax_logits_bwd_staged_kernel<kBlockThreads, kMaxClusterSize>)
VX_CUDA_KERNEL_EXPORT(lars, lars::cluster_lars_staged_kernel<kBlockThreads, kMaxClusterSize>)
VX_CUDA_KERNEL_EXPORT(rowwise, rowwise::cluster_quant_staged_kernel<kBlockThreads, kMaxClusterSize>)
} // namespace credit_kernels
