// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Host side of the CUDA porting layer, over vortex2.h and dxa.h: a
// cudaLaunchKernelEx-shaped launch, cudaFuncSetAttribute, the cluster
// occupancy queries and the cuTensorMapEncode* calls, with CUDA's parameter
// order and enum names. A Vortex kernel takes one argument block, so the
// kernel's parameters are one trivially copyable struct.
//
// Every call returns a vx_result_t: VX_ERR_INVALID_VALUE where CUDA returns
// cudaErrorInvalidValue / CUDA_ERROR_INVALID_VALUE, VX_ERR_INVALID_HANDLE for
// a null kernel, and the codes of vortex2.h / dxa.h for what they check.
//
// The CUDA runtime acts on a current device; here the device is a parameter.

#ifndef __VX_CUDA_RUNTIME_H__
#define __VX_CUDA_RUNTIME_H__

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>

#include <vortex2.h>
#include <dxa.h>

// A kernel's __cluster_dims__ as a constant both sides see (the port's shared
// header); vx_cuda.h defines the same on the device side.
#ifndef VX_CUDA_CLUSTER_DIMS
#define VX_CUDA_CLUSTER_DIMS(name, X, Y, Z) \
  struct name##_cluster_dims { static constexpr uint32_t x = (X), y = (Y), z = (Z); }
#endif

namespace vx {
namespace cuda {

struct dim3 {
  uint32_t x, y, z;
  constexpr dim3(uint32_t x_ = 1, uint32_t y_ = 1, uint32_t z_ = 1) : x(x_), y(y_), z(z_) {}
};

constexpr bool operator==(const dim3& a, const dim3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
constexpr bool operator!=(const dim3& a, const dim3& b) { return !(a == b); }

// cudaLaunchConfig_t with its cluster-dimension and non-portable attributes as
// fields. cluster (0, 0, 0) is a launch without a cluster attribute
// (%is_explicit_cluster 0); (1, 1, 1) is an explicit cluster of one block.
struct launch_config_t {
  dim3     grid;
  dim3     block;
  dim3     cluster{0, 0, 0};
  uint32_t dyn_smem = 0;
  bool     non_portable = false;
};

// cudaFuncAttribute, with CUDA's values.
enum class func_attribute : int {
  max_dynamic_shared_memory_size   = 8,
  non_portable_cluster_size_allowed = 14,
};

// The dynamic shared memory a kernel takes without an opt-in (CUDA's 48 KB).
constexpr uint32_t kDefaultMaxDynamicSharedMemorySize = 48 * 1024;

namespace detail {

struct func_attrs {
  uint32_t max_dyn_smem = kDefaultMaxDynamicSharedMemorySize;
  bool     non_portable = false;
};

// Keyed by the handle and the kernel's entry address: a handle the runtime
// reuses for another kernel after vx_kernel_release starts from the defaults.
typedef std::pair<vx_kernel_h, uint64_t> func_key;

struct func_table {
  std::mutex mtx;
  std::map<func_key, func_attrs> attrs;
};

inline func_table& table() {
  static func_table t;
  return t;
}

inline func_key key_of(vx_kernel_h k) {
  uint64_t pc = 0;
  if (vx_kernel_address(k, &pc) != VX_SUCCESS)
    pc = ~0ull;
  return func_key(k, pc);
}

inline func_attrs get_attrs(vx_kernel_h k) {
  func_table& t = table();
  const func_key key = key_of(k);
  std::lock_guard<std::mutex> lock(t.mtx);
  auto it = t.attrs.find(key);
  return it == t.attrs.end() ? func_attrs() : it->second;
}

// No cluster from __cluster_dims__.
struct no_cluster_dims {};

template <typename D, typename = void>
struct has_cluster_dims : std::false_type {};

template <typename D>
struct has_cluster_dims<D, std::void_t<decltype(D::x), decltype(D::y), decltype(D::z)>> : std::true_type {};

constexpr bool unset_cluster(const dim3& c) {
  return c == dim3(0, 0, 0) || c == dim3(1, 1, 1);
}

// The cluster a launch takes: the kernel's __cluster_dims__ when it has
// them (a different explicit cluster is refused, as CUDA refuses to change a
// compile-time cluster size), else cfg.cluster.
template <typename ClusterDims>
inline vx_result_t resolve_cluster(const launch_config_t& cfg, dim3* out) {
  if constexpr (has_cluster_dims<ClusterDims>::value) {
    const dim3 c(ClusterDims::x, ClusterDims::y, ClusterDims::z);
    if (!unset_cluster(cfg.cluster) && cfg.cluster != c)
      return VX_ERR_INVALID_VALUE;
    *out = c;
  } else {
    static_assert(std::is_same<ClusterDims, no_cluster_dims>::value,
                  "the cluster template argument is a VX_CUDA_CLUSTER_DIMS type");
    *out = cfg.cluster;
  }
  return VX_SUCCESS;
}

inline void fill_info(vx_launch_info_t* li, vx_kernel_h kernel, const launch_config_t& cfg, const dim3& cl,
                      bool non_portable, const void* args, size_t args_size) {
  *li = vx_launch_info_t();
  li->struct_size = sizeof(*li);
  li->kernel      = kernel;
  li->args_host   = args;
  li->args_size   = args_size;
  const bool z = cfg.grid.z > 1 || cfg.block.z > 1 || cl.z > 1;
  const bool y = cfg.grid.y > 1 || cfg.block.y > 1 || cl.y > 1;
  li->ndim = z ? 3 : y ? 2 : 1;
  li->grid_dim[0]    = cfg.grid.x;
  li->grid_dim[1]    = cfg.grid.y;
  li->grid_dim[2]    = cfg.grid.z;
  li->block_dim[0]   = cfg.block.x;
  li->block_dim[1]   = cfg.block.y;
  li->block_dim[2]   = cfg.block.z;
  li->lmem_size      = cfg.dyn_smem;
  li->cluster_dim[0] = cl.x;
  li->cluster_dim[1] = cl.y;
  li->cluster_dim[2] = cl.z;
  li->cluster_policy = VX_CLUSTER_POLICY_DEFAULT;
  li->cluster_flags  = non_portable ? VX_CLUSTER_FLAG_NON_PORTABLE : 0;
}

template <typename ClusterDims>
inline vx_result_t make_info(vx_launch_info_t* li, vx_kernel_h kernel, const launch_config_t& cfg,
                             const void* args, size_t args_size) {
  if (!kernel)
    return VX_ERR_INVALID_HANDLE;
  dim3 cl;
  if (vx_result_t r = resolve_cluster<ClusterDims>(cfg, &cl))
    return r;
  const func_attrs a = get_attrs(kernel);
  fill_info(li, kernel, cfg, cl, cfg.non_portable || a.non_portable, args, args_size);
  return VX_SUCCESS;
}

} // namespace detail

// cudaFuncSetAttribute. max_dynamic_shared_memory_size records the kernel's
// dynamic shared memory limit; a value past the device's local memory per
// core (the most one block can hold) is refused at once.
// non_portable_cluster_size_allowed (0 or 1) sets the launches' default
// for launch_config_t::non_portable.
inline vx_result_t func_set_attribute(vx_device_h dev, vx_kernel_h kernel, func_attribute attr, int value) {
  if (!kernel)
    return VX_ERR_INVALID_HANDLE;
  if (!dev)
    return VX_ERR_INVALID_VALUE;
  detail::func_table& t = detail::table();
  const detail::func_key key = detail::key_of(kernel);
  switch (attr) {
  case func_attribute::max_dynamic_shared_memory_size: {
    uint64_t cap = 0;
    if (vx_result_t r = vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &cap))
      return r;
    if (value < 0 || (uint64_t)value > cap)
      return VX_ERR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(t.mtx);
    t.attrs[key].max_dyn_smem = (uint32_t)value;
    return VX_SUCCESS;
  }
  case func_attribute::non_portable_cluster_size_allowed: {
    if (value != 0 && value != 1)
      return VX_ERR_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(t.mtx);
    t.attrs[key].non_portable = (value != 0);
    return VX_SUCCESS;
  }
  }
  return VX_ERR_INVALID_VALUE;
}

// cudaFuncGetAttributes' maxDynamicSharedMemorySize, as recorded.
inline uint32_t func_max_dynamic_shared_memory_size(vx_kernel_h kernel) {
  return detail::get_attrs(kernel).max_dyn_smem;
}

// cudaLaunchKernelEx. launch_kernel_ex<name_cluster_dims>(...) launches a
// kernel declared with VX_CUDA_CLUSTER_DIMS(name, ...): its cluster comes
// from the constant. cfg.dyn_smem past the kernel's recorded
// max_dynamic_shared_memory_size is refused before the launch; the grid
// must be a whole number of clusters in each dimension (vx_enqueue_launch
// refuses it otherwise).
template <typename ClusterDims = detail::no_cluster_dims, typename Args>
inline vx_result_t launch_kernel_ex(vx_queue_h queue, vx_kernel_h kernel, const launch_config_t& cfg,
                                    const Args& args, vx_event_h* event = nullptr) {
  static_assert(std::is_trivially_copyable<Args>::value, "the kernel argument block is copied as bytes");
  if (!queue)
    return VX_ERR_INVALID_HANDLE;
  vx_launch_info_t li;
  if (vx_result_t r = detail::make_info<ClusterDims>(&li, kernel, cfg, &args, sizeof(Args)))
    return r;
  if (cfg.dyn_smem > detail::get_attrs(kernel).max_dyn_smem)
    return VX_ERR_INVALID_VALUE;
  return vx_enqueue_launch(queue, &li, 0, nullptr, event);
}

// cudaOccupancyMaxActiveClusters: clusters of cfg's shape resident at once.
// Without a cluster (no __cluster_dims__, cfg.cluster (0, 0, 0)) there is
// nothing to count: VX_ERR_INVALID_VALUE, as CUDA refuses it.
template <typename ClusterDims = detail::no_cluster_dims>
inline vx_result_t occupancy_max_active_clusters(vx_device_h dev, vx_kernel_h kernel, const launch_config_t& cfg,
                                                 int* num_clusters) {
  if (!num_clusters)
    return VX_ERR_INVALID_VALUE;
  vx_launch_info_t li;
  if (vx_result_t r = detail::make_info<ClusterDims>(&li, kernel, cfg, nullptr, 0))
    return r;
  if (li.cluster_dim[0] == 0 && li.cluster_dim[1] == 0 && li.cluster_dim[2] == 0)
    return VX_ERR_INVALID_VALUE;
  uint32_t n = 0;
  if (vx_result_t r = vx_occupancy_max_active_clusters(dev, &li, &n))
    return r;
  *num_clusters = (int)n;
  return VX_SUCCESS;
}

// cudaOccupancyMaxPotentialClusterSize: the largest cluster that fits; the
// config's cluster shape is ignored. A kernel with __cluster_dims__ gets its
// required size, as CUDA returns it.
template <typename ClusterDims = detail::no_cluster_dims>
inline vx_result_t occupancy_max_potential_cluster_size(vx_device_h dev, vx_kernel_h kernel,
                                                        const launch_config_t& cfg, int* cluster_size) {
  if (!cluster_size)
    return VX_ERR_INVALID_VALUE;
  vx_launch_info_t li;
  if (vx_result_t r = detail::make_info<detail::no_cluster_dims>(&li, kernel, cfg, nullptr, 0))
    return r;
  if constexpr (detail::has_cluster_dims<ClusterDims>::value) {
    (void)dev;
    *cluster_size = (int)(ClusterDims::x * ClusterDims::y * ClusterDims::z);
    return VX_SUCCESS;
  } else {
    uint32_t n = 0;
    if (vx_result_t r = vx_occupancy_max_potential_cluster_size(dev, &li, &n))
      return r;
    *cluster_size = (int)n;
    return VX_SUCCESS;
  }
}

// ── Tensor maps (cuTensorMapEncode*, CUDA driver API names) ─────────────

typedef vx_tensor_map_t CUtensorMap;

enum CUtensorMapDataType : uint32_t {
  CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
  CU_TENSOR_MAP_DATA_TYPE_UINT16,
  CU_TENSOR_MAP_DATA_TYPE_UINT32,
  CU_TENSOR_MAP_DATA_TYPE_INT32,
  CU_TENSOR_MAP_DATA_TYPE_UINT64,
  CU_TENSOR_MAP_DATA_TYPE_INT64,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ,
  CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
  CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
  CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B,
};

enum CUtensorMapInterleave : uint32_t {
  CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
  CU_TENSOR_MAP_INTERLEAVE_16B,
  CU_TENSOR_MAP_INTERLEAVE_32B,
};

enum CUtensorMapSwizzle : uint32_t {
  CU_TENSOR_MAP_SWIZZLE_NONE = 0,
  CU_TENSOR_MAP_SWIZZLE_32B,
  CU_TENSOR_MAP_SWIZZLE_64B,
  CU_TENSOR_MAP_SWIZZLE_128B,
  CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
  CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
  CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,
};

enum CUtensorMapL2promotion : uint32_t {
  CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
  CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
  CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
};

enum CUtensorMapFloatOOBfill : uint32_t {
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
};

// cuTensorMapEncodeTiled. global_address is a device address
// (vx_buffer_address); global_strides are in bytes, of dims 1..rank-1.
inline vx_result_t tensor_map_encode_tiled(CUtensorMap* tensor_map, CUtensorMapDataType tensor_data_type,
                                           uint32_t tensor_rank, uint64_t global_address,
                                           const uint64_t* global_dim, const uint64_t* global_strides,
                                           const uint32_t* box_dim, const uint32_t* element_strides,
                                           CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
                                           CUtensorMapL2promotion l2_promotion,
                                           CUtensorMapFloatOOBfill oob_fill) {
  namespace d = vortex::dxa;
  return (vx_result_t)d::encode_tiled(tensor_map, (d::TensorMapDataType)tensor_data_type, tensor_rank,
                                      global_address, global_dim, global_strides, box_dim, element_strides,
                                      (d::TensorMapInterleave)interleave, (d::TensorMapSwizzle)swizzle,
                                      (d::TensorMapL2Promotion)l2_promotion,
                                      (d::TensorMapFloatOOBfill)oob_fill);
}

// cuTensorMapEncodeIm2col: rank 3-5, dims C, W[, H[, D]], N; the pixel-box
// corners in order W, H, D.
inline vx_result_t tensor_map_encode_im2col(CUtensorMap* tensor_map, CUtensorMapDataType tensor_data_type,
                                            uint32_t tensor_rank, uint64_t global_address,
                                            const uint64_t* global_dim, const uint64_t* global_strides,
                                            const int* pixel_box_lower_corner, const int* pixel_box_upper_corner,
                                            uint32_t channels_per_pixel, uint32_t pixels_per_column,
                                            const uint32_t* element_strides, CUtensorMapInterleave interleave,
                                            CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2_promotion,
                                            CUtensorMapFloatOOBfill oob_fill) {
  namespace d = vortex::dxa;
  static_assert(sizeof(int) == sizeof(int32_t), "the pixel-box corners are 32-bit");
  return (vx_result_t)d::encode_im2col(tensor_map, (d::TensorMapDataType)tensor_data_type, tensor_rank,
                                       global_address, global_dim, global_strides,
                                       reinterpret_cast<const int32_t*>(pixel_box_lower_corner),
                                       reinterpret_cast<const int32_t*>(pixel_box_upper_corner),
                                       channels_per_pixel, pixels_per_column, element_strides,
                                       (d::TensorMapInterleave)interleave, (d::TensorMapSwizzle)swizzle,
                                       (d::TensorMapL2Promotion)l2_promotion,
                                       (d::TensorMapFloatOOBfill)oob_fill);
}

// cuTensorMapReplaceAddress.
inline vx_result_t tensor_map_replace_address(CUtensorMap* tensor_map, uint64_t global_address) {
  return (vx_result_t)vortex::dxa::replace_address(tensor_map, global_address);
}

} // namespace cuda
} // namespace vx

#endif // __VX_CUDA_RUNTIME_H__
