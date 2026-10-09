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

`ifndef VX_DLMEM_PKG_VH
`define VX_DLMEM_PKG_VH

`include "VX_define.vh"

package VX_dlmem_pkg;

    import VX_gpu_pkg::*;

    // Cores of one hardware cluster: the endpoints of its DLMEM network.
    localparam DLMEM_NCC    = NUM_SOCKETS * `VX_CFG_SOCKET_SIZE;
    localparam DLMEM_CORE_W = `UP(`CLOG2(DLMEM_NCC));

endpackage

`endif // VX_DLMEM_PKG_VH
