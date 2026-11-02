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

`include "VX_define.vh"

interface VX_dcr_csr_if import VX_gpu_pkg::*; ();

    wire                            valid;
    wire [`VX_CSR_ADDR_BITS-1:0]    addr;
    wire [7:0]                      mpm_class;
    wire [VX_DCR_DATA_WIDTH-1:0]    value;
    wire                            ready;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // the core's copy of the launch DCRs behind the cluster identity CSRs
    wire [2:0][CLUSTER_SIZE_W-1:0]  cluster_dim;
    wire [`VX_CFG_LMEM_LOG_SIZE:0]  lmem_size;
    wire                            cluster_explicit;
`endif

    modport master (
        output valid,
        output addr,
        output mpm_class,
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        output cluster_dim,
        output lmem_size,
        output cluster_explicit,
    `endif
        input  value,
        input  ready
    );

    modport slave (
        input  valid,
        input  addr,
        input  mpm_class,
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        input  cluster_dim,
        input  lmem_size,
        input  cluster_explicit,
    `endif
        output value,
        output ready
    );

endinterface
