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

interface VX_dlmem_bus_if #(
    parameter DATAW = 1,
    parameter DST_W = 1
) ();

    logic             valid;
    logic [DATAW-1:0] data;
    logic [DST_W-1:0] dst;
    logic             ready;

    modport master (
        output valid,
        output data,
        output dst,
        input  ready
    );

    modport slave (
        input  valid,
        input  data,
        input  dst,
        output ready
    );

endinterface
