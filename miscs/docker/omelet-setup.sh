#!/usr/bin/env bash

# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# One-time (idempotent) Omelet provisioning inside the Vortex dev container:
# applies the Garnet patches, installs the omelet Python package, and builds
# gem5 Garnet_standalone in the bind-mounted Omelet tree (/omelet). Thin
# wrapper over Omelet's own scripts/install.sh, which guards every step, so
# re-running after a patch or gem5-SimObject change is safe and incremental.
#
# Usage (host):  miscs/docker/vortex-dev.sh exec '/vortex/miscs/docker/omelet-setup.sh'
# Usage (inside the container):  /vortex/miscs/docker/omelet-setup.sh

set -euo pipefail

OMELET_SRC="${OMELET_SRC:-/omelet}"

if [ ! -f "$OMELET_SRC/scripts/install.sh" ]; then
    echo "ERROR: no Omelet checkout at $OMELET_SRC (bind-mount it; see vortex-dev.sh)" >&2
    exit 1
fi

export OMELET_BUILD_JOBS="${OMELET_BUILD_JOBS:-$(nproc)}"
bash "$OMELET_SRC/scripts/install.sh"

echo ""
echo "[omelet-setup] Smoke test:"
echo "  cd $OMELET_SRC && PYTHONPATH=$OMELET_SRC omelet-run examples/config/2_5d_org_mesh.yaml --injectionrate 0.1"
