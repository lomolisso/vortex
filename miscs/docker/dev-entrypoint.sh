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

# Entrypoint for the Vortex dev container. Idempotently prepares an
# out-of-tree build directory against the bind-mounted source at /vortex so
# that `make -s` works immediately, then hands off to the requested command
# (default: an interactive shell).

set -e

# The container usually runs as the host UID, which may not exist in
# /etc/passwd; give tools (ccache, git) a writable HOME and cache regardless.
: "${HOME:=/tmp}"
export HOME
: "${CCACHE_DIR:=/ccache}"
export CCACHE_DIR
mkdir -p "$CCACHE_DIR" 2>/dev/null || true

VORTEX_SRC=/vortex
# Out-of-tree build dir. Mount a local-disk path here (e.g. /scratch/$USER/...)
# to keep heavy compile writes off a shared/NFS source tree; defaults to a dir
# inside the checkout when no separate mount is provided.
BUILD_DIR="${BUILD_DIR:-$VORTEX_SRC/build_docker}"
XLEN="${XLEN:-32}"
TOOLDIR="${TOOLDIR:-/opt}"

if [ -f "$VORTEX_SRC/configure" ]; then
    if [ ! -f "$BUILD_DIR/config.mk" ]; then
        echo "[dev-entrypoint] Configuring $BUILD_DIR (xlen=$XLEN, tooldir=$TOOLDIR)..."
        mkdir -p "$BUILD_DIR"
        ( cd "$BUILD_DIR" && bash "$VORTEX_SRC/configure" --xlen="$XLEN" --tooldir="$TOOLDIR" )
        echo "[dev-entrypoint] Done. Build with: cd $BUILD_DIR && make -s"
    fi
else
    echo "[dev-entrypoint] WARNING: $VORTEX_SRC/configure not found."
    echo "[dev-entrypoint] Bind-mount the Vortex source at $VORTEX_SRC (see miscs/docker/README.md)."
fi

# Omelet (NoC co-simulator), bind-mounted at /omelet when available. The
# package is importable via the image's PYTHONPATH (no pip install needed);
# only the patched-gem5 build is heavy and stays manual: omelet-setup.sh.
OMELET_SRC="${OMELET_SRC:-/omelet}"
if [ -f "$OMELET_SRC/pyproject.toml" ] && \
   [ ! -x "$OMELET_SRC/gem5/build/Garnet_standalone/gem5.opt" ]; then
    echo "[dev-entrypoint] Omelet gem5 not built yet; run: $VORTEX_SRC/miscs/docker/omelet-setup.sh"
fi

exec "$@"
