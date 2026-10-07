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

# Convenience wrapper around the Vortex dev container (Dockerfile.dev).
# Source is bind-mounted from your checkout (hot reload); the build dir lives
# on local disk so compiles don't hammer a shared/NFS file server.
# See miscs/docker/README.md for the full reference.

set -euo pipefail

# --- configuration (override via environment) -------------------------------

# Repo root, derived from this script's location (miscs/docker/vortex-dev.sh).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
VORTEX_SRC="${VORTEX_SRC:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"

IMAGE="${IMAGE:-vortex-dev}"
CONTAINER="${CONTAINER:-vortex-dev}"
CCACHE_VOL="${CCACHE_VOL:-vortex-ccache}"
XLEN="${XLEN:-32}"
# Build root on local disk (regenerable; keeps heavy compile writes off NFS).
VORTEX_BUILD="${VORTEX_BUILD:-/scratch/$USER/vortex-build}"
# Optional subdirectory of the build root to use as the build dir, so one
# mount can hold several build dirs (e.g. build32, build64). Empty: the
# build root itself is the build dir.
BUILD_NAME="${BUILD_NAME:-}"

# Container-internal paths (must match dev-entrypoint.sh / README).
SRC_MNT=/vortex
BUILD_MNT=/build_docker
BUILD_DIR="$BUILD_MNT${BUILD_NAME:+/$BUILD_NAME}"

# --- helpers ----------------------------------------------------------------

err()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
info() { printf '\033[36m%s\033[0m\n' "$*"; }

exists()  { [ -n "$(docker ps -aq -f "name=^${CONTAINER}$")" ]; }
running() { [ -n "$(docker ps -q  -f "name=^${CONTAINER}$")" ]; }

# Allocate a TTY only when attached to one, so piped/non-interactive use of
# 'make'/'exec' doesn't warn "the input device is not a TTY".
tty_flag() { if [ -t 0 ] && [ -t 1 ]; then echo "-it"; else echo "-i"; fi; }

require_image() {
    if [ -z "$(docker images -q "$IMAGE")" ]; then
        err "Image '$IMAGE' not found. Build it first: $0 build"
        exit 1
    fi
}

# --- commands ---------------------------------------------------------------

cmd_build() {
    # Stage a minimal context: the image installs the toolchain from this
    # checkout's own installers, so it always matches its TOOLCHAIN_REV.
    (
        ctx="$(mktemp -d)"
        trap 'rm -rf "$ctx"' EXIT
        mkdir -p "$ctx/bootstrap"
        cp -p "$SCRIPT_DIR/Dockerfile.dev" "$SCRIPT_DIR/dev-entrypoint.sh" "$ctx/"
        cp -p "$VORTEX_SRC/VERSION" "$VORTEX_SRC/configure" "$VORTEX_SRC"/VX_*.toml "$ctx/bootstrap/"
        mkdir -p "$ctx/bootstrap/ci"
        cp -p "$VORTEX_SRC"/ci/*.sh "$VORTEX_SRC"/ci/*.in "$VORTEX_SRC"/ci/*.py "$ctx/bootstrap/ci/"
        info "Building image '$IMAGE' from $SCRIPT_DIR/Dockerfile.dev ($(grep -m1 '^TOOLCHAIN_REV' "$VORTEX_SRC/VERSION")) ..."
        docker build --platform=linux/amd64 -t "$IMAGE" \
            -f "$ctx/Dockerfile.dev" "$ctx" "$@"
    )
}

cmd_up() {
    require_image
    if running; then
        info "Container '$CONTAINER' is already running."
        return 0
    fi
    if exists; then
        info "Starting existing container '$CONTAINER' ..."
        docker start "$CONTAINER" >/dev/null
        return 0
    fi

    mkdir -p "$VORTEX_BUILD"
    info "Creating container '$CONTAINER'"
    info "  source : $VORTEX_SRC  -> $SRC_MNT   (hot reload, NFS ok)"
    info "  build  : $VORTEX_BUILD -> $BUILD_MNT (local disk), build dir $BUILD_DIR"
    info "  xlen   : $XLEN"
    docker run -dit --name "$CONTAINER" \
        --platform=linux/amd64 --privileged=true \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp -e CONFIG_XLEN="$XLEN" -e BUILD_DIR="$BUILD_DIR" \
        -v "$VORTEX_SRC":"$SRC_MNT" \
        -v "$VORTEX_BUILD":"$BUILD_MNT" \
        -v "$CCACHE_VOL":/ccache \
        -w "$SRC_MNT" \
        "$IMAGE" >/dev/null
    info "Up. First start configures $BUILD_DIR (see '$0 logs')."
}

cmd_shell() {
    cmd_up
    docker exec -it -w "$BUILD_DIR" "$CONTAINER" bash
}

cmd_exec() {
    if [ "$#" -eq 0 ]; then err "usage: $0 exec <command...>"; exit 1; fi
    cmd_up
    docker exec "$(tty_flag)" -w "$BUILD_DIR" "$CONTAINER" bash -lc "$*"
}

cmd_make() {
    cmd_up
    docker exec "$(tty_flag)" -w "$BUILD_DIR" "$CONTAINER" make -s "$@"
}

cmd_stop()    { running && { info "Stopping '$CONTAINER' ..."; docker stop "$CONTAINER" >/dev/null; } || info "Not running."; }
cmd_restart() { exists && { info "Restarting '$CONTAINER' ..."; docker restart "$CONTAINER" >/dev/null; } || cmd_up; }
cmd_logs()    { exists && docker logs "$@" "$CONTAINER" || err "No container '$CONTAINER'."; }

cmd_status() {
    echo "image     : $IMAGE  ($(docker images -q "$IMAGE" >/dev/null 2>&1 && docker images "$IMAGE" --format '{{.Size}}' | head -1 || echo absent))"
    echo "container : $(running && echo running || { exists && echo stopped || echo absent; })"
    echo "source    : $VORTEX_SRC"
    echo "build root: $VORTEX_BUILD"
    echo "build dir : $BUILD_DIR (in container)"
    docker ps -a -f "name=^${CONTAINER}$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' 2>/dev/null || true
}

cmd_rm() { exists && { info "Removing container '$CONTAINER' (build dir + ccache kept) ..."; docker rm -f "$CONTAINER" >/dev/null; } || info "No container to remove."; }

cmd_rebuild() { cmd_rm; cmd_build "$@"; }

cmd_clean() {
    cmd_rm
    if [ -d "$VORTEX_BUILD" ]; then
        info "Wiping build root $VORTEX_BUILD ..."
        rm -rf "${VORTEX_BUILD:?}/"* "${VORTEX_BUILD:?}/".[!.]* 2>/dev/null || true
    fi
    info "Next '$0 up' reconfigures from scratch."
}

usage() {
    cat <<EOF
Vortex dev container manager

Usage: $0 <command> [args]

Lifecycle:
  build [docker-build-args]   Build the dev image ($IMAGE) from this checkout's installers.
  up                          Create+start (or resume) the container.
  shell                       Open a shell in the container (starts it first).
  stop                        Stop the container (keeps it + build dir).
  restart                     Restart the container.
  rm                          Remove the container (build dir + ccache persist).
  rebuild [docker-build-args] Remove container, then rebuild the image.
  clean                       Remove container AND wipe the build root.

Work:
  make [args]                 Run 'make -s [args]' in the build dir.
  exec <command...>           Run a command in the build dir (login shell).

Info:
  status                      Show image/container/paths.
  logs [-f]                   Show entrypoint/container logs.

Overridable env: VORTEX_SRC, VORTEX_BUILD, BUILD_NAME, IMAGE, CONTAINER, XLEN, CCACHE_VOL
  current: src=$VORTEX_SRC build=$VORTEX_BUILD${BUILD_NAME:+ (subdir $BUILD_NAME)} xlen=$XLEN

Examples:
  $0 build && $0 up && $0 shell
  $0 make
  $0 exec './ci/blackbox.sh --cores=2 --app=vecadd'
  $0 exec 'mkdir -p $BUILD_MNT/build64 && cd $BUILD_MNT/build64 && $SRC_MNT/configure --xlen=64 --tooldir=/opt'
  BUILD_NAME=build64 $0 make
  $0 stop ; $0 restart ; $0 rm
EOF
}

# --- dispatch ---------------------------------------------------------------

cmd="${1:-help}"; shift || true
case "$cmd" in
    build)        cmd_build "$@";;
    up|start)     cmd_up;;
    shell|sh|bash) cmd_shell;;
    exec)         cmd_exec "$@";;
    make)         cmd_make "$@";;
    stop)         cmd_stop;;
    restart)      cmd_restart;;
    logs)         cmd_logs "$@";;
    status|ps)    cmd_status;;
    rm|down)      cmd_rm;;
    rebuild)      cmd_rebuild "$@";;
    clean)        cmd_clean;;
    help|-h|--help) usage;;
    *) err "Unknown command: $cmd"; echo; usage; exit 1;;
esac
