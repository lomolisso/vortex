# Vortex in Docker

Two images are provided:

- **`Dockerfile.dev`** — a source-less developer image with the full non-ASIC
  toolchain in `/opt`. You bind-mount your host checkout, so edits on the host
  are seen live in the container (hot reload). Use this for day-to-day work.
- **`Dockerfile.prod`** — a self-contained packaged image that bakes a Vortex
  checkout + build inside it. Use this for a throwaway, mount-free environment.

Both install everything except the ASIC EDA tools (Synopsys/Cadence), which
are not redistributable — run those flows on the host. Everything else is
included: SimX, rtlsim/verilator, RISC-V LLVM+GCC, POCL/OpenCL, chipstar/HIP,
mesa/Vulkan, gem5, and SST.

Install Docker first:

- PC: https://docs.docker.com/desktop/install/windows-install
- MAC: https://docs.docker.com/desktop/install/mac-install
- Ubuntu: https://docs.docker.com/desktop/install/ubuntu

Note: Apple Silicon Macs run the container under emulation, so building and
running will be considerably slower — but it still works.

> **Daemon must be able to read your checkout (important).** The dev flow
> bind-mounts your source into the container, and the Docker daemon (root) must
> be able to traverse to it. On shared servers, a home on **NFS with
> root-squash** maps the daemon's root to `nobody`, which cannot traverse a
> `0700` home — bind mounts of `~/...` then fail with `permission denied`.
> The fix keeps your code exactly where it is (no copying to scratch): grant
> *traverse-only* access to your home once with
>
> ```bash
> chmod 711 ~          # adds o+x (traverse), NOT o+r (list)
> ```
>
> The repo's own dirs are already group/other-readable, and the container runs
> as your UID (`--user $(id -u):$(id -g)`), so all NFS reads/writes happen as
> you, never squashed. `711` lets other local users enter *known* paths under
> your home but not list it; if that is unacceptable, the alternative is a
> daemon-readable local-disk checkout, but note local scratch is typically
> unbacked/ephemeral — keep canonical history in your git remote.

---

## Developer flow (hot reload) — recommended

### Quick start with the helper script

`miscs/docker/vortex-dev.sh` wraps the whole lifecycle (it auto-detects the repo
root and defaults the build dir to `/scratch/$USER/vortex-build`):

```bash
miscs/docker/vortex-dev.sh build     # build the image (once)
miscs/docker/vortex-dev.sh up        # create/start the container
miscs/docker/vortex-dev.sh shell     # drop into a shell in the build dir
miscs/docker/vortex-dev.sh make      # build Vortex
miscs/docker/vortex-dev.sh exec './ci/blackbox.sh --cores=2 --app=vecadd'
miscs/docker/vortex-dev.sh stop      # stop  (keeps container + build dir)
miscs/docker/vortex-dev.sh restart   # restart
miscs/docker/vortex-dev.sh rm        # remove container (build dir persists)
miscs/docker/vortex-dev.sh status    # show image/container/paths
```

Override defaults with env vars, e.g. `XLEN=64 miscs/docker/vortex-dev.sh up` or
`VORTEX_BUILD=/scratch/$USER/vx64 XLEN=64 miscs/docker/vortex-dev.sh up`. To keep
several build dirs (e.g. 32- and 64-bit) under one mount, set `BUILD_NAME` to a
subdirectory of `$VORTEX_BUILD`: `BUILD_NAME=build64 XLEN=64 ... up`, and the
same `BUILD_NAME` on later `make`/`exec`/`shell` calls. The manual `docker`
commands below are the equivalent of what the script runs.

### Manual flow

The host repo is bind-mounted at `/vortex`; builds go to a separate
out-of-tree dir mounted at `/build_docker`. Edit on the host, recompile in the
container — no image rebuild or restart.

Two host paths feed the container:

- **`$VORTEX_SRC`** — your checkout. Keep this on your **backed-up home** (NFS on
  shared servers); it holds your editable, version-controlled code.
- **`$VORTEX_BUILD`** — the build directory, mounted at `/build_docker`. Put this
  on a **fast local disk** (e.g. `/scratch/$USER/...`).

> The build dir is mounted at the top-level `/build_docker`, not nested under
> `/vortex`: when the source is on root-squashed NFS, the Docker daemon can't
> create a mountpoint *inside* the NFS mount. The `-e BUILD_DIR=/build_docker`
> below tells the entrypoint to configure there; `configure` resolves the
> source root from `/vortex` regardless, so this is a normal out-of-tree build.

> **Why split them (shared-server etiquette).** A compile writes thousands of
> `.o`/`.a`/`.so` files. If the build dir lives inside an NFS checkout, every one
> of those writes crosses the network to the shared file server on every
> incremental build — that is the load that slows the server for everyone else.
> Source *reads* are cached by the NFS client after first touch, so keeping code
> on NFS is cheap; it is the build *writes* that must go local. With the split
> below, your steady-state edit→compile loop generates almost no shared-server
> traffic (build writes hit local scratch, and `ccache` is already a local
> volume). Local scratch is unbacked/ephemeral — that's fine, the build dir is
> fully regenerable; your code stays safe on NFS.
>
> Residual: Vortex compiles `third_party/softfloat` and `ramulator` *in-tree*, so
> those land under the NFS checkout — but only on the first build, not on the
> incremental loop.

### 1. Build the dev image (once)

```bash
miscs/docker/vortex-dev.sh build
```

The image installs the toolchain with this checkout's own installers, so it
always matches the checkout's `TOOLCHAIN_REV` (see [VERSION](../../VERSION)).
The script stages a small build context (`Dockerfile.dev`, `dev-entrypoint.sh`
and a `bootstrap/` copy of `VERSION`, `configure`, `VX_*.toml` and `ci/`); the
equivalent manual commands are:

```bash
ctx=$(mktemp -d) && mkdir -p $ctx/bootstrap
cp miscs/docker/Dockerfile.dev miscs/docker/dev-entrypoint.sh $ctx/
cp -p VERSION configure VX_*.toml $ctx/bootstrap/ && cp -rp ci $ctx/bootstrap/
docker build --platform=linux/amd64 -t vortex-dev -f $ctx/Dockerfile.dev $ctx
rm -rf $ctx
```

First build downloads the prebuilt toolchain and compiles SST from source
(several to tens of minutes). Rebuild whenever `TOOLCHAIN_REV` changes.

### 2. Start the container

Pick your paths and create the local build dir **once, as yourself** (so it is
owned by your UID — a bind mount keeps host ownership, and the container runs as
you):

```bash
export VORTEX_SRC="$HOME/path/to/vortex"      # your checkout (NFS home is fine)
export VORTEX_BUILD="/scratch/$USER/vortex-build"
mkdir -p "$VORTEX_BUILD"
```

Run as your own UID/GID so files written into the mounts are host-owned, not
root. The entrypoint configures the build dir on first start.

```bash
docker run -dit --name vortex-dev \
    --platform=linux/amd64 --privileged=true \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e XLEN=32 -e BUILD_DIR=/build_docker \
    -v "$VORTEX_SRC":/vortex \
    -v "$VORTEX_BUILD":/build_docker \
    -v vortex-ccache:/ccache \
    -w /vortex \
    vortex-dev
```

Set `-e XLEN=64` for a 64-bit build. (A harmless `groups: cannot find name for
group ID ...` line may print because your host UID has no entry in the
container's `/etc/passwd`.) If you omit the `$VORTEX_BUILD` mount and
`BUILD_DIR`, the build dir falls back to inside the NFS checkout — it works, but
compile writes then load the shared file server.

### 3. Build and test inside the container

```bash
docker exec -it vortex-dev bash
# inside the container:
cd /build_docker
make -s
./ci/blackbox.sh --cores=2 --app=vecadd
```

### 4. Hot reload

Edit any source file in `$VORTEX_SRC` on the host (SimX, RTL, runtime,
tests...). The container sees it immediately at `/vortex`. Just recompile —
`make -s`, or `./ci/blackbox.sh ...` which rebuilds the driver. No restart.

### 5. Stop / start / remove

```bash
docker stop vortex-dev          # stop (keeps the container + build dir)
docker start vortex-dev         # resume later (fast)
docker exec -it vortex-dev bash # re-enter
docker rm -f vortex-dev         # remove the container ($VORTEX_BUILD on local
                                # disk and the ccache volume both persist)
```

The build dir lives on `$VORTEX_BUILD` (local scratch), so it survives
`docker rm`; the next container reuses it. Delete `$VORTEX_BUILD/*` to
reconfigure from scratch.

---

## Packaged flow (self-contained)

`Dockerfile.prod` bakes a checkout + build into the image for a mount-free
environment:

```bash
cd miscs/docker
docker build --platform=linux/amd64 -t vortex-packaged -f Dockerfile.prod .
docker run -it --name vortex --privileged=true --platform=linux/amd64 vortex-packaged
# you land in /vortex/build:
make -s && ./ci/blackbox.sh --cores=2 --app=vecadd
```

Stop/start/remove are the same as above (`docker stop|start|rm vortex`,
re-enter with `docker exec -it vortex bash`). Because the source is baked in,
host edits are NOT reflected — use the dev flow for that.

See `docs/` to learn more about building and testing Vortex.
