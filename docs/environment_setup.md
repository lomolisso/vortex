# Environment Setup

These instructions apply to the development vortex repo using the updated toolchain. The updated toolchain is considered to be any commit of `master` pulled from July 2, 2023 onwards. The toolchain update in question can be viewed in this [commit](https://github.com/vortexgpgpu/vortex-dev/commit/0048496ba28d7b9a209a0e569d52d60f2b68fc04). Therefore, if you are unsure whether you are using the new toolchain or not, then you should check the `ci` folder for the existence of the `toolchain_prebuilt.sh` script. Furthermore, you should notice that the `toolchain_install.sh` script has the legacy `llvm()` split into `llvm-vortex()` and `llvm-pocl()`.

## Set Up on Your Own System

The toolchain binaries provided with Vortex are built on Ubuntu-based systems. To install Vortex on your own system, [follow these instructions](install_vortex.md).

## Servers for Georgia Tech Students and Collaborators

### Volvo

Volvo is a 64-core server provided by HPArch. You need valid credentials to access it. If you don't already have access, you can get in contact with your mentor to ask about setting your account up.

Setup on Volvo:

1. Connect to Georgia Tech's VPN or ssh into another machine on campus
2. `ssh volvo.cc.gatech.edu`
3. Clone Vortex to your home directory: `git clone --recursive https://github.com/vortexgpgpu/vortex.git`
4. `source /nethome/software/set_vortex_env.sh` to set up the necessary environment variables.
5. `make -s` in the `vortex` root directory
6. Run a test program: `./ci/blackbox.sh --cores=2 --app=dogfood`

### Nio

Nio is a 20-core desktop server provided by HPArch. If you have access to Volvo, you also have access to Nio.

Setup on Nio:

1. Connect to Georgia Tech's VPN or ssh into another machine on campus
2. `ssh nio.cc.gatech.edu`
3. Clone Vortex to your home directory: `git clone --recursive https://github.com/vortexgpgpu/vortex.git`
4. `source /opt/set_vortex_env_dev.sh` to set up the necessary environment variables.
5. `make -s` in the `vortex` root directory
6. Run a test program: `./ci/blackbox.sh --cores=2 --app=dogfood`

## Docker

Docker provides isolated, pre-built environments that can be created, shared,
and used across machines. ARM-based hosts (e.g. Apple Silicon) run the
`linux/amd64` image under emulation, which is slower but functional. The
Docker assets live in [../miscs/docker](../miscs/docker) and install the full
non-ASIC toolchain (SimX, rtlsim/verilator, RISC-V LLVM+GCC, POCL/OpenCL,
chipstar/HIP, mesa/Vulkan, gem5, SST). ASIC EDA tools (Synopsys/Cadence) are
not included — run those flows on the host.

### Developer flow (hot reload) — recommended

`miscs/docker/Dockerfile.dev` is a source-less image: it carries the toolchain
in `/opt`, and your host checkout is bind-mounted at `/vortex` so edits on the
host are seen live in the container. The build dir is mounted over
`/vortex/build_docker` (git-ignored), separate from any host-side `build/`.

Keep **source on your backed-up home** and the **build dir on fast local disk**
(e.g. `/scratch`): compile writes are heavy and, if placed on an NFS checkout,
load the shared file server on every incremental build. Source reads are cached
locally after first touch, so code-on-NFS stays cheap.

The easiest path is the helper script `miscs/docker/vortex-dev.sh`
(`build` / `up` / `shell` / `make` / `stop` / `restart` / `rm` / `status`); the
equivalent manual commands are:

```bash
export VORTEX_SRC="$HOME/path/to/vortex"      # checkout (NFS home is fine)
export VORTEX_BUILD="/scratch/$USER/vortex-build"
mkdir -p "$VORTEX_BUILD"                       # create once, as yourself
docker build --platform=linux/amd64 -t vortex-dev -f miscs/docker/Dockerfile.dev miscs/docker
docker run -dit --name vortex-dev --platform=linux/amd64 --privileged=true \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e XLEN=32 -e BUILD_DIR=/build_docker \
    -v "$VORTEX_SRC":/vortex -v "$VORTEX_BUILD":/build_docker \
    -v vortex-ccache:/ccache -w /vortex vortex-dev
docker exec -it vortex-dev bash
# inside the container:
cd /build_docker && make -s && ./ci/blackbox.sh --cores=2 --app=vecadd
```

The build dir is mounted at the top-level `/build_docker` (not nested under
`/vortex`) because the Docker daemon can't create a mountpoint inside a
root-squashed NFS mount; `-e BUILD_DIR` points the entrypoint there.

Edit a source file on the host, rerun `make -s` in the container — no rebuild
or restart. Omitting the `$VORTEX_BUILD` mount and `BUILD_DIR` falls back to a
build dir inside the NFS checkout (works, but loads the shared server).

> The Docker daemon (root) must be able to traverse to `$VORTEX_SRC`. A home on
> **NFS with root-squash** maps the daemon's root to `nobody`, which cannot
> traverse a `0700` home, so bind mounts of `~/...` fail with `permission
> denied`. Fix it in place (no copying) by granting traverse-only access once:
> `chmod 711 ~` (adds `o+x`, not `o+r`). The container runs as your UID, so NFS
> reads/writes happen as you. See [../miscs/docker/README.md](../miscs/docker/README.md).

### Packaged flow (self-contained)

`miscs/docker/Dockerfile.prod` bakes a checkout + build into the image for a
mount-free environment:

```bash
cd miscs/docker
docker build --platform=linux/amd64 -t vortex-packaged -f Dockerfile.prod .
docker run -it --name vortex --privileged=true --platform=linux/amd64 vortex-packaged
# you land in /vortex/build:
make -s && ./ci/blackbox.sh --cores=2 --app=vecadd
```

See [../miscs/docker/README.md](../miscs/docker/README.md) for the full
reference (stopping/restarting, `XLEN=64`, etc.).
