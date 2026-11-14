#!/usr/bin/env bash
# build_dlrm_rocm103.sh — run on one node, detached: DLRM image on the newer base
#   BASE = a PyTorch image with ROCm 10.3 (nightly 10.3.0a20261111, torch 2.13.0+rocm10.3, rocprofiler-sdk 1.5.1 = PyTorch-profiler GPU events work, RCCL 2.32.3); set BASE=<image> in the environment
# Same steps as docker/build_dlrm_rocm102*.sh, adapted to ROCm 10.3 (FBGEMM build recipe):
#   B1 FBGEMM main d651cb36 for gfx1250 (ROCm 10.3: ROCM_VERSION=100300, BUILD_ROCM_VERSION=10.3) + scipy _propack fix
#   B3 TorchRec 1.9.0 + deps pinned against the torch stack   B4 all-to-all shim   B5 AINIC user space (rdma-core + ionic provider, perftest)
#   B6 commit as dlrmv2-${USER}:dlrm-gfx1250-rocm103-ainic (LD_PRELOAD shim), create container dlrm-gfx1250-rocm103-ainic-run
# CPU/network only (no GPU kernels). The working 10.2 image/container are not touched. Log ~/dlrm_build_rocm103_20261112.out; end: BUILD_DONE / BUILD_FAILED
set -u
H=$HOME; BASE=${BASE:?set BASE to the PyTorch ROCm 10.3 base image}; IMG=dlrmv2-${USER}:dlrm-gfx1250-rocm103-ainic; B=dlrm103-build; RUN=dlrm-gfx1250-rocm103-ainic-run
log() { echo "[$(date '+%F %T %Z')] $*"; }
fail() { log "BUILD_FAILED: $*"; exit 1; }
DK="sudo -n docker"
$DK rm -f $B >/dev/null 2>&1
log "base $($DK image inspect -f '{{.Id}}' $BASE | cut -c1-19)"
$DK run -d --name $B --network host --ipc host --privileged --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -e HSA_ENABLE_SDMA=0 --entrypoint sleep $BASE infinity >/dev/null || fail "docker run"
$DK cp $H/param-bench/ainic-build/host_sw_pkg $B:/tmp/ainic_bundle || fail "cp ainic bundle"
$DK cp $H/DLRM_optimization/utils/nccl_alltoall_shim.c $B:/tmp/ || fail "cp shim"
$DK cp $H/patch_fbgemm_rocmsetup.py $B:/tmp/ || fail "cp rocmsetup patch"
log "=== B1: FBGEMM d651cb36 for gfx1250 on ROCm 10.3"
$DK exec $B bash -c '
set -ex
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libtbb-dev git jq > /tmp/apt.log 2>&1
cd /root && git clone -q https://github.com/pytorch/FBGEMM.git && cd FBGEMM && git checkout -q d651cb360528be4755d225e5b3b0bb119bc05af7 && git submodule update --init --recursive -q
python3 /tmp/patch_fbgemm_rocmsetup.py /root/FBGEMM/cmake/modules/RocmSetup.cmake
pip freeze | grep -i -E "^(torch|torchvision|torchaudio|triton|numpy|cmake|ninja|scikit-build)" > /tmp/constraints.txt
pip install -q --break-system-packages -c /tmp/constraints.txt -r fbgemm_gpu/requirements.txt
export ROCM_ROOT=$(rocm-sdk path --root); export ROCM_PATH=$ROCM_ROOT HIP_PATH=$ROCM_ROOT
export CMAKE_PREFIX_PATH=$ROCM_ROOT/lib/cmake:$ROCM_ROOT:${CMAKE_PREFIX_PATH:-}
export ROCM_VERSION=100300 HIPBLAS_V2=1 BUILD_ROCM_VERSION=10.3 MAX_JOBS=256 PYTORCH_ROCM_ARCH=gfx1250 FBGEMM_REPO_ROOT=/root/FBGEMM
# ROCm 10.3 torch exports clang-only compile flags (-Wthread-safety, --offload-arch, ...): build host code with the SDK clang, not g++
export CC=$(command -v amdclang || echo $ROCM_ROOT/lib/llvm/bin/clang) CXX=$(command -v amdclang++ || echo $ROCM_ROOT/lib/llvm/bin/clang++)
echo "CC=$CC CXX=$CXX"; $CXX --version | head -1
cd fbgemm_gpu && git clean -xdf -q
echo "build start $(date -u +%FT%TZ)"
python setup.py --build-variant rocm --build-target default --package_channel nightly -DHIP_ROOT_DIR=$ROCM_ROOT -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA" build develop > /root/fbgemm_build.log 2>&1 || { tail -40 /root/fbgemm_build.log; exit 1; }
echo "build end $(date -u +%FT%TZ)"
f=$(python3 -c "import scipy,os;print(os.path.dirname(scipy.__file__))")/sparse/linalg/_propack.cpython-312-x86_64-linux-gnu.so; [ -f "$f" ] && mv "$f" "$f.stale" || true
cd /root && python3 -c "import torch, fbgemm_gpu, scipy.sparse.linalg; import fbgemm_gpu.split_table_batched_embeddings_ops_training; print(\"B1 imports ok\", torch.__version__)"
' > $HOME/dlrm_build_B1.log 2>&1 || { tail -45 $HOME/dlrm_build_B1.log; fail "B1 FBGEMM build"; }
tail -3 $HOME/dlrm_build_B1.log; grep -E "build (start|end)" $HOME/dlrm_build_B1.log
log "=== B3/B4/B5: TorchRec + deps, shim, AINIC user space"
$DK exec $B bash -c '
set -ex
cd /root
pip install -q --break-system-packages -c /tmp/constraints.txt --no-deps torchrec==1.9.0
pip install -q --break-system-packages -c /tmp/constraints.txt "tensordict==0.15.3" "torchmetrics==1.0.3" iopath "git+https://github.com/mlperf/logging.git"
mkdir -p /opt/rccl-shim && mv /tmp/nccl_alltoall_shim.c /opt/rccl-shim/ && cd /opt/rccl-shim
R=$(dirname $(find /usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries -name "librccl.so.1" | head -1))
gcc -shared -fPIC -O2 -o libnccl_alltoall_shim.so nccl_alltoall_shim.c -L$R -l:librccl.so.1 -Wl,-rpath,$R
cd /tmp/ainic_bundle && bash install.sh -m rdma_core,perftest -y --ignore-compat-fail > /tmp/ainic_install.log 2>&1 || { tail -30 /tmp/ainic_install.log; exit 1; }
tail -2 /tmp/ainic_install.log; cd / && rm -rf /tmp/ainic_bundle /var/lib/apt/lists/*
python3 -c "import torch, fbgemm_gpu, torchrec, torchmetrics, mlperf_logging, iopath; print(\"imports ok\", torch.__version__, torchrec.__version__, torch.cuda.nccl.version())"
dpkg -l | grep -E "libionic1|perftest" | awk "{print \$2, \$3}"
' > $HOME/dlrm_build_B3.log 2>&1 || { tail -45 $HOME/dlrm_build_B3.log; fail "B3-B5"; }
tail -6 $HOME/dlrm_build_B3.log
log "=== B6: commit image + create run container"
$DK commit --change 'ENV LD_PRELOAD=/opt/rccl-shim/libnccl_alltoall_shim.so' -m "DLRM on a ROCm 10.3 PyTorch base (10.3.0a20261111): FBGEMM d651cb36, torchrec 1.9.0, shim, AINIC userspace" $B $IMG >/dev/null || fail "commit"
$DK rm -f $B >/dev/null
$DK rm -f $RUN >/dev/null 2>&1
$DK create --name $RUN --network host --ipc host --privileged --cap-add SYS_PTRACE --security-opt seccomp=unconfined --group-add video \
  --ulimit core=-1 --ulimit memlock=-1 --ulimit stack=-1 --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -v $H/dlrm_dataset:/root/dlrm_dataset -v $H/dlrm_libs:/root/dlrm_libs -v $H/DLRM_optimization:/root/DLRM_optimization -v $H/rccl-develop:/root/rccl-develop \
  --entrypoint sleep $IMG infinity >/dev/null || fail "docker create"
log "image $($DK image inspect -f '{{.Id}} {{.Size}}' $IMG | cut -c1-40)"
log "BUILD_DONE: $IMG, container $RUN"
