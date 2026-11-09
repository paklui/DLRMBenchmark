#!/usr/bin/env bash
# build_dlrm_rocm102_gfx1250dev.sh — FBGEMM ROCm/FBGEMM aryaman/gfx1250-dev at fa0bb1a78 (the commit in dlrmv2-${USER}:dlrm-gfx1250-fbgemm-gfx1250dev)
# + the cub::CTA_SYNC -> __syncthreads patch, built for gfx1250 on the ROCm 10.2 DLRM image dlrmv2-${USER}:dlrm-gfx1250-rocm102-fbgemm-d651cb36.
# Built WITHOUT -DTORCH_USE_HIP_DSA (like the old branch build; the d651cb36 build in the base image has it). Source in /root/FBGEMM-gfx1250dev,
# the base image's /root/FBGEMM (main d651cb36) is left in place but its develop install is replaced.
# Result: image dlrmv2-${USER}:dlrm-gfx1250-rocm102-fbgemm-gfx1250dev, container dlrm-gfx1250-rocm102-gfx1250dev-run.
# Run ON MIAMI:  (setsid nohup bash build_dlrm_rocm102_gfx1250dev.sh > build_dlrm_rocm102_gfx1250dev.out 2>&1 < /dev/null &)
set -eu
BASE=dlrmv2-${USER}:dlrm-gfx1250-rocm102-fbgemm-d651cb36; IMG=dlrmv2-${USER}:dlrm-gfx1250-rocm102-fbgemm-gfx1250dev
B=dlrm-gfx1250-rocm102-gfx1250dev-build; RUN=dlrm-gfx1250-rocm102-gfx1250dev-run; H=$HOME
OPTS="--network host --ipc host --privileged --cap-add SYS_PTRACE --security-opt seccomp=unconfined --group-add video
  --ulimit core=-1 --ulimit memlock=-1 --ulimit stack=-1 --device /dev/kfd --device /dev/dri --device /dev/infiniband
  -v $H/dlrm_dataset:/root/dlrm_dataset -v $H/dlrm_libs:/root/dlrm_libs -v $H/DLRM_optimization:/root/DLRM_optimization -v $H/rccl-develop:/root/rccl-develop"
sudo -n docker rm -f $B >/dev/null 2>&1 || true
sudo -n docker run -d --name $B $OPTS --entrypoint sleep $BASE infinity
sudo -n docker exec $B bash -c '
set -eux
cd /root && git clone https://github.com/ROCm/FBGEMM.git FBGEMM-gfx1250dev && cd FBGEMM-gfx1250dev
git checkout -b dlrm-rocm102 fa0bb1a785015ac522e146b74ee2bc4884a564d6 && git submodule update --init --recursive
f=fbgemm_gpu/src/sparse_ops/sparse_async_batched_cumsum.cu; test $(grep -c "cub::CTA_SYNC();" $f) = 2
sed -i "s/cub::CTA_SYNC();/__syncthreads();/" $f
git -c user.name=dlrm -c user.email=dlrm@localhost commit -q -am "patch: cub::CTA_SYNC -> __syncthreads (removed from current hipcub)"
git log --oneline -2
pip uninstall --break-system-packages -y fbgemm_gpu_nightly-rocm || true
export ROCM_ROOT=$(rocm-sdk path --root); export ROCM_PATH=$ROCM_ROOT HIP_PATH=$ROCM_ROOT
export CMAKE_PREFIX_PATH=$ROCM_ROOT/lib/cmake:$ROCM_ROOT:${CMAKE_PREFIX_PATH:-}
export ROCM_VERSION=100200 HIPBLAS_V2=1 BUILD_ROCM_VERSION=10.2 MAX_JOBS=256 PYTORCH_ROCM_ARCH=gfx1250
cd fbgemm_gpu
echo "build start $(date -u +%FT%TZ)"
python setup.py --build-variant rocm --build-target default -DHIP_ROOT_DIR=$ROCM_ROOT build develop > /root/fbgemm_build_gfx1250dev_rocm102.log 2>&1 || { tail -40 /root/fbgemm_build_gfx1250dev_rocm102.log; exit 1; }
echo "build end $(date -u +%FT%TZ)"
cd /root && python3 -c "import fbgemm_gpu, os, torch, torchrec; print(\"fbgemm_gpu from\", os.path.dirname(fbgemm_gpu.__file__)); import fbgemm_gpu.split_table_batched_embeddings_ops_training; print(\"imports ok\", torch.__version__)"
pip freeze | grep -i fbgemm
'
sudo -n docker commit -m "DLRM ROCm 10.2: FBGEMM aryaman/gfx1250-dev fa0bb1a78 + CTA_SYNC patch, no HIP DSA" $B $IMG
sudo -n docker stop $B >/dev/null
sudo -n docker rm -f $RUN >/dev/null 2>&1 || true
sudo -n docker create --name $RUN $OPTS --entrypoint sleep $IMG infinity >/dev/null
echo "BUILD DONE: $IMG, container $RUN created"
