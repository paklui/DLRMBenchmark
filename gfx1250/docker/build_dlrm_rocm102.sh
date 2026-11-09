#!/usr/bin/env bash
# build_dlrm_rocm102.sh — DLRM image on ROCm 10.2 BKC: base $BASE_IMAGE (ROCm 10.2 PyTorch image with FBGEMM main d651cb36 built for gfx1250)
# (= a ROCm 10.2 BKC PyTorch release/2.11 image + FBGEMM main d651cb36 built for gfx1250, 2026-10-14)
# + torchrec 1.9.0 (same as dlrmv2-${USER}:dlrm-gfx1250*) and its runtime deps. torch/numpy pinned so pip cannot replace the ROCm stack.
# Result: image dlrmv2-${USER}:dlrm-gfx1250-rocm102-fbgemm-d651cb36, container dlrm-gfx1250-rocm102-run (same mounts as dlrm-gfx1250-gfx1250dev-run).
# Run ON MIAMI:  (setsid nohup bash build_dlrm_rocm102.sh > build_dlrm_rocm102.out 2>&1 < /dev/null &)
set -eu
BASE=${BASE_IMAGE:?set BASE_IMAGE to a ROCm 10.2 PyTorch image with FBGEMM main d651cb36 built for gfx1250}; IMG=dlrmv2-${USER}:dlrm-gfx1250-rocm102-fbgemm-d651cb36; B=dlrm-gfx1250-rocm102-build; RUN=dlrm-gfx1250-rocm102-run
H=$HOME
sudo -n docker rm -f $B >/dev/null 2>&1 || true
sudo -n docker run -d --name $B --network host --ipc host --privileged \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --group-add video \
  --ulimit core=-1 --ulimit memlock=-1 --ulimit stack=-1 \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -v $H/dlrm_dataset:/root/dlrm_dataset -v $H/dlrm_libs:/root/dlrm_libs \
  -v $H/DLRM_optimization:/root/DLRM_optimization -v $H/rccl-develop:/root/rccl-develop \
  --entrypoint sleep $BASE infinity
sudo -n docker exec $B bash -c '
set -eux
PIP="pip install --break-system-packages"
pip freeze | grep -i -E "^(torch|torchvision|torchaudio|triton|numpy|fbgemm)" > /tmp/constraints.txt; cat /tmp/constraints.txt
# stale scipy extension from the base image build shadows scipy.sparse.linalg._propack (FBGEMM bench page, gis suite)
f=$(python3 -c "import scipy,os;print(os.path.dirname(scipy.__file__))")/sparse/linalg/_propack.cpython-312-x86_64-linux-gnu.so; [ -f "$f" ] && mv "$f" "$f.stale" || true
$PIP -c /tmp/constraints.txt --no-deps torchrec==1.9.0
$PIP -c /tmp/constraints.txt tensordict "torchmetrics==1.0.3" iopath "git+https://github.com/mlperf/logging.git"
pip freeze | grep -i -E "^(torch|fbgemm|tensordict|torchmetrics|mlperf|iopath|numpy)"
python3 -c "import torch, fbgemm_gpu, torchrec, torchmetrics, mlperf_logging, iopath; from torchrec.distributed.model_parallel import DistributedModelParallel; print(\"imports ok\", torch.__version__, torchrec.__version__, torch.cuda.device_count(), torch.cuda.nccl.version())"
'
sudo -n docker commit -m "DLRM: torchrec 1.9.0 + deps on fbgemm-bench-bkc1002-d651cb36 (ROCm 10.2 BKC 20261002, FBGEMM main d651cb36)" $B $IMG
sudo -n docker stop $B >/dev/null
sudo -n docker rm -f $RUN >/dev/null 2>&1 || true
sudo -n docker create --name $RUN --network host --ipc host --privileged \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --group-add video \
  --ulimit core=-1 --ulimit memlock=-1 --ulimit stack=-1 \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -v $H/dlrm_dataset:/root/dlrm_dataset -v $H/dlrm_libs:/root/dlrm_libs \
  -v $H/DLRM_optimization:/root/DLRM_optimization -v $H/rccl-develop:/root/rccl-develop \
  --entrypoint sleep $IMG infinity >/dev/null
echo "BUILD DONE: $IMG, container $RUN created"
