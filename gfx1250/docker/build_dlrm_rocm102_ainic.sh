#!/usr/bin/env bash
# build_dlrm_rocm102_ainic.sh — ROCm 10.2 DLRM image for 2-node RDMA: dlrmv2-${USER}:dlrm-gfx1250-rocm102-fbgemm-d651cb36-a2ashim
# (FBGEMM main d651cb36, torchrec 1.9.0, RCCL 2.31.2, LD_PRELOAD ncclAllToAll(v) shim) + the AINIC userspace (rdma-core with the ionic provider,
# perftest) from ~/param-bench/ainic-build/host_sw_pkg — the same bundle as dlrmv2-${USER}:ainic-v1 (libionic 26.08.7.001 = host libionic-amd 61.0.26.08.7.001).
# Result: image dlrmv2-${USER}:dlrm-gfx1250-rocm102-ainic, container dlrm-gfx1250-rocm102-ainic-run. Run ON MIAMI.
set -eu
BASE=dlrmv2-${USER}:dlrm-gfx1250-rocm102-fbgemm-d651cb36-a2ashim; IMG=dlrmv2-${USER}:dlrm-gfx1250-rocm102-ainic; B=dlrm-rocm102-ainic-build; RUN=dlrm-gfx1250-rocm102-ainic-run
H=$HOME
OPTS="--network host --ipc host --privileged --cap-add SYS_PTRACE --security-opt seccomp=unconfined --group-add video
  --ulimit core=-1 --ulimit memlock=-1 --ulimit stack=-1 --device /dev/kfd --device /dev/dri --device /dev/infiniband
  -v $H/dlrm_dataset:/root/dlrm_dataset -v $H/dlrm_libs:/root/dlrm_libs -v $H/DLRM_optimization:/root/DLRM_optimization -v $H/rccl-develop:/root/rccl-develop"
sudo -n docker rm -f $B >/dev/null 2>&1 || true
sudo -n docker run -d --name $B --network host --device /dev/infiniband --entrypoint sleep $BASE infinity >/dev/null
sudo -n docker cp $H/param-bench/ainic-build/host_sw_pkg $B:/tmp/ainic_bundle
sudo -n docker exec $B bash -c '
set -eux
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends jq > /tmp/jq_install.log 2>&1
cd /tmp/ainic_bundle && bash install.sh -m rdma_core,perftest -y --ignore-compat-fail > /tmp/ainic_install.log 2>&1 || { tail -30 /tmp/ainic_install.log; exit 1; }
tail -5 /tmp/ainic_install.log; cd / && rm -rf /tmp/ainic_bundle /var/lib/apt/lists/*
dpkg -l | grep -i -E "ibverbs|ionic|perftest" | awk "{print \$2, \$3}"
ibv_devinfo -l 2>&1 | head -6
python3 -c "import torch, torchrec, fbgemm_gpu; print(\"imports ok\", torch.__version__)"
'
sudo -n docker commit -m "AINIC userspace (rdma-core + ionic provider + perftest) from param-bench/ainic-build/host_sw_pkg" $B $IMG
sudo -n docker rm -f $B >/dev/null
sudo -n docker rm -f $RUN >/dev/null 2>&1 || true
sudo -n docker create --name $RUN $OPTS --entrypoint sleep $IMG infinity >/dev/null
echo "BUILD DONE: $IMG, container $RUN created on Miami"
