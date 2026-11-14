#!/usr/bin/env bash
# extract_rccl_2312.sh — copy RCCL 2.31.2 (librccl.so.1 + its gfx1250 device code, the .kpack file) out of the ROCm 10.2 DLRM image into ~/DLRM_optimization/rccl231,
# for docker/mk_rccl231_container.sh. Why: the ROCm 10.3 image bundles RCCL 2.32.3, whose first collective hangs on 2 nodes (see README). Usage: bash extract_rccl_2312.sh [OLD_IMAGE]
set -eu
OLD=${1:-dlrmv2-${USER}:dlrm-gfx1250-rocm102-ainic}; P=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries; R=$HOME/DLRM_optimization/rccl231
mkdir -p $R/lib $R/.kpack
C=$(sudo docker create $OLD)
sudo docker cp $C:$P/lib/librccl.so.1 $R/lib/librccl.so.1
sudo docker cp $C:$P/.kpack/rccl_lib_gfx1250.kpack $R/.kpack/rccl_lib_gfx1250.kpack
sudo docker rm $C >/dev/null
sudo chown -R "$(id -u):$(id -g)" $R
strings $R/lib/librccl.so.1 | grep -m1 -E "RCCL version|NCCL version" || true
ls -l $R/lib $R/.kpack
