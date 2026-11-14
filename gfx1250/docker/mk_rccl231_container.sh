#!/usr/bin/env bash
# mk_rccl231_container.sh — ON EACH NODE: test container dlrm-gfx1250-rocm103-rccl231-run = the new image (dlrmv2-${USER}:dlrm-gfx1250-rocm103-ainic) with RCCL 2.31.2 (librccl.so.1 + its gfx1250 device code kpack,
# extracted from the old image into ~/DLRM_optimization/rccl231) bind-mounted read-only over the image's RCCL. The image itself is untouched; remove the container to undo.
set -u; H=$HOME; CT=dlrm-gfx1250-rocm103-rccl231-run; P=/usr/local/lib/python3.12/dist-packages; R=$H/DLRM_optimization/rccl231
sudo -n docker rm -f $CT >/dev/null 2>&1
sudo -n docker create --name $CT --network host --ipc host --privileged --cap-add SYS_PTRACE --security-opt seccomp=unconfined --group-add video \
  --ulimit core=-1 --ulimit memlock=-1 --ulimit stack=-1 --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -v $H/dlrm_dataset:/root/dlrm_dataset -v $H/dlrm_libs:/root/dlrm_libs -v $H/DLRM_optimization:/root/DLRM_optimization -v $H/rccl-develop:/root/rccl-develop \
  -v $R/lib/librccl.so.1:$P/_rocm_sdk_libraries/lib/librccl.so.1:ro -v $R/lib/librccl.so.1:$P/_rocm_sdk_devel/lib/librccl.so.1:ro \
  -v $R/.kpack/rccl_lib_gfx1250.kpack:$P/_rocm_sdk_libraries/.kpack/rccl_lib_gfx1250.kpack:ro -v $R/.kpack/rccl_lib_gfx1250.kpack:$P/_rocm_sdk_devel/.kpack/rccl_lib_gfx1250.kpack:ro \
  --entrypoint sleep dlrmv2-${USER}:dlrm-gfx1250-rocm103-ainic infinity >/dev/null && echo "$(hostname -s): container $CT created"
