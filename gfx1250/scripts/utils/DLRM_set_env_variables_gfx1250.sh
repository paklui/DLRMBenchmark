# Set Global Env Vars for DLRM on Helios-P Miami/Cairo (gfx1250, AMD Pensando AINIC)
# Adapted from DLRM_set_env_variables.sh (MI350 GT cluster, 8 GPU / 8 Broadcom bnxt_re NICs).
# Miami/Cairo are 4 GPU / 4 AINIC ionic NICs each
# for the full adaptation rationale.
#
# UPDATED 2026-09-23: aligned flag-for-flag with ~/param-bench/env.sh (the canonical PARAM
# comms.py config), after confirming that config gets real 2-node RDMA over ionic_0-3 working
# cleanly in-docker (rccl-tests all_reduce_perf, 714 GB/s busbw @ 1GB, zero errors) using
# run_2n_param_flags_docker.sh on this exact rack. The RCCL library was already correct here
# (rccl-develop, not -containerbkc) -- the gap was purely in these flags (previously missing
# IONIC_LOCKFREE, RCCL_IB_ABORT_ON_ERROR, RCCL_CTS_*, NCCL_MNNVL_ENABLE, NCCL_RINGS, etc, plus
# NCCL_IB_QPS_PER_CONNECTION=2 instead of 1 and NCCL_GDR_FLUSH_DISABLE=1 instead of 0). This is
# the prime suspect for the earlier "rail-mismatch" IBV_WC_REM_OP_ERR seen in 2-node RDMA DLRM
# runs -- retest with this file before concluding the rail mismatch is a real HW/topology issue.

# Working gfx1250 RCCL build (validated in PARAM/Primus benchmark work on this rack) --
# NOT the stale /apps/build/... path from the old GT cluster, which doesn't exist here.
export LD_LIBRARY_PATH=/root/rccl-develop/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/root/rccl-develop/lib/librccl.so.1.0

export HIP_VISIBLE_DEVICES=0,1,2,3
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3

# Network and transport -- both control-plane socket vars set (param-bench/env.sh sets both)
export NCCL_SOCKET_IFNAME=ens1f0np0
export NCCL_SOCKET_IFNAME2=ens1f0np0
export NCCL_OOB_NET_IFNAME="ens1f0np0"

export NCCL_GDR_FLUSH_DISABLE=0
export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_GDRCOPY_ENABLE=0
export NCCL_IB_QPS_PER_CONNECTION=1
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DMABUF_ENABLE=1
export NCCL_DEBUG=INFO

export NCCL_IB_USE_INLINE=1
export RCCL_IB_ABORT_ON_ERROR=1
export IONIC_LOCKFREE=all
export NCCL_TOPO_DUMP_FILE=/tmp/system.txt
export NCCL_IB_GID_INDEX=1
export RCCL_CTS_OFFLOAD_ENABLED=0
export RCCL_CTS_INLINE_DATA=0
export NCCL_NET_OPTIONAL_RECV_COMPLETION=1
export NCCL_PXN_DISABLE=1
export RCCL_AINIC_ROCE=1
export NCCL_GIN_ENABLE=0
export NCCL_IB_TC=128
export NCCL_IB_FIFO_TC=192
export RCCL_MODEL_MATCHING_DISABLE=1

export HSA_HOTSWAP_DISABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_MNNVL_ENABLE=1        # route cross-node traffic that CAN go over UALoE/IFoE scale-up fabric
export NCCL_RINGS='N1 1 0 2 3 N3 | N3 3 0 1 2 N2 | N2 2 3 1 0 N0 | N0 0 2 3 1 N1 | '

# DLRM-specific extras (not in param-bench/env.sh, kept from the original DLRM adaptation --
# not known to conflict, left in place; revisit if results differ from the pure rccl-tests case)
export NCCL_NCHANNELS_PER_NET_PEER=1
export HSA_ENABLE_IPC_MODE_LEGACY=1

# Halved from the original 64 (tuned for 8 NICs) proportionally for 4 NICs -- not empirically
# retuned yet, revisit if throughput looks off. param-bench/env.sh leaves these unset (RCCL default).
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=32
