# DLRM v2 on gfx1250 (Helios-P, 1 and 2 nodes) — scripts, image recipe and results

Backup of the setup used to train DLRM v2 (TorchRec, MLPerf recommendation_v2 / Criteo multi-hot) on two Helios-P EVT2 compute trays,
each with 4x gfx1250 GPUs and 4x AMD Pensando Vulcano AINIC (800G, RoCE). GPUs run in DPX/NPS2 mode, so each node exposes 8 logical
GPUs (ranks). The scripts were written against the `DLRM_optimization` layout (launch scripts at the top level, model code in
`training/recommendation_v2/torchrec_dlrm/`); they are kept here as-is, together with the local changes as patches.

## Contents

| Path | What |
| --- | --- |
| `scripts/dlrm_1n8.sh` | 1 node, 8 ranks. Generates the env/launch variants from the files below and runs one job in the container (watchdog for teardown hangs, hard timeout). |
| `scripts/dlrm_2n.sh` | 2 nodes, `NPN` ranks per node (8 = all logical GPUs). Run on node rank 0; starts the second node over ssh. |
| `scripts/dlrm_2n_verify.sh` | 2-node run that samples the NIC RDMA counters every 5 s (back-end traffic check). |
| `scripts/utils/DLRM_set_env_variables_gfx1250.sh` | Validated 2-node RDMA environment (RCCL / AINIC settings). |
| `scripts/DLRM_local_launch_gfx1250.sh`, `scripts/DLRM_config_node_gfx1250.sh` | Per-node launch script (torchrun) and node config used by the generators. |
| `docker/build_dlrm_rocm102.sh` → `build_dlrm_rocm102_gfx1250dev.sh` → `build_dlrm_rocm102_ainic.sh` | Image chain: ROCm 10.2 PyTorch base (`BASE_IMAGE`) + FBGEMM (main d651cb36) + TorchRec 1.9.0, then the AINIC userspace (rdma-core with the ionic provider, perftest). Images are named `dlrmv2-${USER}:<tag>`; final image `dlrmv2-${USER}:dlrm-gfx1250-rocm102-ainic`. The `…-a2ashim` input of the last step is the `d651cb36` image with `nccl_alltoall_shim.c` compiled into `/usr/local/lib` and added to `LD_PRELOAD` (a manual step, not scripted). |
| `docker/nccl_alltoall_shim.c` | LD_PRELOAD shim that maps PyTorch's deprecated `ncclAllToAll(v)` calls to the current RCCL API (silences a per-iteration deprecation warning). |
| `docker/backup_image_20261029.sh` | Saves the image as `.tar.zst` with a checksum. |
| `patches/dlrm_main_destroy_process_group.diff` | Calls `dist.destroy_process_group()` at the end of `dlrm_main.py`, so the process group is torn down while the runtime is alive (avoids a `double free or corruption (!prev)` abort at exit). |
| `patches/DLRM_optimization_local_changes.diff` | Local changes to the generic launch/setup scripts; base commit in `patches/BASE.txt`. |

## Running

1 node (on the node; recommended batch 16,384 per rank):

```bash
cd ~/DLRM_optimization
CT=dlrm-gfx1250-rocm102-ainic-run RCCL=bundled CUMEM=1 ITERS=600 BATCH=16384 \
  EXTRA="PYTORCH_TUNABLEOP_ENABLED=0 HSA_ENABLE_SDMA=1" WD_IDLE=300 bash dlrm_1n8.sh
```

2 nodes, 16 ranks (on node rank 0):

```bash
cd ~/DLRM_optimization
NPN=8 CT=dlrm-gfx1250-rocm102-ainic-run RCCL=bundled CUMEM=1 ITERS=600 BATCH=16384 \
  EXTRA="PYTORCH_TUNABLEOP_ENABLED=0 HSA_ENABLE_SDMA=0" WD_IDLE=300 TMO=3600 bash dlrm_2n.sh
```

Steady-state throughput is averaged from iteration 250, so run at least ~500 iterations. Below 1 it/s the progress bar shows s/it.
Records/s = it/s x batch per rank x ranks.

Settings that matter: keep `NCCL_PXN_DISABLE=1` (PXN relays with GPU-direct reset the node on gfx1250 DPX; recent RCCL develop keeps PXN off
by default on gfx1250), `NCCL_CUMEM_ENABLE=1` with RCCL 2.31.2, 32 channels (64 is 33% slower), TunableOp off (the bundled tuning files are
MI300 results).

## Results (image RCCL 2.31.2, 2026-10-31 and 2026-11-08/09)

| Config | Ranks | Batch / rank | Steady it/s | Records/s |
| --- | --- | --- | --- | --- |
| 1 node | 8 | 16,384 | 8.45 | 1.11 M |
| 1 node | 8 | 65,536 | 2.20 | 1.15 M |
| 1 node | 8 | 131,072 | 1.04 | 1.09 M |
| 1 node | 8 | 262,144 | 0.50 | 1.05 M |
| 2 nodes | 16 | 16,384 | 7.13 (full epoch: 7.09) | 1.87 M |
| 2 nodes | 16 | 65,536 | 1.89 | 1.99 M |
| 2 nodes | 16 | 131,072 | 0.93 | 1.96 M |

- 2 nodes give ~1.7x the records/s of 1 node (~85% per GPU). Throughput is flat from 16 K to 262 K per rank: larger batches fit in memory
  but do not speed up training.
- At batch 256 per rank, 2 nodes are latency-bound (~1.03x of 1 node).
- With RCCL develop 2.32.3 (LD_PRELOAD), batch 256 is ~9% faster, but the job aborts at exit with `double free or corruption (!prev)`;
  the image RCCL 2.31.2 exits cleanly.

## Known issues

| Symptom | Cause / fix |
| --- | --- |
| `double free or corruption (!prev)` / SIGABRT after training | RCCL develop builds (2.30.7, 2.32.3) at teardown; use the image RCCL 2.31.2. The `destroy_process_group()` patch is applied. |
| `DDP expects same model across all ranks ... inconsistent 0 params` | Old RCCL 2.30.x with `NCCL_CUMEM_ENABLE=1` in DPX; use RCCL 2.31.2. |
| `ncclAllToAllv is deprecated` on every iteration | PyTorch calls the old RCCL names; silenced by the shim in `docker/`. |
| 2-node job hangs before training / RDMA completion errors | Check NIC rail connectivity (all 4 rails) before the run. |
| Odd DPX partitions get about half the NIC bandwidth | Platform: RDMA into the far die crosses the AID-to-AID link (higher latency). Affects all-to-all heavy phases when both partitions share a NIC. |
