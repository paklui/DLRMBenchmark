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
| `docker/build_dlrm_rocm102.sh` → shim step → `build_dlrm_rocm102_ainic.sh` | Image chain used for all results. `BASE_IMAGE` = a ROCm 10.2 BKC PyTorch image with FBGEMM `main` `d651cb36` built for gfx1250 (in the image: `git checkout d651cb36`, `pip install -r fbgemm_gpu/requirements.txt` with the torch stack pinned, then `python setup.py --build-variant rocm --build-target default -DHIP_ROOT_DIR=$(rocm-sdk path --root) build develop` with `BUILD_ROCM_VERSION=10.2 ROCM_VERSION=100200 HIPBLAS_V2=1 PYTORCH_ROCM_ARCH=gfx1250`; ROCm is a Python SDK there, no `/opt/rocm`). `build_dlrm_rocm102.sh` adds TorchRec 1.9.0 and the DLRM runtime packages. Shim step (manual): compile `nccl_alltoall_shim.c` into `/opt/rccl-shim/libnccl_alltoall_shim.so` (build line in the file header) and `docker commit --change 'ENV LD_PRELOAD=/opt/rccl-shim/libnccl_alltoall_shim.so'` as `…-d651cb36-a2ashim`. `build_dlrm_rocm102_ainic.sh` adds the AINIC userspace (rdma-core with the ionic provider, perftest). Images are named `dlrmv2-${USER}:<tag>`; final image `dlrmv2-${USER}:dlrm-gfx1250-rocm102-ainic`. |
| `docker/build_dlrm_rocm102_gfx1250dev.sh` | Optional side branch, not part of the final image: the same image with FBGEMM from a ROCm development branch instead of `main` (used to compare FBGEMM versions; no speed difference). |
| `docker/nccl_alltoall_shim.c` | LD_PRELOAD shim that maps PyTorch's deprecated `ncclAllToAll(v)` calls to the current RCCL API (silences a per-iteration deprecation warning). |
| `docker/backup_image_20261029.sh` | Saves the image as `.tar.zst` with a checksum. |
| `docker/build_dlrm_rocm103.sh` + `docker/patch_fbgemm_rocmsetup.py` | The same image chain on a ROCm 10.3 PyTorch base (`BASE=<image>`): FBGEMM `d651cb36` built with `amdclang`, TorchRec 1.9.0, shim, AINIC user space. The patch removes two quoted warning flags in FBGEMM's `RocmSetup.cmake` that clang takes as file names. |
| `docker/extract_rccl_2312.sh` + `docker/mk_rccl231_container.sh` | Run RCCL 2.31.2 inside the ROCm 10.3 image (read-only bind mounts of `librccl.so.1` and its `.kpack`), because the bundled RCCL 2.32.3 hangs on 2 nodes (see Known issues). |
| `patches/dlrm_main_bf16_autocast_and_auroc.diff` | Optional BF16 autocast around the training step (`DLRM_BF16=1`), an end-of-training test AUROC (`DLRM_EVAL=1`), and the torchmetrics fix the AUROC code needs (`compute_on_step` was removed upstream). |
| `patches/dlrm_main_destroy_process_group.diff` | Calls `dist.destroy_process_group()` at the end of `dlrm_main.py`, so the process group is torn down while the runtime is alive (avoids a `double free or corruption (!prev)` abort at exit). |
| `patches/DLRM_optimization_local_changes.diff` | Local changes to the generic launch/setup scripts; base commit in `patches/BASE.txt`. |

## Running

1 node (on the node; recommended batch 131,072 per rank, ~10 min for 600 iterations):

```bash
cd ~/DLRM_optimization
CT=dlrm-gfx1250-rocm102-ainic-run RCCL=bundled CUMEM=1 ITERS=600 BATCH=131072 \
  EXTRA="PYTORCH_TUNABLEOP_ENABLED=0 HSA_ENABLE_SDMA=1" WD_IDLE=300 bash dlrm_1n8.sh
```

2 nodes, 16 ranks (on node rank 0):

```bash
cd ~/DLRM_optimization
NPN=8 CT=dlrm-gfx1250-rocm102-ainic-run RCCL=bundled CUMEM=1 ITERS=600 BATCH=131072 \
  EXTRA="PYTORCH_TUNABLEOP_ENABLED=0 HSA_ENABLE_SDMA=0" WD_IDLE=300 TMO=3600 bash dlrm_2n.sh
```

Steady-state throughput is averaged from iteration 250, so run at least ~500 iterations. Below 1 it/s the progress bar shows s/it (at 131,072 on 2 nodes: ~1.07 s/it = 0.93 it/s); `dlrm_2n.sh` then prints an invalid `steady` value in its summary (it averages only it/s samples), so read the s/it progress lines.
Records/s = it/s x batch per rank x ranks.

Settings that matter: keep `NCCL_PXN_DISABLE=1` (PXN relays with GPU-direct reset the node on gfx1250 DPX; recent RCCL develop keeps PXN off
by default on gfx1250), `NCCL_CUMEM_ENABLE=1` with RCCL 2.31.2, 32 channels (64 is 33% slower), TunableOp off (the bundled tuning files are
MI300 results).

## Results (image RCCL 2.31.2, 2026-10-31 and 2026-11-08/09)

| Config | Ranks | Batch / rank | Steady it/s | Records/s |
| --- | --- | --- | --- | --- |
| 1 node | 8 | 16,384 | 8.45 | 1.11 M |
| 1 node | 8 | 65,536 | 2.20 | 1.15 M |
| 1 node | 8 | **131,072 (recommended)** | 1.04 | 1.09 M |
| 1 node | 8 | 262,144 | 0.50 | 1.05 M |
| 2 nodes | 16 | 16,384 | 7.13 (full epoch: 7.09) | 1.87 M |
| 2 nodes | 16 | 65,536 | 1.89 | 1.99 M |
| 2 nodes | 16 | **131,072 (recommended)** | 0.93 | **1.96 M (1.79x)** |

- 2 nodes give ~1.7x the records/s of 1 node (~85% per GPU). Throughput is flat from 16 K to 262 K per rank: larger batches fit in memory
  but do not speed up training.
- Recommended 131,072 per rank: 2-node throughput at the top of the range (1.96 M records/s; 65,536 gave 1.99 M, within run-to-run variance)
  and the best 2-node scaling (1.79x).
- At batch 256 per rank, 2 nodes are latency-bound (~1.03x of 1 node).
- With RCCL develop 2.32.3 (LD_PRELOAD), batch 256 is ~9% faster, but the job aborts at exit with `double free or corruption (!prev)`;
  the image RCCL 2.31.2 exits cleanly.

## ROCm 10.3 image: FP32, GEMM tuning, BF16 (2026-11-13)

Image built with `docker/build_dlrm_rocm103.sh` on a ROCm 10.3 nightly PyTorch base (10.3.0a20261111, torch 2.13.0+rocm10.3, hipBLASLt/rocBLAS from the same nightly,
`rocprofiler-sdk` 1.5.1 so the PyTorch profiler records GPU kernels). Runs use `TF32_MODE=0` (FP32; see below), batch 131,072 per rank, 600 iterations (1 node) or
2,000 iterations (2 nodes); 2-node runs use RCCL 2.31.2 (`docker/mk_rccl231_container.sh`). Records/s = it/s x batch per rank x ranks (this README's convention);
the per-rank figure (it/s x batch) is the one the benchmark's `results.csv` reports.

| Config | Ranks | Steady it/s | Records/s | Per rank |
| --- | --- | --- | --- | --- |
| 1 node, ROCm 10.2 image, TF32 (table above) | 8 | 1.04 | 1.09 M | 137 K |
| 1 node, FP32 | 8 | 1.76 | 1.84 M | 230.5 K (mean of 5 runs) |
| 1 node, FP32, TunableOp tuned | 8 | 1.84 | 1.93 M | 240.9 K (single pair) |
| 1 node, BF16 autocast | 8 | 6.07 | 6.36 M | 795.4 K (single run) |
| 2 nodes, FP32 | 16 | 1.34 | 2.81 M | 175.7 K (mean of 2) |
| 2 nodes, FP32, TunableOp tuned | 16 | 1.43 | 3.00 M | 187.8 K (mean of 2) |
| 2 nodes, FP32, tuned, over the NICs only (`NCCL_MNNVL_ENABLE=0`) | 16 | 1.41 | 2.95 M | 184.5 K (single run) |

- **FP32, not TF32.** On this software the TF32 GEMM kernels are slower than FP32 (1-GPU `torch.matmul` at the DLRM shapes: TF32 14-22, FP32 30-40, BF16 210-675 TFLOP/s). Set `TF32_MODE=0`
  (the runners drop `--allow_tf32`) and `TORCH_BLAS_PREFER_HIPBLASLT=1` in `EXTRA`.
- **GEMMs are about 59% of GPU kernel time** (1 node: 453 of 570 ms per step; RCCL 18%, FBGEMM 14.5%). All FP32 GEMM winners found by tuning stay in one tile family
  (`MT32x32x16 / MI16x16x1`), so tuning is worth about 4%: PyTorch TunableOp +3.9% (1 node) / +6.9% (2 nodes); a 5x longer TunableOp search +0.4% more; hipBLASLt runtime tuning
  (`HIPBLASLT_TUNING_MODE=tune|cache`) +0.7%. Capping RCCL channels only slows the run down (4 channels: -16%, 16 channels: -2%).
- **TunableOp:** tune once, replay afterwards. Tune (50 iterations):
  `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_VERBOSE=1 PYTORCH_TUNABLEOP_FILENAME=<dir>/tunable_ops_result.csv`;
  replay: the same with `PYTORCH_TUNABLEOP_TUNING=0`. Tuning results are specific to the batch size, precision and software versions (the file records them as validators).
- **BF16 (optional, changes numerics).** Apply `patches/dlrm_main_bf16_autocast_and_auroc.diff` and run with `EXTRA="DLRM_BF16=1 ..."` (autocast around the training step;
  embeddings and optimizer stay FP32). 1 node: 3.45x the FP32 rate. Test AUROC (`DLRM_EVAL=1`, evaluated in FP32) with 1 node, batch 131,072 per rank, lr 0.005:

  | Training | AUROC |
  | --- | --- |
  | FP32, 2,000 iterations (half an epoch; 2 runs) | 0.7855 / 0.7893 |
  | BF16, 2,000 iterations (2 runs) | 0.7883 / 0.7862 |
  | FP32, 1 epoch (4,001 iterations, 4.2 B samples) | 0.7909 |
  | BF16, 1 epoch | 0.7910 |

  BF16 and FP32 are indistinguishable (the two FP32 runs differ by 0.004 from each other). Neither reached the MLPerf target of 0.80275 within one epoch with these settings
  (learning rate and batch size are not tuned for 131,072 per rank).
- **One epoch** is 4,001 iterations on 1 node (2,001 on 2 nodes) at this batch size.
- **Batch size:** on the ROCm 10.3 image, 1 node, FP32: 16,384 gives 218 K, 65,536 gives 228 K and 131,072 gives 230 K records/s per rank; a 262,144 run died with an NCCL collective timeout.

## Known issues

| Symptom | Cause / fix |
| --- | --- |
| `double free or corruption (!prev)` / SIGABRT after training | RCCL develop builds (2.30.7, 2.32.3) at teardown; use the image RCCL 2.31.2. The `destroy_process_group()` patch is applied. |
| `DDP expects same model across all ranks ... inconsistent 0 params` | Old RCCL 2.30.x with `NCCL_CUMEM_ENABLE=1` in DPX; use RCCL 2.31.2. |
| 2-node job on the ROCm 10.3 image hangs at the first collective (watchdog timeout, SIGABRT on every rank) | The image's RCCL 2.32.3 (`bd1130f`): both transports, same environment as the working setup; RCCL 2.31.2 passes. Use `docker/mk_rccl231_container.sh`. |
| `ValueError: Unexpected keyword arguments: compute_on_step` when evaluating AUROC | torchmetrics in the image no longer accepts `compute_on_step`; `patches/dlrm_main_bf16_autocast_and_auroc.diff` fixes the call. |
| `ncclAllToAllv is deprecated` on every iteration | PyTorch calls the old RCCL names; silenced by the shim in `docker/`. |
| 2-node job hangs before training / RDMA completion errors | Check NIC rail connectivity (all 4 rails) before the run. |
| Odd DPX partitions get about half the NIC bandwidth | Platform: RDMA into the far die crosses the AID-to-AID link (higher latency). Affects all-to-all heavy phases when both partitions share a NIC. |
