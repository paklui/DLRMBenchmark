#!/usr/bin/env bash
# dlrm_1n8.sh — single-node DLRM v2 on Miami in DPX mode: 8 ranks = 8 logical devices = 4 physical gfx1250 GPUs, image dlrmv2-${USER}:dlrm-gfx1250-fbgemm-gfx1250dev
# (container dlrm-gfx1250-gfx1250dev-run). Generates its env/launch variants from the current gfx1250 files (originals untouched):
#   utils/DLRM_set_env_variables_gfx1250_1n8.sh : HIP_VISIBLE_DEVICES=0-7, 2-node NCCL_RINGS commented out, NCCL_CUMEM_ENABLE=0, rest as the current env file
#   DLRM_local_launch_gfx1250_1n8_<ITERS>iter.sh : nproc_per_node 8, nnodes WORLD_SIZE/8, NUM_TRAIN_BATCHES=<ITERS>, sources the 1n8 env file
# After epoch_stop, if the log is idle for 120 s (teardown hang), the container is stopped. Hard timeout 3600 s.
# Env: ITERS (default 2000), PORT (default 29660), GEN_ONLY=1 (only generate the files), CT (container, default dlrm-gfx1250-gfx1250dev-run),
#      RCCL=develop (default: LD_PRELOAD ~/rccl-develop, as in the 2N env) | bundled (the image's own RCCL; env file suffix _bundled), CUMEM (default 0),
#      EXTRA (space-separated VAR=value exported just before torchrun, so it overrides the env file and the launch script), TAG (log name),
#      PROF (container path: wrap torchrun in rocprofv3 --kernel-trace --stats --summary, output there), WD_IDLE (watchdog idle seconds, default 120).
# Run ON MIAMI:  cd ~/DLRM_optimization && (ITERS=2000 setsid nohup bash dlrm_1n8.sh > dlrm_1n8_launch.out 2>&1 < /dev/null &)
# Core dumps are disabled (ulimit -c 0): on 2026-10-31 crashed ranks dumping GPU (SVM) memory while the GPU was hung locked up Miami.
set -u
ITERS=${ITERS:-2000}; BATCH=${BATCH:-256}; PORT=${PORT:-29660}; CT=${CT:-dlrm-gfx1250-gfx1250dev-run}; RCCL=${RCCL:-develop}; CUMEM=${CUMEM:-0}
DL=$HOME/DLRM_optimization; cd "$DL" || exit 1
SFX=; [ "$RCCL" = bundled ] && SFX=_bundled; [ "$CUMEM" = 1 ] && SFX=${SFX}_cumem1
ENV=utils/DLRM_set_env_variables_gfx1250_1n8$SFX.sh; LAUNCH=DLRM_local_launch_gfx1250_1n8${SFX}_${ITERS}iter.sh
sed -e 's/^export HIP_VISIBLE_DEVICES=.*/export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7/' \
    -e 's/^export NCCL_RINGS=/# 1n8: 2-node ring layout, not used on 1 node: export NCCL_RINGS=/' \
    -e "s/^export NCCL_CUMEM_ENABLE=1.*/export NCCL_CUMEM_ENABLE=$CUMEM   # 1n8: with 1, collectives between the two DPX halves of one GPU hang with rccl-develop 2.30.7 (coll_check 2026-10-28)/" \
    utils/DLRM_set_env_variables_gfx1250.sh > $ENV
[ "$RCCL" = bundled ] && sed -i -e 's/^export LD_PRELOAD=/# 1n8 bundled RCCL: export LD_PRELOAD=/' -e 's/^export LD_LIBRARY_PATH=\/root\/rccl-develop/# 1n8 bundled RCCL: export LD_LIBRARY_PATH=\/root\/rccl-develop/' $ENV
sed -e "s/^export NUM_TRAIN_BATCHES=2000\$/export NUM_TRAIN_BATCHES=$ITERS/" \
    -e 's#--nnodes $(($WORLD_SIZE / 4))#--nnodes $(($WORLD_SIZE / 8))#' \
    -e 's/--nproc_per_node 4/--nproc_per_node 8/' \
    -e "s#utils/DLRM_set_env_variables_gfx1250.sh#$ENV#" \
    -e 's#^\( *\)torchrun #\1[ -n "${EXTRA:-}" ] \&\& export $EXTRA \&\& echo "EXTRA env: $EXTRA"\n\1${PROF:+rocprofv3 --kernel-trace --stats --summary --output-format csv -d $PROF -- }torchrun #' \
    DLRM_local_launch_gfx1250.sh > $LAUNCH; chmod +x $LAUNCH $ENV
[ "${GEN_ONLY:-0}" = 1 ] && { echo "generated $ENV $LAUNCH"; exit 0; }

c=$(for p in TransferBench all_reduce_perf sendrecv_perf torchrun hipblaslt-bench rocblas-bench; do pgrep -xc $p; done | paste -sd+ | bc)
[ "$c" != 0 ] && { echo "ABORT: GPU test processes running ($c)"; exit 2; }
others=$(sudo -n fuser /dev/kfd 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | xargs -r ps -o comm= -p | grep -v -x inb-node-agent | wc -l)
[ "$others" != 0 ] && { echo "ABORT: other processes hold /dev/kfd"; exit 2; }

TS=$(date +%Y%m%d-%H%M%S); LOG=$DL/training_logs/dlrm_1n8${SFX}_${CT#dlrm-gfx1250-}${TAG:+_$TAG}_${ITERS}iter_$TS.log
echo "log $LOG"; echo "start $(date -u +%FT%TZ) ITERS=$ITERS CT=$CT RCCL=$RCCL CUMEM=$CUMEM EXTRA=${EXTRA:-} PROF=${PROF:-}"
sudo -n docker start $CT >/dev/null
( while sleep 15; do
    [ -f "$LOG" ] || continue
    if grep -a -q '"key": "epoch_stop"' "$LOG" && [ $(( $(date +%s) - $(stat -c %Y "$LOG") )) -gt ${WD_IDLE:-120} ]; then
      echo "watchdog: epoch_stop and log idle >${WD_IDLE:-120} s at $(date -u +%T), stopping $CT"; sudo -n docker stop $CT >/dev/null; break; fi
  done ) & W=$!
sudo -n docker exec $CT timeout 3600 bash -c "ulimit -c 0; source /opt/venv/bin/activate 2>/dev/null; cd /root/DLRM_optimization; EXTRA='${EXTRA:-}' PROF='${PROF:-}' ./$LAUNCH 0 127.0.0.1 127.0.0.1 $PORT 8 $BATCH" > "$LOG" 2>&1
rc=$?; kill $W 2>/dev/null
echo "end $(date -u +%FT%TZ) rc=$rc"
sudo -n docker stop $CT >/dev/null 2>&1
grep -a -o 'Epoch 0:[^]]*it/s]' "$LOG" | tail -1
grep -a -c -E 'RuntimeError|Memory access fault|IBV_WC|Watchdog caught' "$LOG" | sed 's/^/error lines: /'
