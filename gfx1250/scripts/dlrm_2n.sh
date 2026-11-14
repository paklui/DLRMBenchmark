#!/usr/bin/env bash
# dlrm_2n.sh — 2-node DLRM v2 (TorchRec), Miami = node rank 0 / master (192.168.1.215), Cairo = node rank 1 (192.168.1.218). Run ON MIAMI.
# Generates env/launch variants from the 2-node files (originals untouched):
#   utils/DLRM_set_env_variables_gfx1250_2n<NPN>[_bundled]_cumem<C>.sh : from DLRM_set_env_variables_gfx1250.sh (the validated 2-node RDMA env);
#        HIP_VISIBLE_DEVICES=0..NPN-1, NCCL_CUMEM_ENABLE=<C>, NCCL_RINGS kept only for NPN=2 (it describes 2x2 ranks), bundled = no LD_PRELOAD rccl-develop
#   DLRM_local_launch_gfx1250_2n<NPN>..._<ITERS>iter.sh : from DLRM_local_launch_gfx1250.sh; nproc_per_node NPN, nnodes WORLD/NPN, EXTRA exported before torchrun
# Env: NPN (ranks per node: 2 = devices 0,1 = both halves of physical GPU 1, as in TC90; 8 = all 4 GPUs in DPX), ITERS (2000), CT (container name, same on
#      both nodes), RCCL=develop|bundled, CUMEM (default 1, as in the 2-node env), EXTRA (default PYTORCH_TUNABLEOP_ENABLED=0: the shared tuning files are
#      MI300 results and both nodes would write them), TAG, PORT, WD_IDLE (watchdog: stop both containers when rank 0 logged epoch_stop and both logs are
#      idle this many s, default 180), BATCH (per rank, default 256), TMO (hard timeout per node in s, default 3600), GEN_ONLY=1.
# Core dumps are disabled (ulimit -c 0): on 2026-10-31 crashed ranks dumping GPU (SVM) memory while the GPU was hung locked up Miami.
set -u
NPN=${NPN:-8}; ITERS=${ITERS:-2000}; BATCH=${BATCH:-256}; PORT=${PORT:-29700}; CT=${CT:-dlrm-gfx1250-gfx1250dev-run}; RCCL=${RCCL:-develop}; CUMEM=${CUMEM:-1}
EXTRA=${EXTRA-PYTORCH_TUNABLEOP_ENABLED=0}; WD_IDLE=${WD_IDLE:-180}
MASTER=192.168.1.215; CAIRO=192.168.1.218; SSHC="ssh -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR $CAIRO"
DL=$HOME/DLRM_optimization; cd "$DL" || exit 1
SFX=_2n$NPN; [ "$RCCL" = bundled ] && SFX=${SFX}_bundled; SFX=${SFX}_cumem$CUMEM
TF32_MODE=${TF32_MODE:-1}; [ "$TF32_MODE" = 0 ] && SFX=${SFX}_fp32   # TF32_MODE=0: launch-script FP32 branch (no --allow_tf32; rocBLAS unless EXTRA sets TORCH_BLAS_PREFER_HIPBLASLT=1)
ENV=utils/DLRM_set_env_variables_gfx1250$SFX.sh; LAUNCH=DLRM_local_launch_gfx1250${SFX}_${ITERS}iter.sh
DEVS=$(seq -s, 0 $((NPN-1)))
sed -e "s/^export HIP_VISIBLE_DEVICES=.*/export HIP_VISIBLE_DEVICES=$DEVS/" \
    -e "s/^export NCCL_CUMEM_ENABLE=.*/export NCCL_CUMEM_ENABLE=$CUMEM   # dlrm_2n.sh/" utils/DLRM_set_env_variables_gfx1250.sh > $ENV
[ "$NPN" != 2 ] && sed -i 's/^export NCCL_RINGS=/# 2n: ring layout is for 2 nodes x 2 ranks, not used: export NCCL_RINGS=/' $ENV
[ "$RCCL" = bundled ] && sed -i -e 's/^export LD_PRELOAD=/# bundled RCCL: export LD_PRELOAD=/' -e 's/^export LD_LIBRARY_PATH=\/root\/rccl-develop/# bundled RCCL: export LD_LIBRARY_PATH=\/root\/rccl-develop/' $ENV
sed -e "s/^export NUM_TRAIN_BATCHES=2000\$/export NUM_TRAIN_BATCHES=$ITERS/" \
    -e "s/^export TF32_MODE=1/export TF32_MODE=$TF32_MODE/" \
    -e "s#--nnodes \$((\$WORLD_SIZE / 4))#--nnodes \$((\$WORLD_SIZE / $NPN))#" \
    -e "s/--nproc_per_node 4/--nproc_per_node $NPN/" \
    -e "s#utils/DLRM_set_env_variables_gfx1250.sh#$ENV#" \
    -e 's#^\( *\)torchrun #\1[ -n "${EXTRA:-}" ] \&\& export $EXTRA \&\& echo "EXTRA env: $EXTRA"\n\1torchrun #' \
    DLRM_local_launch_gfx1250.sh > $LAUNCH; chmod +x $LAUNCH $ENV
[ "${GEN_ONLY:-0}" = 1 ] && { echo "generated $ENV $LAUNCH"; exit 0; }

busy() { sudo -n fuser /dev/kfd 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | xargs -r ps -o comm= -p | grep -v -x inb-node-agent | wc -l; }
b0=$(busy); b1=$($SSHC "$(declare -f busy); busy")
[ "$b0" != 0 ] || [ "$b1" != 0 ] && { echo "ABORT: GPUs busy (Miami $b0, Cairo $b1)"; exit 2; }

WS=$((2 * NPN)); TS=$(date +%Y%m%d-%H%M%S); NAME=dlrm_2n${NPN}${SFX#_2n$NPN}_${CT#dlrm-gfx1250-}${TAG:+_$TAG}_${ITERS}iter_$TS
L0=$DL/training_logs/${NAME}_miami.log; L1=$DL/training_logs/${NAME}_cairo.log
echo "logs $L0 $L1"; echo "start $(date -u +%FT%TZ) NPN=$NPN WS=$WS BATCH=$BATCH ITERS=$ITERS CT=$CT RCCL=$RCCL CUMEM=$CUMEM EXTRA=$EXTRA"
sudo -n docker restart $CT >/dev/null; $SSHC "sudo -n docker restart $CT >/dev/null"
CMD="ulimit -c 0; source /opt/venv/bin/activate 2>/dev/null; cd /root/DLRM_optimization; EXTRA='$EXTRA' ./$LAUNCH"
$SSHC "sudo -n docker exec $CT timeout ${TMO:-3600} bash -c \"$CMD 1 $CAIRO $MASTER $PORT $WS $BATCH\"" > "$L1" 2>&1 & P1=$!
sudo -n docker exec $CT timeout ${TMO:-3600} bash -c "$CMD 0 $MASTER $MASTER $PORT $WS $BATCH" > "$L0" 2>&1 & P0=$!
( while sleep 15; do
    if grep -a -q '"key": "epoch_stop"' "$L0" 2>/dev/null; then
      a0=$(( $(date +%s) - $(stat -c %Y "$L0") )); a1=$(( $(date +%s) - $(stat -c %Y "$L1") ))
      if [ $a0 -gt $WD_IDLE ] && [ $a1 -gt $WD_IDLE ]; then
        echo "watchdog: epoch_stop and both logs idle >${WD_IDLE}s at $(date -u +%T), stopping $CT on both nodes"
        sudo -n docker stop $CT >/dev/null; $SSHC "sudo -n docker stop $CT >/dev/null"; break; fi
    fi
  done ) & W=$!
wait $P0; r0=$?; wait $P1; r1=$?; kill $W 2>/dev/null
echo "end $(date -u +%FT%TZ) rc_miami=$r0 rc_cairo=$r1"
sudo -n docker stop $CT >/dev/null 2>&1; $SSHC "sudo -n docker stop $CT >/dev/null 2>&1"
steady() { grep -a -o "| *[0-9]*/[0-9]* \[[0-9:]*<[^,]*, *[0-9.]*it/s\]" "$1" | sed -E 's/^\| *([0-9]+)\/.* ([0-9.]+)it\/s\]/\1 \2/' |
  awk '$1>=250{n++;s+=$2; if(min==""||$2<min)min=$2; if($2>max)max=$2} END{if(n)printf "mean %.2f min %.2f max %.2f (n=%d)",s/n,min,max,n; else printf "no data"}'; }
echo "SUMMARY $NAME"
echo "  iterations: $(grep -a -o 'Total number of iterations: [0-9]*' "$L0" | grep -o '[0-9]*$')   steady (rank 0): $(steady "$L0")"
echo "  records/s at mean: global batch $((WS * BATCH))"
echo "  comm: $(grep -a -m1 -o 'nRanks [0-9]* nNodes [0-9]*' "$L0")  RCCL $(grep -a -m1 -o 'RCCL version : [^ ]*' "$L0" | cut -d' ' -f4)  NET/IB channels: $(grep -a -c 'via NET/IB' "$L0")  NET/Socket: $(grep -a -c 'via NET/Socket' "$L0")"
for f in "$L0" "$L1"; do echo "  $(basename $f): aborts=$(grep -a -c -E 'exitcode *: -6|SIGABRT' $f) doublefree=$(grep -a -c -E 'double free|corrupted size' $f) IBV_WC=$(grep -a -c 'IBV_WC_' $f) watchdog_timeout=$(grep -a -c -E 'Watchdog caught|collective operation timeout' $f) errors=$(grep -a -c -E 'RuntimeError|Memory access fault' $f)"; done
