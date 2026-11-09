#!/usr/bin/env bash
# dlrm_2n_verify.sh — short 2-node DLRM run (best setup, 16 ranks, ITERS default 500) with RCCL per-collective logging
# (NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,NET,COLL) and the back-end NIC RDMA byte counters of all 4 ionic NICs sampled every 5 s on both nodes.
# Note: RoCE traffic is not counted in the netdev counters (ip -s link / ethtool tx_bytes); use /sys/class/infiniband/ionic_*/ports/1/hw_counters.
# Env: ITERS (500), BATCH (256), VERIFY_EXTRA (extra VAR=value, e.g. NCCL_MNNVL_ENABLE=0: since 2026-10-30 the AIFM vPOD spans both trays, so without it
# cross-node traffic goes over the scale-up fabric and the back-end NICs stay idle).
# Waits until no other dlrm_2n.sh / dlrm_2n_long job is running. Run ON MIAMI:  bash dlrm_2n_verify.sh  (output also in verify_<ts>/)
set -u
cd ~/DLRM_optimization
CAIRO=192.168.1.218; SSHC="ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR $CAIRO"
while ps -u $(whoami) -o args= | grep -q -E "^bash dlrm_2n(_long_20261029)?\.sh$"; do sleep 15; done
OUT=$HOME/DLRM_optimization/verify_$(date +%Y%m%d-%H%M%S); mkdir -p $OUT
SAMPLER='while [ -f '$OUT'/sampling ]; do t=$(date +%s.%N); for d in 0 1 2 3; do c=/sys/class/infiniband/ionic_$d/ports/1/hw_counters; echo "$t ionic_$d $(cat $c/tx_rdma_ucast_bytes) $(cat $c/rx_rdma_ucast_bytes)"; done; sleep 5; done'
touch $OUT/sampling
bash -c "$SAMPLER" > $OUT/nic_miami.txt 2>&1 & S0=$!
$SSHC "bash -c '$SAMPLER'" > $OUT/nic_cairo.txt 2>&1 & S1=$!
NPN=8 CT=dlrm-gfx1250-rocm102-ainic-run RCCL=bundled CUMEM=1 ITERS=${ITERS:-500} \
  BATCH=${BATCH:-256} EXTRA="PYTORCH_TUNABLEOP_ENABLED=0 HSA_ENABLE_SDMA=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,COLL ${VERIFY_EXTRA:-}" TAG=verify bash dlrm_2n.sh > $OUT/run.out 2>&1
rm -f $OUT/sampling; $SSHC "rm -f $OUT/sampling"; sleep 7; kill $S0 $S1 2>/dev/null
grep -E '^(ABORT|start|end|SUMMARY|  )' $OUT/run.out
L0=$(grep -o '/home[^ ]*_miami.log' $OUT/run.out | head -1)
echo "RCCL collective log lines on Miami (8 ranks): $(grep -a -c -E 'NCCL INFO (AllToAll|AlltoAll|AllReduce|Broadcast|AllGather|ReduceScatter|Send|Recv)' "$L0")"
grep -a -o -E 'NCCL INFO (AllToAllv?|AlltoAllv?|AllReduce|Broadcast|AllGather|ReduceScatter|Send|Recv)' "$L0" | sort | uniq -c | sort -rn | head -8
for n in miami cairo; do
  echo "== back-end NIC RDMA rate on $n (GB/s per 5 s sample; tx/rx), busiest 6 samples + totals"
  python3 - $OUT/nic_$n.txt <<'PY'
import sys, collections
rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) == 4 and p[2].isdigit(): rows[p[1]].append((float(p[0]), int(p[2]), int(p[3])))
rates = collections.defaultdict(dict); tot = {}
for d, r in sorted(rows.items()):
    tot[d] = ((r[-1][1] - r[0][1]) / 1e9, (r[-1][2] - r[0][2]) / 1e9)
    for (t0, a0, b0), (t1, a1, b1) in zip(r, r[1:]):
        rates[round(t1)][d] = ((a1 - a0) / (t1 - t0) / 1e9, (b1 - b0) / (t1 - t0) / 1e9)
busy = sorted(rates.items(), key=lambda kv: -sum(x[0] + x[1] for x in kv[1].values()))[:6]
for t, v in sorted(busy):
    print("  t+%4ds " % (t - min(rates)), "  ".join("%s %5.1f/%5.1f" % (d, a, b) for d, (a, b) in sorted(v.items())))
print("  totals over the run (GB tx/rx):", "  ".join("%s %.1f/%.1f" % (d, a, b) for d, (a, b) in sorted(tot.items())))
PY
done
echo "files: $OUT"
