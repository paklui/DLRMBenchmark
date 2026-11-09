#!/usr/bin/env bash
# backup_image_20261029.sh — back up the working DLRM image dlrmv2-${USER}:dlrm-gfx1250-rocm102-ainic (ROCm 10.2 BKC 20261002, FBGEMM main d651cb36,
# torchrec 1.9.0, RCCL 2.31.2, ncclAllToAll(v) shim, AINIC rdma-core/libionic; used for the 1-node and 2-node runs of 2026-10-29):
#  1. docker tag + push to a private Docker Hub repo ($DOCKER_REPO) (tag dlrm-gfx1250-rocm102-ainic-20261029)
#  2. docker save | zstd to ~/docker-images/ (NFS home) + sha256 + zstd integrity test
# Run ON MIAMI:  (setsid nohup bash backup_image_20261029.sh > backup_image_20261029.out 2>&1 < /dev/null &)
set -u
SRC=dlrmv2-${USER}:dlrm-gfx1250-rocm102-ainic; DST=${DOCKER_REPO:?set DOCKER_REPO to <user>/<repo>}:dlrm-gfx1250-rocm102-ainic-20261029
DIR=$HOME/docker-images; F=$DIR/dlrm-gfx1250-rocm102-ainic-20261029.tar.zst; mkdir -p $DIR
log() { echo "[$(date '+%F %T %Z')] $*"; }
log "image id $(sudo -n docker image inspect -f '{{.Id}}' $SRC)"
sudo -n docker tag $SRC $DST && log "tagged $DST"
log "push start"; sudo -n docker push $DST > $DIR/push_20261029.log 2>&1; rc=$?; tail -3 $DIR/push_20261029.log; log "push rc=$rc"
log "save start"; sudo -n docker save $SRC | zstd -T16 -3 -q -o $F; log "save rc=${PIPESTATUS[*]} size $(du -h $F | cut -f1)"
sudo -n chown $(id -u):$(id -g) $F 2>/dev/null
sha256sum $F > $F.sha256 && log "sha256 $(cut -c1-16 $F.sha256)..."
zstd -t -q $F && log "zstd integrity test OK"
log "done. Restore: zstd -dc $F | sudo docker load"
