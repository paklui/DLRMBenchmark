// nccl_alltoall_shim.c — LD_PRELOAD shim for PyTorch builds that still call the deprecated RCCL entry points.
// The gfx1250 BKC PyTorch (ROCm/pytorch release/2.13) calls ncclAllToAll / ncclAllToAllv; RCCL >= 2.28 prints
// "ncclAllToAllv is deprecated, please use ncclAlltoAllv instead" on every call. This shim defines the old names and forwards
// to ncclAlltoAll / ncclAlltoAllv (same arguments), so the warning disappears without rebuilding PyTorch or RCCL.
// Not needed once PyTorch has pytorch/pytorch#197534 and #199124.
// Build (inside the ROCm 10.2 DLRM container, links the same librccl that torch uses):
//   R=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib
//   gcc -shared -fPIC -O2 -o libnccl_alltoall_shim.so nccl_alltoall_shim.c -L$R -l:librccl.so.1 -Wl,-rpath,$R
// Use: LD_PRELOAD=/root/DLRM_optimization/utils/libnccl_alltoall_shim.so
#include <stddef.h>

typedef int ncclResult_t;              // enum in rccl.h
typedef int ncclDataType_t;            // enum in rccl.h
typedef struct ncclComm* ncclComm_t;
typedef struct ihipStream_t* hipStream_t;

extern ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclComm_t comm, hipStream_t stream);
extern ncclResult_t ncclAlltoAllv(const void* sendbuff, const size_t sendcounts[], const size_t sdispls[], void* recvbuff,
                                  const size_t recvcounts[], const size_t rdispls[], ncclDataType_t datatype,
                                  ncclComm_t comm, hipStream_t stream);

__attribute__((visibility("default")))
ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                          ncclComm_t comm, hipStream_t stream) {
  return ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

__attribute__((visibility("default")))
ncclResult_t ncclAllToAllv(const void* sendbuff, const size_t sendcounts[], const size_t sdispls[], void* recvbuff,
                           const size_t recvcounts[], const size_t rdispls[], ncclDataType_t datatype,
                           ncclComm_t comm, hipStream_t stream) {
  return ncclAlltoAllv(sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype, comm, stream);
}
