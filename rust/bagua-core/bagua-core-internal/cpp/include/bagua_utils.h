
#ifndef __BAGUA_UTILS_HPP__
#define __BAGUA_UTILS_HPP__

#define CUDACHECK(cmd) do { cudaError_t e = cmd; if( e != cudaSuccess ) { printf("Failed: Cuda error %s:%d '%s'\n", __FILE__,__LINE__,cudaGetErrorString(e)); exit(EXIT_FAILURE); } } while(0)
#define NCCLCHECK(cmd) do { ncclResult_t r = cmd; if (r!= ncclSuccess) { printf("Failed, NCCL error %s:%d '%s'\n", __FILE__,__LINE__,ncclGetErrorString(r)); exit(EXIT_FAILURE); } } while(0)

#define ALIGN_SIZE(size, align) (((size) + (align) - 1) / (align) * (align))
#define DIVUP(x, y) (((x)+(y)-1)/(y))

#include<nccl.h>


ncclResult_t ncclAllToAll(void *sendbuf, void *recvbuf,
                          size_t count,
                          ncclDataType_t datatype,
                          ncclComm_t comm,
                          int nranks,
                          int rank,
                          cudaStream_t stream);

#endif
