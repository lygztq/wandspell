#include "nanologging/nanologging.h"
#include <cuda_runtime.h>

#define CUDART_CALL(expr)                                                      \
  {                                                                            \
    cudaError_t e = (expr);                                                    \
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {                   \
      LOG_ERROR("CUDA runtime: ", cudaGetErrorString(e));                      \
    }                                                                          \
  }
