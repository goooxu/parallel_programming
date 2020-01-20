#ifndef HELPER_CUDA_H_
#define HELPER_CUDA_H_

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *file,
                              const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d from file <%s>, "
            "line %i.\n",
            err, file, line);
    exit(EXIT_FAILURE);
  }
}

#endif