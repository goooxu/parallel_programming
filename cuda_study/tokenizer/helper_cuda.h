#ifndef HELPER_CUDA_H_
#define HELPER_CUDA_H_

#include <cstdio>
#include <cuda.h>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
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

#endif