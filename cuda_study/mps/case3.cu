#include "helper_cuda.h"
#include "kernels.h"
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

void test(const char *d_buffer, size_t buffer_size, size_t buffer_offset,
          size_t *d_separators, int *d_index,
          steady_clock::time_point &kernel_launch_start,
          steady_clock::time_point &kernel_launch_end,
          steady_clock::time_point &kernel_execution_end) {
  const size_t threads = buffer_size / 2;
  const size_t block_dim = 256;
  const size_t grid_dim = 16;
  const size_t kernels = threads / grid_dim / block_dim;

  const size_t num_threads = 8;

  printf("Run %zu kernels (grid_dim=%zu, block_dim=%zu) on %zu "
         "threads, %zu kernels per thread\n",
         kernels, grid_dim, block_dim, num_threads, kernels / num_threads);

  CUcontext ctxs[num_threads];

#pragma omp parallel num_threads(num_threads)
  checkCudaErrors(cuCtxCreate(&ctxs[omp_get_thread_num()], 0, 0));

  kernel_launch_start = steady_clock::now();

#pragma omp parallel num_threads(num_threads)
  {
    const size_t kernels_per_thread = kernels / omp_get_num_threads();
    int i = omp_get_thread_num();

    for (size_t j = 0; j < kernels_per_thread; j++) {
      const size_t k = i * kernels_per_thread + j;
      size_t offset = k * buffer_size / kernels;
      kernelFunc<<<grid_dim, block_dim>>>(
          d_buffer + offset, buffer_offset + offset, d_separators, d_index);
    }
  }

  kernel_launch_end = steady_clock::now();
  checkCudaErrors(cudaDeviceSynchronize());
  kernel_execution_end = steady_clock::now();

#pragma omp parallel num_threads(num_threads)
  checkCudaErrors(cuCtxDestroy(ctxs[omp_get_thread_num()]));
}