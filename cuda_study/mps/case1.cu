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

  printf("Run %zu kernels (grid_dim=%zu, block_dim=%zu)\n", kernels, grid_dim,
         block_dim);

  kernel_launch_start = steady_clock::now();
  for (size_t k = 0; k < kernels; k++) {
    size_t offset = k * buffer_size / kernels;
    kernelFunc<<<grid_dim, block_dim>>>(
        d_buffer + offset, buffer_offset + offset, d_separators, d_index);
  }
  kernel_launch_end = steady_clock::now();
  checkCudaErrors(cudaDeviceSynchronize());
  kernel_execution_end = steady_clock::now();
}