#include "helper_cuda.h"
#include "kernels.h"
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

void test(const char *d_buffer, size_t buffer_size, int *d_counter,
          steady_clock::time_point &kernel_launch_start,
          steady_clock::time_point &kernel_launch_end,
          steady_clock::time_point &kernel_execution_end) {
  const size_t threads = buffer_size / 2;
  const size_t block_dim = 256;
  const size_t grid_dim = 16;
  const size_t kernels = threads / grid_dim / block_dim;

  const size_t num_streams = 8;

  printf("Run %zu kernels (grid_dim=%zu, block_dim=%zu) on %zu "
         "streams, %zu kernels per stream\n",
         kernels, grid_dim, block_dim, num_streams, kernels / num_streams);

  cudaStream_t streams[num_streams];
  for (size_t i = 0; i < num_streams; i++) {
    checkCudaErrors(cudaStreamCreate(&streams[i]));
  }

  kernel_launch_start = steady_clock::now();

  const size_t kernels_per_stream = kernels / num_streams;

  for (size_t j = 0; j < kernels_per_stream; j++) {
    for (size_t i = 0; i < num_streams; i++) {
      const size_t k = i * kernels_per_stream + j;
      size_t offset = k * buffer_size / kernels;
      kernelFunc<<<grid_dim, block_dim, 0, streams[i]>>>(d_buffer + offset,
                                                         d_counter);
    }
  }

  kernel_launch_end = steady_clock::now();
  checkCudaErrors(cudaDeviceSynchronize());
  kernel_execution_end = steady_clock::now();

  for (size_t i = 0; i < num_streams; i++) {
    checkCudaErrors(cudaStreamDestroy(streams[i]));
  }
}