#include "helper_cuda.h"
#include <chrono>
#include <cstdio>

using namespace std::chrono;

__global__ void work(const char *buffer, size_t *tokens, int *token_index) {

  size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1;

  if (buffer[i] == '\r') {
    if (buffer[i + 1] == '\n') {
      tokens[atomicAdd(token_index, 1)] = i + 2;
    }
  } else if (buffer[i] == '\n') {
    if (buffer[i - 1] == '\r') {
      tokens[atomicAdd(token_index, 1)] = i + 1;
    }
  }
}

void *allocHostMemory(size_t size) {
  char *buffer;
  checkCudaErrors(cudaMallocHost(&buffer, size));
  return buffer;
}

void freeHostMemory(void *ptr) { checkCudaErrors(cudaFreeHost(ptr)); }

void test(const char *buffer, size_t buffer_size, size_t *breaks,
          size_t max_breaks_count, steady_clock::time_point &start,
          steady_clock::time_point &stop) {

  auto t1 = steady_clock::now();

  checkCudaErrors(cudaFree(0));

  const char *d_buffer;
  size_t *d_breaks;
  int *d_break_index;
  size_t d_buffer_padding_size =
      (buffer_size % 2048 != 0) ? (2048 - buffer_size % 2048) : 0;

  checkCudaErrors(
      cudaMalloc((void **)&d_buffer, buffer_size + d_buffer_padding_size + 1));
  checkCudaErrors(
      cudaMalloc((void **)&d_breaks, max_breaks_count * sizeof(size_t)));
  checkCudaErrors(cudaMalloc((void **)&d_break_index, sizeof(int)));

  auto t2 = steady_clock::now();

  checkCudaErrors(cudaMemcpy((void *)d_buffer, buffer, buffer_size,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset((void *)(d_buffer + buffer_size), 0,
                             d_buffer_padding_size + 1));
  checkCudaErrors(cudaMemset(d_break_index, 0, sizeof(int)));
  checkCudaErrors(cudaDeviceSynchronize());

  auto t3 = steady_clock::now();

  size_t threads = (buffer_size + d_buffer_padding_size) / 2;
  const size_t block_dim = 1024;
  size_t blocks = threads / block_dim;
  printf("Run kernel, blocks=%zu, block_dim=%zu\n", blocks, block_dim);

  work<<<blocks, block_dim>>>(d_buffer, d_breaks, d_break_index);
  checkCudaErrors(cudaDeviceSynchronize());

  auto t4 = steady_clock::now();

  checkCudaErrors(cudaMemcpy(breaks, d_breaks,
                             max_breaks_count * sizeof(size_t),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree((void *)d_buffer));
  checkCudaErrors(cudaFree((void *)d_breaks));
  checkCudaErrors(cudaFree((void *)d_break_index));
  checkCudaErrors(cudaDeviceSynchronize());

  auto t5 = steady_clock::now();

  printf("CUDA initialization and memory allocation took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t2 - t1).count() / 1000);
  printf("H2D memory copy took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t3 - t2).count() / 1000);
  printf("Kernel took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t4 - t3).count() / 1000);
  printf("D2H memory copy and memory deallocation took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t5 - t4).count() / 1000);

  start = t1;
  stop = t5;
}