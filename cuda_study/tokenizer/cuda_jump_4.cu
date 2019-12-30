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

  int total_gpu_count;
  checkCudaErrors(cudaGetDeviceCount(&total_gpu_count));

  int master_device = 0;
  const size_t use_gpu_count = 4;

  printf(
      "Device count is %d, using first %zu devices, the master device is %d\n",
      total_gpu_count, use_gpu_count, master_device);

  for (size_t i = 0; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaFree(0));
    if (i > 0) {
      checkCudaErrors(cudaDeviceEnablePeerAccess(0, 0));
    }
  }

  const char *d_buffer[use_gpu_count];
  size_t *d_breaks;
  int *d_break_index;
  size_t d_buffer_padding_size =
      (buffer_size % 2048 != 0) ? (2048 - buffer_size % 2048) : 0;

  for (size_t i = 0; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaMalloc((void **)&d_buffer[i],
                               i == 0 ? buffer_size + d_buffer_padding_size + 1
                                      : (buffer_size + d_buffer_padding_size) /
                                            use_gpu_count));
  }

  checkCudaErrors(cudaSetDevice(master_device));
  checkCudaErrors(
      cudaMalloc((void **)&d_breaks, max_breaks_count * sizeof(size_t)));
  checkCudaErrors(cudaMalloc((void **)&d_break_index, sizeof(int)));

  auto t2 = steady_clock::now();

  for (size_t i = 0; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaMemcpyAsync(
        (void *)d_buffer[i],
        buffer + i * (buffer_size + d_buffer_padding_size) / use_gpu_count,
        (buffer_size + d_buffer_padding_size) / use_gpu_count,
        cudaMemcpyHostToDevice));
  }

  checkCudaErrors(cudaSetDevice(master_device));
  checkCudaErrors(cudaMemsetAsync((void *)(d_buffer[0] + buffer_size), 0,
                                  d_buffer_padding_size + 1));
  checkCudaErrors(cudaMemsetAsync(d_break_index, 0, sizeof(int)));

  for (size_t i = 1; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaMemcpyPeerAsync(
        (void *)(d_buffer[0] +
                 i * (buffer_size + d_buffer_padding_size) / use_gpu_count),
        0, d_buffer[i], i,
        (buffer_size + d_buffer_padding_size) / use_gpu_count));
  }
  for (size_t i = 1; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaDeviceSynchronize());
  }

  auto t3 = steady_clock::now();

  size_t threads = (buffer_size + d_buffer_padding_size) / 2;
  const size_t block_dim = 1024;
  size_t blocks = threads / block_dim;
  printf("Run kernel, blocks=%zu, block_dim=%zu\n", blocks, block_dim);

  checkCudaErrors(cudaSetDevice(master_device));
  work<<<blocks, block_dim>>>(d_buffer[0], d_breaks, d_break_index);
  checkCudaErrors(cudaDeviceSynchronize());

  auto t4 = steady_clock::now();

  checkCudaErrors(cudaMemcpy(breaks, d_breaks,
                             max_breaks_count * sizeof(size_t),
                             cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaFree((void *)d_buffer[i]));
  }
  checkCudaErrors(cudaSetDevice(master_device));
  checkCudaErrors(cudaFree((void *)d_breaks));
  checkCudaErrors(cudaFree((void *)d_break_index));
  for (size_t i = 0; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaDeviceSynchronize());
  }

  auto t5 = steady_clock::now();

  printf("CUDA initialization and memory allocation took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t2 - t1).count() / 1000);
  printf("H2D & P2P memory copy took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t3 - t2).count() / 1000);
  printf("Kernel took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t4 - t3).count() / 1000);
  printf("D2H memory copy and memory deallocation took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t5 - t4).count() / 1000);

  start = t1;
  stop = t5;
}