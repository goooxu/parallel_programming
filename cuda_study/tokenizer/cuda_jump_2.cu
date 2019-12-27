#include <chrono>
#include <cstdio>

using namespace std::chrono;

__global__ void warmup() {}

__global__ void foo(const char *buffer, size_t buffer_size, size_t *tokens,
                    int *token_index) {

  size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  if (i == 0) {
    if (buffer[0] == '\r' && buffer[1] == '\n') {
      tokens[atomicAdd(token_index, 1)] = i + 2;
    }
  } else if (i == buffer_size - 1) {
    if (buffer[i - 1] == '\r' && buffer[i] == '\n') {
      tokens[atomicAdd(token_index, 1)] = i + 1;
    }
  } else if (i < buffer_size - 1) {
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
}

void *allocHostMemory(size_t size) {
  char *buffer;
  cudaMallocHost(&buffer, size);
  return buffer;
}

void freeHostMemory(void *ptr) { cudaFreeHost(ptr); }

void test(const char *buffer, size_t buffer_size, size_t *breaks,
          size_t max_breaks_count, steady_clock::time_point &start,
          steady_clock::time_point &stop) {

  warmup<<<80, 64>>>();
  cudaDeviceSynchronize();

  size_t threads = (buffer_size + 1) / 2;
  const size_t block_dim = 1024;
  size_t blocks = threads / block_dim + ((threads % block_dim) != 0 ? 1 : 0);

  printf("blocks=%zu, block_dim=%zu\n", blocks, block_dim);

  auto t1 = steady_clock::now();

  const char *d_buffer;
  size_t *d_breaks;
  int *d_break_index;

  cudaMalloc((void **)&d_buffer, buffer_size);
  cudaMalloc((void **)&d_breaks, max_breaks_count * sizeof(size_t));
  cudaMalloc((void **)&d_break_index, sizeof(int));
  cudaMemcpy((void *)d_buffer, buffer, buffer_size, cudaMemcpyHostToDevice);
  cudaMemset(d_break_index, 0, sizeof(int));
  cudaDeviceSynchronize();

  auto t2 = steady_clock::now();

  foo<<<blocks, block_dim>>>(d_buffer, buffer_size, d_breaks, d_break_index);
  cudaDeviceSynchronize();

  auto t3 = steady_clock::now();

  cudaMemcpy(breaks, d_breaks, max_breaks_count * sizeof(size_t),
             cudaMemcpyDeviceToHost);
  cudaFree((void *)d_buffer);
  cudaFree((void *)d_breaks);
  cudaFree((void *)d_break_index);
  cudaDeviceSynchronize();

  auto t4 = steady_clock::now();

  printf("H2D memory copy took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t2 - t1).count() / 1000);
  printf("Kernel took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t3 - t2).count() / 1000);
  printf("D2H memory copy took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t4 - t3).count() / 1000);

  start = t1;
  stop = t4;
}