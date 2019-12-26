#include <chrono>
#include <cstdio>

using namespace std::chrono;

const int WARMUP_CYCLES = 1024 * 1024;

__global__ void warmup() {
  float f = 0.0f;
  for (int i = 0; i < WARMUP_CYCLES; i++) {
    f = cos(f * 2);
  }
}

static const size_t BLOCK_SIZE = 1024;

__global__ void foo(const char *buffer, int buffer_size, size_t *tokens,
                    int *token_index) {

  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

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

  warmup<<<28, 1024>>>();
  cudaDeviceSynchronize();

  const char *d_buffer;
  size_t *d_breaks;
  int *d_break_index;

  auto t1 = steady_clock::now();
  cudaMalloc((void **)&d_buffer, buffer_size);
  cudaMalloc((void **)&d_breaks, max_breaks_count * sizeof(size_t));
  cudaMalloc((void **)&d_break_index, sizeof(int));
  cudaMemcpy((void *)d_buffer, buffer, buffer_size, cudaMemcpyHostToDevice);
  cudaMemset(d_break_index, 0, sizeof(int));
  cudaDeviceSynchronize();

  int threads = (buffer_size + 1) / 2;
  int blocks = threads / BLOCK_SIZE;
  if (threads % BLOCK_SIZE != 0) {
    blocks += 1;
  }

  auto t2 = steady_clock::now();
  foo<<<blocks, BLOCK_SIZE>>>(d_buffer, static_cast<int>(buffer_size), d_breaks,
                              d_break_index);
  cudaDeviceSynchronize();

  auto t3 = steady_clock::now();

  cudaMemcpy(breaks, d_breaks, max_breaks_count * sizeof(size_t),
             cudaMemcpyDeviceToHost);
  cudaFree((void *)d_buffer);
  cudaFree((void *)d_breaks);
  cudaFree((void *)d_break_index);
  cudaDeviceSynchronize();

  auto t4 = steady_clock::now();

  printf("H2D copy took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t2 - t1).count() / 1000);
  printf("Kernel took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t3 - t2).count() / 1000);
  printf("D2H memory copy took %.3f milliseconds\n",
         1.0f * duration_cast<microseconds>(t4 - t3).count() / 1000);

  start = t1;
  stop = t4;
}