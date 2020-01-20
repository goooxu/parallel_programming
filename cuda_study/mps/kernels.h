const long long int CLOCK_CYCLES = 100 * 1000;

__device__ void coreFunc(const char *buffer, size_t offset, size_t *separators,
                         int *index) {
  size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1;

  if (buffer[i] == '\r') {
    if (buffer[i + 1] == '\n') {
      separators[atomicAdd(index, 1)] = offset + i + 2;
    }
  } else if (buffer[i] == '\n') {
    if (buffer[i - 1] == '\r') {
      separators[atomicAdd(index, 1)] = offset + i + 1;
    }
  }
}

__global__ void smallKernelFunc(const char *buffer, size_t offset,
                                size_t *separators, int *index) {

  coreFunc(buffer, offset, separators, index);
}

__global__ void largeKernelFunc(const char *buffer, size_t offset,
                                size_t *separators, int *index) {

  long long int start = clock64();

  coreFunc(buffer, offset, separators, index);

  for (;;) {
    long long int duration = clock64() - start;
    if (duration >= CLOCK_CYCLES) {
      break;
    }
  }
}

#define kernelFunc smallKernelFunc