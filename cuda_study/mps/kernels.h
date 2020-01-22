static __device__ void coreFunc(const char *buffer, int *counter) {

  size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1;

  if (buffer[i] == '\r') {
    if (buffer[i + 1] == '\n') {
      atomicAdd(counter, 1);
    }
  } else if (buffer[i] == '\n') {
    if (buffer[i - 1] == '\r') {
      atomicAdd(counter, 1);
    }
  }
}

static __global__ void tinyKernelFunc(const char *buffer, int *counter) {
  coreFunc(buffer, counter);
}

const long long int SMALL_KERNEL_CLOCK_CYCLES = 1000000;

static __global__ void smallKernelFunc(const char *buffer, int *counter) {
  long long int start = clock64();

  coreFunc(buffer, counter);

  for (;;) {
    long long int duration = clock64() - start;
    if (duration >= SMALL_KERNEL_CLOCK_CYCLES) {
      break;
    }
  }
}

#define kernelFunc smallKernelFunc