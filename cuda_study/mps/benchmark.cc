#include "helper_cuda.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <memory>
#include <numeric>
#include <vector>

using namespace std;
using namespace std::chrono;

void test(const char *d_buffer, size_t buffer_size, size_t buffer_offset,
          size_t *d_separators, int *d_index,
          steady_clock::time_point &kernel_launch_start,
          steady_clock::time_point &kernel_launch_end,
          steady_clock::time_point &kernel_execution_end);

int main(int argc, char *argv[]) {
  const char *input_file_name = argv[1];
  size_t max_separator_count = strtoull(argv[2], nullptr, 10);

  FILE *fp = fopen(input_file_name, "rb");
  fseek(fp, 0, SEEK_END);
  size_t buffer_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  char *buffer;
  checkCudaErrors(
      cudaMallocHost(reinterpret_cast<void **>(&buffer), buffer_size));

  size_t read_size = 0;
  while (read_size < buffer_size) {
    read_size += fread(buffer + read_size, 1, buffer_size - read_size, fp);
  }
  fclose(fp);

  vector<size_t> separators(max_separator_count);

  char *d_buffer;
  size_t *d_separators;
  int *d_index;

  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&d_buffer), buffer_size + 1));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_separators),
                             separators.size() * sizeof(size_t)));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_index), sizeof(int)));

  checkCudaErrors(
      cudaMemcpy(d_buffer, buffer, buffer_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_buffer + buffer_size, 0, 1));
  checkCudaErrors(
      cudaMemset(d_separators, 0, separators.size() * sizeof(size_t)));
  checkCudaErrors(cudaMemset(d_index, 0, sizeof(int)));

  steady_clock::time_point kernel_launch_start, kernel_launch_end,
      kernel_execution_end;
  test(d_buffer, buffer_size, 0, d_separators, d_index, kernel_launch_start,
       kernel_launch_end, kernel_execution_end);

  checkCudaErrors(cudaMemcpy(separators.data(), d_separators,
                             separators.size() * sizeof(size_t),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_buffer));
  checkCudaErrors(cudaFree(d_separators));
  checkCudaErrors(cudaFree(d_index));
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFreeHost(buffer));

  size_t sum =
      accumulate(separators.begin(), separators.end(), static_cast<size_t>(0));
  printf("Sum of separators: %zu\n", sum);

  printf(
      "Kernel launch took %.3f milliseconds\n",
      1.0f *
          duration_cast<microseconds>(kernel_launch_end - kernel_launch_start)
              .count() /
          1000);
  printf("Kernel launch and execution took %.3f milliseconds\n",
         1.0f *
             duration_cast<microseconds>(kernel_execution_end -
                                         kernel_launch_start)
                 .count() /
             1000);

  return 0;
}
