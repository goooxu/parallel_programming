#include "helper_cuda.h"
#include <chrono>
#include <cstdio>
#include <mpi.h>
#include <numeric>
#include <vector>

using namespace std;
using namespace std::chrono;

void test(const char *d_buffer, size_t buffer_size, int *d_counter,
          steady_clock::time_point &kernel_launch_start,
          steady_clock::time_point &kernel_launch_end,
          steady_clock::time_point &kernel_execution_end);

int main(int argc, char *argv[]) {
  const char *input_file_name = argv[1];

  MPI_Init(nullptr, nullptr);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  FILE *fp = fopen(input_file_name, "rb");
  fseek(fp, 0, SEEK_END);
  size_t buffer_size = ftell(fp);

  size_t proc_buffer_size = buffer_size / world_size;
  size_t proc_buffer_offset = proc_buffer_size * world_rank;

  fseek(fp, proc_buffer_offset, SEEK_SET);

  cuInit(0);

  char *buffer;
  checkCudaErrors(
      cudaMallocHost(reinterpret_cast<void **>(&buffer), proc_buffer_size));

  size_t read_size = 0;
  while (read_size < proc_buffer_size) {
    read_size += fread(buffer + read_size, 1, proc_buffer_size - read_size, fp);
  }
  fclose(fp);

  char *d_buffer;
  int *d_counter;
  int counter;

  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&d_buffer), proc_buffer_size + 1));
  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&d_counter), sizeof(int)));

  checkCudaErrors(
      cudaMemcpy(d_buffer, buffer, proc_buffer_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_buffer + proc_buffer_size, 0, 1));
  checkCudaErrors(cudaMemset(d_counter, 0, sizeof(int)));

  steady_clock::time_point kernel_launch_start, kernel_launch_end,
      kernel_execution_end;
  test(d_buffer, proc_buffer_size, d_counter, kernel_launch_start,
       kernel_launch_end, kernel_execution_end);

  checkCudaErrors(
      cudaMemcpy(&counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_buffer));
  checkCudaErrors(cudaFree(d_counter));
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFreeHost(buffer));

  if (world_rank == 0) {
    for (int i = 1; i < world_size; i++) {
      int peer_counter;
      MPI_Recv(&peer_counter, 1, MPI_INT, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      counter += peer_counter;
    }
    printf("Number of separators: %d\n", counter);
  } else {
    MPI_Send(&counter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();

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