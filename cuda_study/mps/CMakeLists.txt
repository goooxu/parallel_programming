cmake_minimum_required(VERSION 3.10)
project(MPS LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1 -O2 -Wall ${OpenMP_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -Xcompiler -Wall -arch sm_${SM} -gencode arch=compute_${SM},code=sm_${SM} ${OpenMP_CUDA_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB cu_files *.cu)

add_library(benchmarklib STATIC benchmark.cc)
add_library(mpi_benchmarklib STATIC mpi_benchmark.cc)

add_executable(datum datum.cc)
foreach(file IN LISTS cu_files)
    get_filename_component(name ${file} NAME_WE)
    cuda_add_executable(${name} ${file})
    target_link_libraries(${name} benchmarklib cuda)
endforeach()
foreach(file IN LISTS cu_files)
    get_filename_component(name ${file} NAME_WE)
    cuda_add_executable("mpi_${name}" ${file})
    target_link_libraries("mpi_${name}" mpi_benchmarklib cuda ${MPI_LIBRARIES})
endforeach()