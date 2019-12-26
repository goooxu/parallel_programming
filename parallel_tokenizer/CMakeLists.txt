cmake_minimum_required(VERSION 3.10)
project(ParallelTokenizer LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -mavx2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -Xcompiler -Wall -arch sm_${SM} -gencode arch=compute_${SM},code=sm_${SM}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB infra_files build.cc benchmark.cc)
file(GLOB cc_files *.cc)
file(GLOB openacc_files openacc_*.cc)
file(GLOB cu_files *.cu)
list(REMOVE_ITEM cc_files ${infra_files} ${openacc_files})

add_library(benchmarklib STATIC benchmark.cc)

add_executable(build build.cc)
foreach(file IN LISTS cc_files)
    get_filename_component(name ${file} NAME_WE)
    add_executable(${name} ${file})
    target_link_libraries(${name} benchmarklib OpenMP::OpenMP_CXX)
endforeach()
foreach(file IN LISTS cu_files)
    get_filename_component(name ${file} NAME_WE)
    cuda_add_executable(${name} ${file})
    target_link_libraries(${name} benchmarklib)
endforeach()