cmake_minimum_required(VERSION 3.10)
project(Tokenizer LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall ${OpenMP_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -Xcompiler -Wall -arch sm_${SM} -gencode arch=compute_${SM},code=sm_${SM} ${OpenMP_CUDA_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB cc_files *.cc)
file(GLOB cu_files *.cu)
list(FILTER cc_files EXCLUDE REGEX "datum\.cc|benchmark\.cc")

add_library(benchmarklib STATIC benchmark.cc)

add_executable(datum datum.cc)
foreach(file IN LISTS cc_files)
    get_filename_component(name ${file} NAME_WE)
    add_executable(${name} ${file})
    target_link_libraries(${name} benchmarklib)
endforeach()
foreach(file IN LISTS cu_files)
    get_filename_component(name ${file} NAME_WE)
    cuda_add_executable(${name} ${file})
    target_link_libraries(${name} benchmarklib)
endforeach()
