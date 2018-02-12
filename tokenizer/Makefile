CPP_SOURCE_FILES = $(wildcard *.cc)
CUDA_SOURCE_FILES = $(wildcard *.cu)
CPP_PROGRAMS = $(filter $(FILTER), $(filter-out build benchmark, $(CPP_SOURCE_FILES:%.cc=%)))
CUDA_PROGRAMS = $(filter $(FILTER), $(CUDA_SOURCE_FILES:%.cu=%))

CPP_OBJECTS = $(CPP_PROGRAMS:%=bin/%.o)
CUDA_OBJECTS = $(CUDA_PROGRAMS:%=bin/%.o)

PARALLEL_PROGRAMS = $(filter parallel%, $(CPP_PROGRAMS))
AVX_PROGRAMS = $(filter %avx, $(CPP_PROGRAMS))
PARALLEL_BINARIES = $(PARALLEL_PROGRAMS:%=bin/%)
AVX_BINARIES = $(AVX_PROGRAMS:%=bin/%)

CPP_BINARIES = $(CPP_PROGRAMS:%=bin/%)
CUDA_BINARIES = $(CUDA_PROGRAMS:%=bin/%)
BINARIES = $(CPP_BINARIES) $(CUDA_BINARIES)
RUN_BINARIES = $(BINARIES:%=%.run)

CFLAGS = $(OPTLEVEL)

all: $(RUN_BINARIES)

$(PARALLEL_BINARIES): CFLAGS += -fopenmp
$(AVX_BINARIES): CFLAGS += -mavx2

bin/benchmark.o: bin/%.o: %.cc
	g++ -Wall -std=c++11 $(CFLAGS) -c $< -o $@

$(CPP_OBJECTS): bin/%.o: %.cc
	g++ -Wall -std=c++11 $(CFLAGS) -c $< -o $@

$(CUDA_OBJECTS): bin/%.o: %.cu
	nvcc --compiler-options -Wall $(CFLAGS) -c $< -o $@

$(CPP_BINARIES): %: %.o bin/benchmark.o
	g++ -Wall -std=c++11 $(CFLAGS) bin/benchmark.o $(@:%=%.o) -o $@

$(CUDA_BINARIES): %: %.o bin/benchmark.o
	nvcc --compiler-options -Wall $(CFLAGS) bin/benchmark.o $(@:%=%.o) -o $@

$(RUN_BINARIES): bin clean $(BINARIES)
	$(@:%.run=%) --sample $(SAMPLE) --epochs $(EPOCHS)

bin:
	mkdir -p bin

clean:
	rm -f bin/*

.PHONY: all clean $(RUN_BINARIES)