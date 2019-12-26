SOURCE_FILES = $(filter $(FILTER), $(filter-out build.cc benchmark.cc, $(wildcard *.c?)))

CPP_PROGRAMS = $(patsubst %.cc, %, $(filter %.cc, $(SOURCE_FILES)))
AVX2_PROGRAMS = $(filter %avx, $(CPP_PROGRAMS))
OPENMP_PROGRAMS = $(filter openmp%, $(CPP_PROGRAMS))
OPENACC_PROGRAMS = $(filter openacc%, $(CPP_PROGRAMS))
NORMAL_CPP_PROGRAMS = $(filter-out $(OPENACC_PROGRAMS), $(CPP_PROGRAMS))
CUDA_PROGRAMS = $(patsubst %.cu, %, $(filter %.cu, $(SOURCE_FILES)))

AVX2_OBJECTS = $(AVX2_PROGRAMS:%=bin/%.o)
OPENMP_OBJECTS = $(OPENMP_PROGRAMS:%=bin/%.o)
OPENACC_OBJECTS = $(OPENACC_PROGRAMS:%=bin/%.o)
NORMAL_CPP_OBJECTS = $(NORMAL_CPP_PROGRAMS:%=bin/%.o)
CUDA_OBJECTS = $(CUDA_PROGRAMS:%=bin/%.o)

AVX2_BINARIES = $(AVX2_PROGRAMS:%=bin/%)
OPENMP_BINARIES = $(OPENMP_PROGRAMS:%=bin/%)
OPENACC_BINARIES = $(OPENACC_PROGRAMS:%=bin/%)
NORMAL_CPP_BINARIES = $(NORMAL_CPP_PROGRAMS:%=bin/%)
CUDA_BINARIES = $(CUDA_PROGRAMS:%=bin/%)

BINARIES = $(OPENACC_BINARIES) $(NORMAL_CPP_BINARIES) $(CUDA_BINARIES)
RUN_BINARIES = $(BINARIES:%=%.run)

CFLAGS =
$(AVX2_BINARIES): CFLAGS += -mavx2
$(OPENMP_BINARIES): CFLAGS += -fopenmp

all: $(RUN_BINARIES)

bin/benchmark.o $(NORMAL_CPP_OBJECTS): bin/%.o: %.cc
	g++ -std=c++11 -O2 -Wall $(CFLAGS) -c $< -o $@

$(OPENACC_OBJECTS): bin/%.o: %.cc
	pgc++ -acc -O2 -Minfo=accel -ta=tesla:cuda9.0 -c $< -o $@

$(CUDA_OBJECTS): bin/%.o: %.cu
	nvcc -std=c++11 -O2 -Xcompiler=-Wall -c $< -o $@
 
$(NORMAL_CPP_BINARIES): %: %.o bin/benchmark.o
	g++ -std=c++11 -O2 -Wall $(CFLAGS) bin/benchmark.o $(@:%=%.o) -o $@

$(OPENACC_BINARIES): %: %.o bin/benchmark.o
	pgc++ -acc -O2 -ta=tesla:cuda9.0 bin/benchmark.o $(@:%=%.o) -o $@

$(CUDA_BINARIES): %: %.o bin/benchmark.o
	nvcc -O2 -Xcompiler=-Wall bin/benchmark.o $(@:%=%.o) -o $@

$(RUN_BINARIES): bin clean $(BINARIES)
	$(@:%.run=%) --sample $(SAMPLE) --epochs $(EPOCHS)

bin:
	mkdir -p bin

clean:
	rm -f bin/*

.PHONY: all clean $(RUN_BINARIES)