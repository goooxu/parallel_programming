SOURCE_FILES = $(wildcard *.cc)
PROGRAMS = $(filter $(FILTER), $(filter-out build benchmark, $(SOURCE_FILES:%.cc=%)))

PARALLEL_PROGRAMS = $(filter parallel%, $(PROGRAMS))
AVX_PROGRAMS = $(filter %avx, $(PROGRAMS))

CFLAGS = $(FLAGS)

BINARIES =  $(PROGRAMS:%=bin/%)
PARALLEL_BINARIES = $(PARALLEL_PROGRAMS:%=bin/%)
AVX_BINARIES = $(AVX_PROGRAMS:%=bin/%)
RUN_BINARIES = $(BINARIES:%=%.run)

all: $(RUN_BINARIES)

$(PARALLEL_BINARIES): CFLAGS += -fopenmp
$(AVX_BINARIES): CFLAGS += -mavx2

$(BINARIES): bin clean benchmark.cc
	g++ -Wall $(CFLAGS) benchmark.cc $(@:bin/%=%.cc) -o $@

$(RUN_BINARIES): $(BINARIES)
	$(@:%.run=%) --sample $(SAMPLE) --epoch $(EPOCHS)

bin:
	mkdir -p bin

clean:
	rm -f bin/*

.PHONY: all clean $(RUN_BINARIES)