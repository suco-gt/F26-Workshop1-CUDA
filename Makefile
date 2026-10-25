.DEFAULT_GOAL := help
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -lineinfo -Xcompiler=-Wall,-Wextra
CPPFLAGS ?=
ARCH ?=
ARCH_FLAGS = $(if $(strip $(ARCH)),-arch=$(ARCH),)
N ?= 1000003
CXX ?= c++
CXXFLAGS ?= -O3 -Wall -Wextra -pedantic

STARTERS := build/ex1 build/ex2 build/ex3
SOLUTIONS := build/sol1 build/sol2 build/sol3
HEADERS := include/workshop.hpp include/cuda_check.cuh

.PHONY: help check-nvcc all solutions ex1 ex2 ex3 sol1 sol2 sol3 race \
        test test-ex1 test-ex2 test-ex3 host-test bench clean

help:
	@printf '%s\n' \
	  'CUDA C++ workshop (run from this directory)' \
	  '  make ex1 / ex2 / ex3   Build and run a starter (unfinished work should fail)' \
	  '  make sol1 / sol2 / sol3 Build and run a complete solution' \
	  '  make race              Run the deliberately incorrect race demonstration' \
	  '  make all               Build everything, without running or requiring a GPU' \
	  '  make solutions         Build only the three main solutions' \
	  '  make test              Run host checks and GPU correctness/argument tests' \
	  '  make test-ex1 / test-ex2 / test-ex3  Test your edited starter at multiple sizes' \
	  '  make host-test         Run C++ host checks without CUDA or a GPU' \
	  '  make bench             Compare correct CPU/atomic/shared reductions' \
	  '  make clean             Remove build/ only' \
	  'Use N=257 to change run size (0..16777216); or ./build/sol3 257.' \
	  'Override NVCC, NVCCFLAGS, CPPFLAGS, ARCH (e.g. sm_80), CXX, CXXFLAGS.' \
	  'After changing compiler flags/ARCH, run make clean before rebuilding.'

check-nvcc:
	@command -v "$(NVCC)" >/dev/null 2>&1 || { \
	  printf 'ERROR: nvcc not found (%s). Load or install the CUDA toolkit and put nvcc on PATH.\n' "$(NVCC)" >&2; exit 1; }

build:
	mkdir -p build

build/ex1: exercises/01_vector_add/starter.cu $(HEADERS) | build check-nvcc
	$(NVCC) -std=c++17 $(NVCCFLAGS) $(CPPFLAGS) -Iinclude $(ARCH_FLAGS) $< -o $@
build/sol1: exercises/01_vector_add/solution.cu $(HEADERS) | build check-nvcc
	$(NVCC) -std=c++17 $(NVCCFLAGS) $(CPPFLAGS) -Iinclude $(ARCH_FLAGS) $< -o $@
build/ex2: exercises/02_atomics/starter.cu $(HEADERS) | build check-nvcc
	$(NVCC) -std=c++17 $(NVCCFLAGS) $(CPPFLAGS) -Iinclude $(ARCH_FLAGS) $< -o $@
build/sol2: exercises/02_atomics/solution.cu $(HEADERS) | build check-nvcc
	$(NVCC) -std=c++17 $(NVCCFLAGS) $(CPPFLAGS) -Iinclude $(ARCH_FLAGS) $< -o $@
build/race: exercises/02_atomics/race.cu $(HEADERS) | build check-nvcc
	$(NVCC) -std=c++17 $(NVCCFLAGS) $(CPPFLAGS) -Iinclude $(ARCH_FLAGS) $< -o $@
build/ex3: exercises/03_reduction/starter.cu $(HEADERS) | build check-nvcc
	$(NVCC) -std=c++17 $(NVCCFLAGS) $(CPPFLAGS) -Iinclude $(ARCH_FLAGS) $< -o $@
build/sol3: exercises/03_reduction/solution.cu $(HEADERS) | build check-nvcc
	$(NVCC) -std=c++17 $(NVCCFLAGS) $(CPPFLAGS) -Iinclude $(ARCH_FLAGS) $< -o $@
build/bench: benchmark/bench.cu benchmark/cpu_sum.cpp benchmark/cpu_sum.hpp $(HEADERS) | build check-nvcc
	$(NVCC) -std=c++17 $(NVCCFLAGS) $(CPPFLAGS) -Iinclude $(ARCH_FLAGS) benchmark/bench.cu benchmark/cpu_sum.cpp -o $@
build/host-tests: tests/host_tests.cpp include/workshop.hpp benchmark/cpu_sum.cpp benchmark/cpu_sum.hpp | build
	$(CXX) -std=c++17 $(CXXFLAGS) $(CPPFLAGS) -Iinclude tests/host_tests.cpp benchmark/cpu_sum.cpp -o $@

all: $(STARTERS) $(SOLUTIONS) build/race build/bench
solutions: $(SOLUTIONS)

ex1 ex2 ex3 sol1 sol2 sol3 race: %: build/%
	./build/$@ "$(N)"
bench: build/bench
	./build/bench "$(N)"
host-test: build/host-tests
	./build/host-tests
test: host-test $(SOLUTIONS) build/bench
	sh tests/run.sh
test-ex1 test-ex2 test-ex3: test-%: build/%
	sh tests/run.sh $*
clean:
	rm -rf build
