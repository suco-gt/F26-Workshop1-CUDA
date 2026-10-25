# Workshop 1: Intro to CUDA C++

## Table of Contents

- [Setup](#setup)
- [Exercises](#exercises)
  - [Exercise 1](#exercise-1)
  - [Exercise 2](#exercise-2)
  - [Exercise 3](#exercise-3)
- [Tests](#tests)
- [Benchmark](#benchmark)

## Setup

Run the commands below from this repository's root on your **allocated GPU compute node**.

```sh
nvidia-smi
nvcc --version
make help
```

## Exercises

Complete the TODOs in each `starter.cu`. Each Make command builds and runs
its program. Solutions are in the corresponding `solution.cu` files.
To change input size, add `N=257`, for example: `make ex1 N=257`.
The default is 1,000,003; supported sizes are 0–16,777,216.

### Exercise 1

Vector addition: edit [starter.cu](exercises/01_vector_add/starter.cu).

```sh
make ex1
make sol1
make test-ex1
```

### Exercise 2

Atomics: edit [starter.cu](exercises/02_atomics/starter.cu).

```sh
make race       # Deliberately incorrect; a matching result does not prove correctness.
make ex2
make sol2
make test-ex2
```

### Exercise 3

Shared-memory reduction: edit [starter.cu](exercises/03_reduction/starter.cu).

```sh
make ex3
make sol3
make test-ex3
```

## Tests

```sh
make test       # Test complete solutions and benchmark validation.
make host-test  # Host checks only; no CUDA toolkit or GPU needed.
```

## Benchmark

Compare the CPU, per-element atomic, and shared-memory reductions:

```sh
make bench
make bench N=1048579
```

Kernel timings exclude transfers and output reset. End-to-end timings include
transfers, output reset, and synchronization. Both exclude allocation.

```sh
make all        # Build everything without running.
make clean      # Remove generated files.
```
