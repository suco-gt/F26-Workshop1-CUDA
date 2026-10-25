# Workshop 1: Intro to CUDA C++

## Table of Contents

- [Setup](#setup)
- [Exercises](#exercises)
  - [Exercise 1](#exercise-1)
  - [Exercise 2](#exercise-2)
  - [Exercise 3](#exercise-3)
- [Tests](#tests)
- [Benchmark](#benchmark)

## Setup (PACE ICE)

Run the commands below from this repository's root on your **allocated GPU compute node**.

```sh
nvidia-smi
nvcc --version
make help
```

## Setup (Kaggle)

If you can't get a GPU allocation in the ICE cluster, you can use Kaggle instead.

First, create a Kaggle notebook and at the top, select Settings -> Accelerator -> GPU T4 x2

Next, write this into the first cell:

```
%cd /kaggle/working
!git clone https://github.com/suco-gt/F26-Workshop1-CUDA.git
%cd /kaggle/working/F26-Workshop1-CUDA

!nvidia-smi
!nvcc --version
!make sol1 ARCH=native
```

Run the cell. It should display `PASS` at the bottom after finishing.

Create another cell after the first one and put this at the top:

```
%%writefile exercises/01_vector_add/starter.cu
```

Then, paste the contents from the corresponding file into the **same** cell, below the pasted first line.

Now, you can edit the exercises. To run the tests, create a third cell below and paste:

```
!make ex1 ARCH=native
!make test-ex1 ARCH=native
```

Then run the cell and you will see the output of the tests.

The other solutions can be completed/executed analogously.

To run the race example, paste `!make race ARCH=native` into a new cell and run the cell.

To run the benchmark, paste:

```
!make bench ARCH=native
!make bench ARCH=native N=1048579
```

into a new cell and run the cell.

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
