#pragma once
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t status, const char* expression,
                       const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                 ": " + expression + ": " + cudaGetErrorString(status) +
                                 ". Check your GPU allocation, "
                                 "driver/toolkit, and kernel bounds.");
    }
}
#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)

inline void require_device() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        throw std::runtime_error(
            std::string("No usable CUDA device: ") +
            (status == cudaSuccess ? "zero visible devices" : cudaGetErrorString(status)) +
            ". On PACE, run inside an allocated GPU compute node. "
            "Check CUDA_VISIBLE_DEVICES and the driver.");
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.maxThreadsPerBlock < 256 || prop.maxGridSize[0] < 65536) {
        throw std::runtime_error("Device does not support this workshop's launch limits.");
    }
    std::cout << "GPU: " << prop.name << " (compute " << prop.major << '.'
              << prop.minor << ")\n";
}

// Explicit cudaFree calls on success remain in each program. These best-effort
// functions are only for exception paths, where another CUDA failure is possible.
inline void cleanup_device(void* pointer) noexcept {
    if (!pointer) return;
    const cudaError_t status = cudaFree(pointer);
    if (status != cudaSuccess) {
        std::cerr << "Cleanup cudaFree: " << cudaGetErrorString(status) << '\n';
    }
}

inline void cleanup_event(cudaEvent_t event) noexcept {
    if (!event) return;
    const cudaError_t status = cudaEventDestroy(event);
    if (status != cudaSuccess) {
        std::cerr << "Cleanup cudaEventDestroy: " << cudaGetErrorString(status) << '\n';
    }
}
