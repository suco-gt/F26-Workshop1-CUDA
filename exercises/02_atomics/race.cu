#include "cuda_check.cuh"
#include "workshop.hpp"

// DELIBERATELY INCORRECT: shared updates race.
__global__ void reduce_racy(const float* input, float* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) *sum += input[i];
}

int main(int argc, char** argv) {
    float *d_input = nullptr, *d_sum = nullptr;
    try {
        int n = static_cast<int>(workshop::parse_size(argc, argv));
        require_device();
        if (n == 0) {
            std::cout << "EMPTY: sum=0, no kernel to check.\n";
            return 0;
        }
        auto input = workshop::reduction_input(n);
        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        reduce_racy<<<blocks, threads>>>(d_input, d_sum, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float actual = 0;
        CUDA_CHECK(cudaMemcpy(&actual, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_sum)); d_sum = nullptr;
        CUDA_CHECK(cudaFree(d_input)); d_input = nullptr;
        std::cout << "RACY: observed=" << actual
                  << ", expected=" << workshop::reference_sum(input)
                  << "\nA matching result does not prove correctness.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        cleanup_device(d_sum);
        cleanup_device(d_input);
        return 1;
    }
}
