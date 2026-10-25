#include "cuda_check.cuh"
#include "workshop.hpp"

__global__ void vector_sum_shared(
    const float* a, float* out, int n) {
    __shared__ float s[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;

    // TODO: load a[i] or zero into s[t], then synchronize
    // TODO: reduce from blockDim.x / 2, all threads must reach each barrier
    // TODO: thread 0 atomically adds the block sum to the output
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
        vector_sum_shared<<<blocks, threads>>>(d_input, d_sum, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float actual = 0;
        CUDA_CHECK(cudaMemcpy(&actual, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_sum)); d_sum = nullptr;
        CUDA_CHECK(cudaFree(d_input)); d_input = nullptr;
        return workshop::check_sum(actual, workshop::reference_sum(input)) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        cleanup_device(d_sum);
        cleanup_device(d_input);
        return 1;
    }
}
