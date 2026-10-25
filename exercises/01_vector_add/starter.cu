#include "cuda_check.cuh"
#include "workshop.hpp"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO: compute the index, check bounds, and add the elements
}

int main(int argc, char** argv) {
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    try {
        int n = static_cast<int>(workshop::parse_size(argc, argv));
        require_device();
        if (n == 0) {
            std::cout << "EMPTY: no kernel to check.\n";
            return 0;
        }
        std::vector<float> a(n), b(n), c(n);
        workshop::fill_vectors(a, b);
        int bytes = n * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));
        CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));

        // TODO: choose threads per block, calculate blocks, and launch vector_add
        int threads = 0;
        int blocks = 0;
        // launch vector_add

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_c)); d_c = nullptr;
        CUDA_CHECK(cudaFree(d_b)); d_b = nullptr;
        CUDA_CHECK(cudaFree(d_a)); d_a = nullptr;
        return workshop::check_vectors(a, b, c) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        cleanup_device(d_c);
        cleanup_device(d_b);
        cleanup_device(d_a);
        return 1;
    }
}
