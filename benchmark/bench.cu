#include "cuda_check.cuh"
#include "workshop.hpp"
#include "cpu_sum.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>

// Kept identical to the main solution kernels so each exercise stays standalone.
__global__ void reduce_atomic(const float* input, float* sum, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(sum, input[i]);
}

// Launch exactly 256 threads per block (a power of two).
__global__ void vector_sum_shared(
    const float* a, float* out, int n)
{
    __shared__ float s[256];

    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;

    s[t] = (i < n) ? a[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0; stride /= 2) {
        if (t < stride) {
            s[t] += s[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        atomicAdd(out, s[0]);
    }
}

namespace {
using Clock = std::chrono::steady_clock;
constexpr int kWarmups = 2;
constexpr int kRepeats = 7;
volatile std::uint64_t cpu_sink = 0;  // Consume each result outside the timed region.

double milliseconds(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void report(const char* label, const std::vector<double>& times) {
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    const double minimum = *std::min_element(times.begin(), times.end());
    std::cout << "  " << std::left << std::setw(23) << label
              << " mean " << mean << " ms; min " << minimum << " ms\n";
}

void validate(float actual, std::uint64_t expected) {
    if (actual != static_cast<float>(expected)) {
        throw std::runtime_error("Benchmark validation failed: got " +
                                 std::to_string(actual) + ", expected " +
                                 std::to_string(expected));
    }
}

// Only benchmark dispatch is shared; the actual CUDA kernels are above.
void launch(int variant, unsigned blocks, const float* input, float* sum, std::size_t n) {
    switch (variant) {
        case 0: reduce_atomic<<<blocks, 256>>>(input, sum, n); break;
        case 1: vector_sum_shared<<<blocks, 256>>>(input, sum, n); break;
        default: throw std::logic_error("Unknown benchmark variant");
    }
    CUDA_CHECK(cudaGetLastError());
}
}  // namespace

int main(int argc, char** argv) {
    float *d_input = nullptr, *d_sum = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    try {
        const std::size_t n = workshop::parse_size(argc, argv);
        require_device();
        if (n == 0) {
            std::cout << "EMPTY: sum=0; no kernels or meaningful timings for empty input.\n";
            return 0;
        }
        const std::vector<float> input = workshop::reduction_input(n);
        const std::uint64_t expected = workshop::reference_sum(input);
        const std::size_t bytes = n * sizeof(float);
        const unsigned blocks = workshop::blocks_for(n);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sum), sizeof(float)));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        std::cout << "N=" << n << ", float input (0/1), expected sum=" << expected
                  << "\nblocks=" << blocks << ", threads/block=256; device 0, default stream"
                  << "\nwarm-ups=" << kWarmups << ", measured runs=" << kRepeats
                  << " per method/timing\n"
                  << "All units: milliseconds; mean and minimum, no promised ranking.\n"
                  << "Kernel: CUDA events, resident input; excludes copies, allocation, and output reset.\n"
                  << "End-to-end: host wall clock; includes H2D input, output reset, kernel,\n"
                  << "  launch checks, synchronization, and D2H result; excludes allocation.\n"
                  << "Both exclude context setup, input generation, reference checks, and event creation.\n"
                  << "CPU: optimized single thread, four wide accumulators, warm host memory; sum only.\n"
                  << std::fixed << std::setprecision(6);

        for (int rep = 0; rep < kWarmups; ++rep) {
            const std::uint64_t actual = cpu_sum(input.data(), n);
            cpu_sink = actual;
            if (actual != expected) throw std::runtime_error("CPU warm-up validation failed.");
        }
        std::vector<double> cpu_times;
        for (int rep = 0; rep < kRepeats; ++rep) {
            const auto begin = Clock::now();
            const std::uint64_t actual = cpu_sum(input.data(), n);
            const auto end = Clock::now();
            cpu_sink = actual;
            if (actual != expected) throw std::runtime_error("CPU validation failed.");
            cpu_times.push_back(milliseconds(begin, end));
        }
        std::cout << "\nCPU baseline\n";
        report("CPU sum", cpu_times);

        const char* names[] = {"Atomic per element", "Shared tree per block"};
        for (int variant = 0; variant < 2; ++variant) {
            // Warm up the complete path, including copies. Each run starts at zero.
            for (int rep = 0; rep < kWarmups; ++rep) {
                CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
                launch(variant, blocks, d_input, d_sum, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                float actual = 0;
                CUDA_CHECK(cudaMemcpy(&actual, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
                validate(actual, expected);
            }

            std::vector<double> kernel_times, total_times;
            for (int rep = 0; rep < kRepeats; ++rep) {
                // Default-stream ordering puts this reset BEFORE the start event.
                CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
                CUDA_CHECK(cudaEventRecord(start));
                launch(variant, blocks, d_input, d_sum, n);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));  // Also catches execution errors.
                float elapsed = 0;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
                float actual = 0;
                CUDA_CHECK(cudaMemcpy(&actual, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
                validate(actual, expected);
                kernel_times.push_back(elapsed);
            }
            for (int rep = -kWarmups; rep < kRepeats; ++rep) {
                float actual = 0;
                const auto begin = Clock::now();
                CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
                launch(variant, blocks, d_input, d_sum, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(&actual, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
                const auto end = Clock::now();  // D2H copy has completed.
                validate(actual, expected);
                if (rep >= 0) total_times.push_back(milliseconds(begin, end));
            }
            std::cout << '\n' << names[variant] << '\n';
            report("GPU kernel (events)", kernel_times);
            report("GPU end-to-end (wall)", total_times);
        }

        CUDA_CHECK(cudaEventDestroy(stop)); stop = nullptr;
        CUDA_CHECK(cudaEventDestroy(start)); start = nullptr;
        CUDA_CHECK(cudaFree(d_sum)); d_sum = nullptr;
        CUDA_CHECK(cudaFree(d_input)); d_input = nullptr;
        std::cout << "\nPASS: every measured and warm-up result matched the CPU reference.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        cleanup_event(stop);
        cleanup_event(start);
        cleanup_device(d_sum);
        cleanup_device(d_input);
        return 1;
    }
}
