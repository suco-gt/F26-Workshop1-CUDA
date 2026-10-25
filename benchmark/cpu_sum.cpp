#include "cpu_sum.hpp"

// Separate translation unit: calls cannot be hoisted out of the timing loop by
// ordinary compilation. Do not enable LTO for this teaching benchmark.
// Four independent accumulators reduce the serial dependency chain. Each float
// is exactly 0 or 1, so converting to a wide integer preserves its value.
std::uint64_t cpu_sum(const float* input, std::size_t n) {
    std::uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += static_cast<std::uint64_t>(input[i]);
        s1 += static_cast<std::uint64_t>(input[i + 1]);
        s2 += static_cast<std::uint64_t>(input[i + 2]);
        s3 += static_cast<std::uint64_t>(input[i + 3]);
    }
    for (; i < n; ++i) s0 += static_cast<std::uint64_t>(input[i]);
    return s0 + s1 + s2 + s3;
}
