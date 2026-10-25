#pragma once

// Host-only helpers: no CUDA setup or launches are hidden here.
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace workshop {
constexpr unsigned kThreads = 256;
constexpr std::size_t kDefaultN = 1'000'003;  // Deliberately not a multiple of 256.
constexpr std::size_t kMaxN = 1u << 24;

inline std::size_t parse_size(int argc, char** argv) {
    if (argc > 2) {
        throw std::invalid_argument("Usage: " + std::string(argv[0]) + " [N]");
    }
    if (argc == 1) return kDefaultN;
    const std::string arg(argv[1]);
    if (arg.empty()) throw std::invalid_argument("N must be a decimal integer.");
    std::size_t n = 0;
    for (char ch : arg) {
        if (ch < '0' || ch > '9') {
            throw std::invalid_argument("N must contain only decimal digits (no sign).");
        }
        const unsigned digit = static_cast<unsigned>(ch - '0');
        if (n > (kMaxN - digit) / 10) {
            throw std::invalid_argument("N must be in [0, 16777216] (exact float sums).");
        }
        n = n * 10 + digit;
    }
    return n;
}

inline unsigned blocks_for(std::size_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

inline void fill_vectors(std::vector<float>& a, std::vector<float>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(static_cast<int>(i % 17) - 8);
        b[i] = static_cast<float>(static_cast<int>(i % 13) - 6);
    }
}

inline bool check_vectors(const std::vector<float>& a, const std::vector<float>& b,
                          const std::vector<float>& c) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::int64_t expected = static_cast<std::int64_t>(a[i]) +
                                      static_cast<std::int64_t>(b[i]);
        if (c[i] != static_cast<float>(expected)) {
            std::cerr << "FAIL at i=" << i << ": got " << c[i]
                      << ", expected " << expected << '\n';
            return false;
        }
    }
    std::cout << "PASS\n";
    return true;
}

inline std::vector<float> reduction_input(std::size_t n) {
    std::vector<float> input(n);
    for (std::size_t i = 0; i < n; ++i) {
        input[i] = (i % 3 == 1) ? 0.0f : 1.0f;
    }
    return input;
}

inline std::uint64_t reference_sum(const std::vector<float>& input) {
    std::uint64_t sum = 0;
    for (float value : input) sum += static_cast<std::uint64_t>(value);
    return sum;
}

inline bool check_sum(float actual, std::uint64_t expected) {
    if (actual != static_cast<float>(expected)) {
        std::cerr << std::setprecision(9) << "FAIL: sum=" << actual << ", expected=" << expected
                  << ". Check TODOs, indexing, output reset, and synchronization.\n";
        return false;
    }
    std::cout << "PASS: sum=" << expected << " (exact 0/1 input sum)\n";
    return true;
}
}  // namespace workshop
