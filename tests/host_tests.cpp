#include "workshop.hpp"
#include "../benchmark/cpu_sum.hpp"

#include <sstream>

namespace {
unsigned checks = 0;
void check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
std::size_t parse(const std::string& text) {
    std::string program = "host-tests", arg = text;
    char* argv[] = {program.data(), arg.data()};
    return workshop::parse_size(2, argv);
}
void rejects(const std::string& text) {
    bool rejected = false;
    try { (void)parse(text); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Parser accepted invalid input: " + text);
}
}  // namespace

int main() {
    try {
        std::string program = "host-tests";
        char* argv[] = {program.data(), program.data(), program.data()};
        check(workshop::parse_size(1, argv) == 1'000'003, "Wrong default size");
        bool rejected_extra = false;
        try { (void)workshop::parse_size(3, argv); }
        catch (const std::invalid_argument&) { rejected_extra = true; }
        check(rejected_extra, "Parser accepted extra argument");
        check(parse("0") == 0 && parse("0001") == 1, "Zero/leading zero parse");
        check(parse("16777216") == workshop::kMaxN, "Maximum size parse");
        for (const char* arg : {"", "-1", "+1", "1.5", "1e3", "abc", " 1", "1 ",
                                "0x10", "16777217", "99999999999999999999999999999"}) {
            rejects(arg);
        }
        check(workshop::blocks_for(0) == 0, "Empty launch size");
        check(workshop::blocks_for(1) == 1, "One-element launch size");
        check(workshop::blocks_for(256) == 1 && workshop::blocks_for(257) == 2,
              "Rounded launch size");
        check(workshop::blocks_for(workshop::kMaxN) == 65536, "Maximum launch size");

        for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3},
                              std::size_t{31}, std::size_t{32}, std::size_t{33},
                              std::size_t{255}, std::size_t{256}, std::size_t{257},
                              std::size_t{1003}, std::size_t{1048579}, workshop::kMaxN}) {
            const auto input = workshop::reduction_input(n);
            const std::uint64_t expected = 2 * (n / 3) + (n % 3 != 0 ? 1 : 0);
            check(workshop::reference_sum(input) == expected, "Wrong 0/1 reference");
            check(cpu_sum(input.data(), n) == expected, "Wrong optimized CPU sum");
            check(static_cast<std::uint64_t>(static_cast<float>(expected)) == expected,
                  "Reference not exactly representable as float");
        }
        std::vector<float> a(257), b(257);
        workshop::fill_vectors(a, b);
        check(a[0] == -8 && b[0] == -6, "Expected negative vector inputs");
        for (std::size_t i = 0; i < a.size(); ++i) {
            check(a[i] >= -8 && a[i] <= 8 && b[i] >= -6 && b[i] <= 6,
                  "Vector bounds violated");
            const std::int64_t expected = static_cast<std::int64_t>(a[i]) +
                                          static_cast<std::int64_t>(b[i]);
            check(a[i] + b[i] == static_cast<float>(expected), "Inexact vector input");
        }
        // Exercise failure reporting without printing a misleading FAIL in a passing suite.
        std::ostringstream expected_error;
        auto* previous = std::cerr.rdbuf(expected_error.rdbuf());
        const bool rejected_result = !workshop::check_sum(0.0f, 1);
        std::cerr.rdbuf(previous);
        check(rejected_result && expected_error.str().find("FAIL") != std::string::npos,
              "Wrong result accepted or failure diagnostic missing");
        std::cout << "PASS: " << checks << " host checks (parser, launch sizes, data, CPU baseline).\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Host test failure: " << e.what() << '\n';
        return 1;
    }
}
