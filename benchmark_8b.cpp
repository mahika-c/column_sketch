// benchmark_8b.cpp
//
// Simple 8-byte (uint64_t) versions of scan, Column Sketch, and BitWeaving/V
// on uniform numeric data.
//
// Example compile:
//   g++ -O3 -march=native -std=c++17 -DNDEBUG -funroll-loops benchmark_8b.cpp -o bench8
//
// Example usage:
//   ./bench8 scan uniform numeric8 10000000
//   ./bench8 cs   uniform numeric8 10000000
//   ./bench8 bwv  uniform numeric8 10000000

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using std::uint64_t;
using std::uint8_t;

// ------------------------------------------------------------
// Dataset generation (8B)
// ------------------------------------------------------------

struct Dataset64 {
    std::vector<uint64_t> base;  // base column
};

Dataset64 make_uniform_numeric64(std::size_t n,
                                 uint64_t min_v = 0,
                                 uint64_t max_v = 1'000'000'000ULL)
{
    Dataset64 d;
    d.base.resize(n);
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<uint64_t> dist(min_v, max_v);
    for (std::size_t i = 0; i < n; ++i) {
        d.base[i] = dist(rng);
    }
    return d;
}

// ------------------------------------------------------------
// Full sequential scan (8B)
// SELECT count(*) WHERE col < x
// ------------------------------------------------------------

std::size_t full_scan_less64(const std::vector<uint64_t>& col, uint64_t x) {
    std::size_t cnt = 0;
    const std::size_t n = col.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (col[i] < x) ++cnt;
    }
    return cnt;
}

// ------------------------------------------------------------
// Column Sketch (8B) - scalar version mirroring the 4B logic
// ------------------------------------------------------------

struct ColumnSketch64 {
    std::array<uint64_t, 256> endpoints{};
    std::vector<uint8_t> codes;
};

static void build_endpoints_from_sample64(
    const std::vector<uint64_t>& sample,
    std::array<uint64_t, 256>& endpoints)
{
    std::vector<uint64_t> s = sample;
    std::sort(s.begin(), s.end());

    const std::size_t m = s.size();
    if (m == 0) {
        for (int i = 0; i < 256; ++i) endpoints[i] = 0;
        return;
    }

    for (int c = 0; c < 256; ++c) {
        double q = static_cast<double>(c + 1) / 256.0;
        std::size_t idx = static_cast<std::size_t>(q * m);
        if (idx >= m) idx = m - 1;
        endpoints[c] = s[idx];
    }
}

ColumnSketch64 build_column_sketch64(const std::vector<uint64_t>& base) {
    ColumnSketch64 cs;
    const std::size_t n = base.size();

    const std::size_t S = std::min<std::size_t>(200'000, n);
    std::vector<uint64_t> sample;
    sample.reserve(S);

    std::mt19937_64 rng(124);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);
    for (std::size_t i = 0; i < S; ++i) {
        sample.push_back(base[dist(rng)]);
    }

    build_endpoints_from_sample64(sample, cs.endpoints);

    cs.codes.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        uint64_t v = base[i];

        int lo = 0, hi = 255, pos = 255;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (cs.endpoints[mid] >= v) {
                pos = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        cs.codes[i] = static_cast<uint8_t>(pos);
    }
    return cs;
}

uint8_t cs_code_for_value64(const ColumnSketch64& cs, uint64_t x) {
    int lo = 0, hi = 255, pos = 255;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cs.endpoints[mid] >= x) {
            pos = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    return static_cast<uint8_t>(pos);
}

std::size_t cs_scan_less64(const ColumnSketch64& cs,
                           const std::vector<uint64_t>& base,
                           uint64_t x)
{
    const std::size_t n = base.size();
    uint8_t sx = cs_code_for_value64(cs, x);

    std::size_t cnt = 0;
    for (std::size_t i = 0; i < n; ++i) {
        uint8_t code = cs.codes[i];
        if (code < sx) {
            ++cnt;
        } else if (code == sx) {
            if (base[i] < x) ++cnt;
        }
    }
    return cnt;
}

// ------------------------------------------------------------
// BitWeaving/V (8B) - 64-bit bit-sliced layout
// ------------------------------------------------------------

struct BitWeavingV64 {
    static constexpr int BITS = 64;
    std::size_t n = 0;
    std::size_t n_words = 0;
    std::vector<std::array<uint64_t, BITS>> words;
};

BitWeavingV64 build_bitweaving_v64(const std::vector<uint64_t>& base) {
    BitWeavingV64 bw;
    bw.n = base.size();
    bw.n_words = (bw.n + 63) / 64;
    bw.words.assign(bw.n_words, {});

    for (std::size_t i = 0; i < bw.n; ++i) {
        uint64_t v = base[i];
        std::size_t w = i / 64;
        std::size_t bit_idx = i % 64;
        std::uint64_t mask = 1ULL << bit_idx;
        for (int b = 0; b < BitWeavingV64::BITS; ++b) {
            if (v & (1ULL << b)) {
                bw.words[w][b] |= mask;
            }
        }
    }
    return bw;
}

std::size_t bwv_scan_less64(const BitWeavingV64& bw, uint64_t x) {
    if (bw.n == 0) return 0;

    const int BITS = BitWeavingV64::BITS;
    std::size_t total = 0;

    std::size_t last_bits = bw.n % 64;
    std::uint64_t last_mask = last_bits ? ((1ULL << last_bits) - 1ULL) : ~0ULL;

    for (std::size_t w = 0; w < bw.n_words; ++w) {
        std::uint64_t active = (w == bw.n_words - 1) ? last_mask : ~0ULL;
        std::uint64_t lt = 0;
        std::uint64_t eq = active;

        for (int b = BITS - 1; b >= 0; --b) {
            std::uint64_t xi = bw.words[w][b];
            std::uint64_t cword = ((x >> b) & 1ULL) ? active : 0ULL;

            std::uint64_t not_xi = ~xi;
            std::uint64_t eq_and = ~(xi ^ cword);

            lt |= (not_xi & cword & eq);
            eq &= eq_and;
        }

        std::uint64_t res = lt & active;
        total += static_cast<std::size_t>(__builtin_popcountll(res));
    }

    return total;
}

// ------------------------------------------------------------
// Timing helper
// ------------------------------------------------------------

template <typename F>
double time_seconds(F&& f, int repeats = 5) {
    using clock = std::chrono::high_resolution_clock;
    double best = 1e100;
    for (int r = 0; r < repeats; ++r) {
        auto start = clock::now();
        f();
        auto end   = clock::now();
        double secs = std::chrono::duration<double>(end - start).count();
        if (secs < best) best = secs;
    }
    return best;
}

// ------------------------------------------------------------
// Main driver (8B, uniform numeric only)
// ------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: "
                  << argv[0]
                  << " {scan|cs|bwv} uniform numeric8 N [predicate]\n";
        return 1;
    }

    std::string method   = argv[1];
    std::string dist_str = argv[2];
    std::string type_str = argv[3];
    std::size_t n        = std::stoull(argv[4]);

    if (dist_str != "uniform") {
        std::cerr << "Only uniform implemented in this 8B driver.\n";
        return 1;
    }
    if (type_str != "numeric8") {
        std::cerr << "Type must be numeric8 for this 8B driver.\n";
        return 1;
    }

    std::cout << "Generating 8B uniform dataset: n=" << n
              << ", method=" << method << "\n";

    Dataset64 d = make_uniform_numeric64(n);

    uint64_t predicate_value = 300'000'000ULL;
    if (argc >= 6) {
        predicate_value = static_cast<uint64_t>(std::stoull(argv[5]));
    }

    if (method == "scan") {
        volatile std::size_t warm = full_scan_less64(d.base, predicate_value);
        (void)warm;

        double secs = time_seconds([&]() {
            volatile std::size_t res = full_scan_less64(d.base, predicate_value);
            (void)res;
        });

        double tuples_per_sec = static_cast<double>(n) / secs;
        std::cout << "Full scan 8B: "
                  << std::fixed << std::setprecision(6)
                  << secs << " s, "
                  << tuples_per_sec / 1e6 << " Mtuples/s\n";
    } else if (method == "cs") {
        std::cout << "Building 8B Column Sketch...\n";
        ColumnSketch64 cs = build_column_sketch64(d.base);

        volatile std::size_t warm = cs_scan_less64(cs, d.base, predicate_value);
        (void)warm;

        double secs = time_seconds([&]() {
            volatile std::size_t res = cs_scan_less64(cs, d.base, predicate_value);
            (void)res;
        });

        double tuples_per_sec = static_cast<double>(n) / secs;
        std::cout << "CS scan 8B:  "
                  << std::fixed << std::setprecision(6)
                  << secs << " s, "
                  << tuples_per_sec / 1e6 << " Mtuples/s\n";
    } else if (method == "bwv") {
        std::cout << "Building 8B BitWeaving/V layout...\n";
        BitWeavingV64 bw = build_bitweaving_v64(d.base);

        volatile std::size_t warm = bwv_scan_less64(bw, predicate_value);
        (void)warm;

        double secs = time_seconds([&]() {
            volatile std::size_t res = bwv_scan_less64(bw, predicate_value);
            (void)res;
        });

        double tuples_per_sec = static_cast<double>(n) / secs;
        std::cout << "BWV scan 8B: "
                  << std::fixed << std::setprecision(6)
                  << secs << " s, "
                  << tuples_per_sec / 1e6 << " Mtuples/s\n";
    } else {
        std::cerr << "Unknown method: " << method << "\n";
        return 1;
    }

    return 0;
}


