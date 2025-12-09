// benchmark.cpp

// Compilation Instructions:
// g++ -O3 -march=native -std=c++17 -DNDEBUG benchmark.cpp -o bench

// Running Instructions: (Bash)
// ./bench scan uniform numeric 1000000000
// ./bench cs uniform numeric 1000000000
// ./bench scan categorical numeric 1000000000
// ./bench cs categorical numeric 1000000000

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using std::uint32_t;
using std::uint8_t;

// ------------------------------------------------------------
// Dataset generation
// ------------------------------------------------------------

enum class DistType { Uniform, Categorical };

struct Dataset {
    std::vector<uint32_t> base;  // base column
};

Dataset make_uniform_numeric(std::size_t n, uint32_t min_v = 0, uint32_t max_v = 1'000'000'000) {
    Dataset d;
    d.base.resize(n);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint32_t> dist(min_v, max_v);
    for (std::size_t i = 0; i < n; ++i) {
        d.base[i] = dist(rng);
    }
    return d;
}

// "Categorical" = many repetitions from a finite domain of categories
// generate integers in [0, num_categories)
Dataset make_categorical(std::size_t n, std::size_t num_categories = 10'000) {
    Dataset d;
    d.base.resize(n);
    std::mt19937_64 rng(43);
    std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(num_categories - 1));
    for (std::size_t i = 0; i < n; ++i) {
        d.base[i] = dist(rng);
    }
    return d;
}

// ------------------------------------------------------------
// Full sequential scan (baseline "FScan" without SIMD)
// SELECT count(*) WHERE col < x
// ------------------------------------------------------------

std::size_t full_scan_less(const std::vector<uint32_t>& col, uint32_t x) {
    std::size_t cnt = 0;
    const std::size_t n = col.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (col[i] < x) cnt++;
    }
    return cnt;
}

// ------------------------------------------------------------
// Simplified 1-byte Column Sketch (CS) based on paper:
// - Build phase: equi-depth buckets via sampling/sorting
// - Probe phase: scan codes, touch base only for one endpoint bucket
// ------------------------------------------------------------

struct ColumnSketch {
    // endpoints[i] = max base value that belongs to code i
    std::array<uint32_t, 256> endpoints{};
    // codes[i] = 0..255 code for base[i]
    std::vector<uint8_t> codes;
};

// Build equi-depth histogram endpoints from a sample
static void build_endpoints_from_sample(
    const std::vector<uint32_t>& sample,
    std::array<uint32_t, 256>& endpoints)
{
    std::vector<uint32_t> s = sample;
    std::sort(s.begin(), s.end());

    const std::size_t m = s.size();
    if (m == 0) {
        // degenerate
        for (int i = 0; i < 256; ++i) endpoints[i] = 0;
        return;
    }

    for (int c = 0; c < 256; ++c) {
        // position in sorted sample: (c+1)/256 quantile
        double q = static_cast<double>(c + 1) / 256.0;
        std::size_t idx = static_cast<std::size_t>(q * m);
        if (idx >= m) idx = m - 1;
        endpoints[c] = s[idx];
    }
}

// Build Column Sketch (no unique-value optimization for simplicity)
ColumnSketch build_column_sketch(const std::vector<uint32_t>& base) {
    ColumnSketch cs;
    const std::size_t n = base.size();

    // 1) Sample up to S rows (here: min(n, 200k) like in paper)
    const std::size_t S = std::min<std::size_t>(200'000, n);
    std::vector<uint32_t> sample;
    sample.reserve(S);

    std::mt19937_64 rng(44);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);
    for (std::size_t i = 0; i < S; ++i) {
        sample.push_back(base[dist(rng)]);
    }

    // 2) Compute endpoints (compression map)
    build_endpoints_from_sample(sample, cs.endpoints);

    // 3) Build sketched column: codes[i] = S(base[i])
    cs.codes.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        uint32_t v = base[i];

        // Binary search over endpoints to find smallest c with endpoints[c] >= v
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

// Helper: S(x) = code for query constant x
uint8_t cs_code_for_value(const ColumnSketch& cs, uint32_t x) {
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

// Probe: SELECT count(*) WHERE base < x using Column Sketch
std::size_t cs_scan_less(const ColumnSketch& cs,
                         const std::vector<uint32_t>& base,
                         uint32_t x)
{
    const std::size_t n = base.size();
    uint8_t sx = cs_code_for_value(cs, x);

    std::size_t cnt = 0;
    for (std::size_t i = 0; i < n; ++i) {
        uint8_t code = cs.codes[i];
        if (code < sx) {
            // definitely qualifies
            cnt++;
        } else if (code == sx) {
            // boundary bucket -> check base
            if (base[i] < x) cnt++;
        } else {
            // code > sx => definitely fails
        }
    }
    return cnt;
}

// ------------------------------------------------------------
// Timing
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
// Main benchmark driver
// ------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: "
                  << argv[0]
                  << " {scan|cs} {uniform|categorical} {numeric} N\n";
        return 1;
    }

    std::string method   = argv[1];
    std::string dist_str = argv[2];
    std::string type_str = argv[3];
    std::size_t n        = std::stoull(argv[4]);

    if (type_str != "numeric") {
        std::cerr << "Only numeric implemented in this scaffold.\n";
        return 1;
    }

    DistType dist;
    if (dist_str == "uniform") dist = DistType::Uniform;
    else if (dist_str == "categorical") dist = DistType::Categorical;
    else {
        std::cerr << "Unknown dist type.\n";
        return 1;
    }

    std::cout << "Generating dataset: n=" << n
              << ", dist=" << dist_str
              << ", method=" << method << "\n";

    Dataset d;
    if (dist == DistType::Uniform) {
        d = make_uniform_numeric(n);
    } else {
        d = make_categorical(n, 10'000);
    }

    // Choose predicate constant: for uniform, ~30% selectivity
    // for categorical, a mid category value
    uint32_t predicate_value;
    if (dist == DistType::Uniform) {
        predicate_value = 300'000'000; // desired selectivity
    } else {
        predicate_value = 5'000;       // about half the categories
    }

    if (method == "scan") {
        // warmup
        volatile std::size_t warm = full_scan_less(d.base, predicate_value);
        (void)warm;

        double secs = time_seconds([&]() {
            volatile std::size_t res = full_scan_less(d.base, predicate_value);
            (void)res;
        });

        double tuples_per_sec = static_cast<double>(n) / secs;
        std::cout << "Full scan: "
                  << std::fixed << std::setprecision(6)
                  << secs << " s, "
                  << tuples_per_sec / 1e6 << " Mtuples/s\n";
    } else if (method == "cs") {
        std::cout << "Building Column Sketch...\n";
        ColumnSketch cs = build_column_sketch(d.base);

        // warmup
        volatile std::size_t warm = cs_scan_less(cs, d.base, predicate_value);
        (void)warm;

        double secs = time_seconds([&]() {
            volatile std::size_t res = cs_scan_less(cs, d.base, predicate_value);
            (void)res;
        });

        double tuples_per_sec = static_cast<double>(n) / secs;
        std::cout << "CS scan:   "
                  << std::fixed << std::setprecision(6)
                  << secs << " s, "
                  << tuples_per_sec / 1e6 << " Mtuples/s\n";
    } else {
        std::cerr << "Unknown method: " << method << "\n";
        return 1;
    }

    return 0;
}