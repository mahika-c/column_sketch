// benchmark.cpp

// Compilation Instructions:
// g++ -O3 -march=native -std=c++17 -DNDEBUG -funroll-loops benchmark.cpp -o bench

// Running Instructions: (Bash)
// ./bench all numeric 1000000000

// Running Instructions: (Bash)
// for method in scan cs; do
//   for dist in uniform categorical; do
//     echo "=== $method $dist numeric 1000000000 ==="
//     /usr/bin/time -v ./bench "$method" "$dist" numeric 1000000000
//     echo
//   done
// done

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
#ifdef __AVX2__
#include <immintrin.h>
#endif

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

std::size_t full_scan_less_scalar(const std::vector<uint32_t>& col, uint32_t x) {
    std::size_t cnt = 0;
    const std::size_t n = col.size();
    for (std::size_t i = 0; i < n; ++i) {
        cnt += (col[i] < x);
    }
    return cnt;
}

// SIMD-accelerated version (falls back to scalar if AVX2 unavailable)
std::size_t full_scan_less(const std::vector<uint32_t>& col, uint32_t x) {
#ifdef __AVX2__
    const std::size_t n = col.size();
    const uint32_t* data = col.data();
    std::size_t cnt = 0;

    const std::size_t step = 8; // 8 × 32-bit ints per AVX2 register
    const std::size_t n_vec = (n / step) * step;

    __m256i vx = _mm256_set1_epi32(static_cast<int32_t>(x));

    std::size_t i = 0;
    for (; i < n_vec; i += step) {
        __m256i v   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        // v < x  <=>  x > v
        __m256i cmp = _mm256_cmpgt_epi32(vx, v);
        int mask    = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        cnt += static_cast<std::size_t>(__builtin_popcount(mask));
    }

    // Remainder
    for (; i < n; ++i) {
        cnt += (data[i] < x);
    }
    return cnt;
#else
    return full_scan_less_scalar(col, x);
#endif
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

// Scalar probe: SELECT count(*) WHERE base < x using Column Sketch
std::size_t cs_scan_less_scalar(const ColumnSketch& cs,
                                const std::vector<uint32_t>& base,
                                uint32_t x)
{
    const std::size_t n = base.size();
    const uint8_t* codes = cs.codes.data();
    const uint32_t* b    = base.data();

    uint8_t sx = cs_code_for_value(cs, x);
    uint32_t sx32 = sx;  // promote once

    std::size_t cnt = 0;

    // Branchless loop: the idea is
    // cnt += (code < sx) + ((code == sx) & (base < x));
    for (std::size_t i = 0; i < n; ++i) {
        uint32_t c = codes[i];  // promote to avoid repeated casts

        // These booleans become 0 or 1; bitwise & keeps it branchless.
        std::size_t less_code  = (c < sx32);
        std::size_t eq_code    = (c == sx32);
        std::size_t less_base  = (b[i] < x);

        cnt += less_code + (eq_code & less_base);
    }

    return cnt;
}

// SIMD-accelerated probe that favors 1-byte codes and only touches base
// data for boundary-bucket codes.
std::size_t cs_scan_less(const ColumnSketch& cs,
                         const std::vector<uint32_t>& base,
                         uint32_t x)
{
#ifdef __AVX2__
    const std::size_t n = base.size();
    const uint8_t* codes = cs.codes.data();
    const uint32_t* b    = base.data();

    uint8_t sx = cs_code_for_value(cs, x);

    std::size_t cnt = 0;

    const std::size_t step = 32; // 32 × 1-byte codes per AVX2 register
    const std::size_t n_vec = (n / step) * step;

    __m256i v_bias  = _mm256_set1_epi8(static_cast<char>(0x80));
    __m256i v_sx    = _mm256_set1_epi8(static_cast<char>(sx));
    __m256i v_sx_s  = _mm256_xor_si256(v_sx, v_bias); // unsigned trick

    std::size_t i = 0;
    for (; i < n_vec; i += step) {
        __m256i v_codes   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + i));
        __m256i v_codes_s = _mm256_xor_si256(v_codes, v_bias);

        // codes < sx (unsigned)  via signed compare after XOR with 0x80
        __m256i lt_mask = _mm256_cmpgt_epi8(v_sx_s, v_codes_s);
        int lt_bits     = _mm256_movemask_epi8(lt_mask);
        cnt += static_cast<std::size_t>(__builtin_popcount(lt_bits));

        // codes == sx -> need to consult base
        __m256i eq_mask = _mm256_cmpeq_epi8(v_codes, v_sx);
        int eq_bits     = _mm256_movemask_epi8(eq_mask);

        while (eq_bits) {
            int bit = __builtin_ctz(eq_bits);   // index 0..31
            eq_bits &= (eq_bits - 1);           // clear lowest set bit
            std::size_t idx = static_cast<std::size_t>(bit);
            if (b[i + idx] < x) {
                ++cnt;
            }
        }
    }

    // Remainder (scalar)
    for (; i < n; ++i) {
        uint8_t c = codes[i];
        if (c < sx) {
            ++cnt;
        } else if (c == sx) {
            if (b[i] < x) ++cnt;
        }
    }

    return cnt;
#else
    return cs_scan_less_scalar(cs, base, x);
#endif
}

// ------------------------------------------------------------
// Timing
// ------------------------------------------------------------

template <typename F>
double time_seconds(F&& f, int repeats = 7) {
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
// Convenience: run all four configurations in one go
// ------------------------------------------------------------

void run_all_numeric(std::size_t n) {
    // Uniform dataset
    {
        Dataset d = make_uniform_numeric(n);
        uint32_t predicate_value = 300'000'000; // ~30% selectivity

        // Full scan
        {
            volatile std::size_t warm = full_scan_less(d.base, predicate_value);
            (void)warm;

            double secs = time_seconds([&]() {
                volatile std::size_t res = full_scan_less(d.base, predicate_value);
                (void)res;
            });

            double tuples_per_sec = static_cast<double>(n) / secs;
            std::cout << "scan uniform: "
                      << std::fixed << std::setprecision(6)
                      << secs << " s, "
                      << tuples_per_sec / 1e6 << " Mtuples/s\n";
        }

        // Column Sketch
        {
            ColumnSketch cs = build_column_sketch(d.base);

            volatile std::size_t warm = cs_scan_less(cs, d.base, predicate_value);
            (void)warm;

            double secs = time_seconds([&]() {
                volatile std::size_t res = cs_scan_less(cs, d.base, predicate_value);
                (void)res;
            });

            double tuples_per_sec = static_cast<double>(n) / secs;
            std::cout << "cs   uniform: "
                      << std::fixed << std::setprecision(6)
                      << secs << " s, "
                      << tuples_per_sec / 1e6 << " Mtuples/s\n";
        }
    }

    std::cout << "\n";

    // Categorical dataset
    {
        Dataset d = make_categorical(n, 10'000);
        uint32_t predicate_value = 5'000; // about half the categories

        // Full scan
        {
            volatile std::size_t warm = full_scan_less(d.base, predicate_value);
            (void)warm;

            double secs = time_seconds([&]() {
                volatile std::size_t res = full_scan_less(d.base, predicate_value);
                (void)res;
            });

            double tuples_per_sec = static_cast<double>(n) / secs;
            std::cout << "scan categorical: "
                      << std::fixed << std::setprecision(6)
                      << secs << " s, "
                      << tuples_per_sec / 1e6 << " Mtuples/s\n";
        }

        // Column Sketch
        {
            ColumnSketch cs = build_column_sketch(d.base);

            volatile std::size_t warm = cs_scan_less(cs, d.base, predicate_value);
            (void)warm;

            double secs = time_seconds([&]() {
                volatile std::size_t res = cs_scan_less(cs, d.base, predicate_value);
                (void)res;
            });

            double tuples_per_sec = static_cast<double>(n) / secs;
            std::cout << "cs   categorical: "
                      << std::fixed << std::setprecision(6)
                      << secs << " s, "
                      << tuples_per_sec / 1e6 << " Mtuples/s\n";
        }
    }
}

// ------------------------------------------------------------
// Main benchmark driver
// ------------------------------------------------------------

int main(int argc, char** argv) {
    // Batch mode: run all four configurations in one go.
    if (argc >= 2 && std::string(argv[1]) == "all") {
        if (argc != 4) {
            std::cerr << "Usage: "
                      << argv[0]
                      << " all numeric N\n";
            return 1;
        }

        std::string type_str = argv[2];
        if (type_str != "numeric") {
            std::cerr << "Only numeric implemented in this scaffold.\n";
            return 1;
        }

        std::size_t n = std::stoull(argv[3]);
        run_all_numeric(n);
        return 0;
    }

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

    // Optional override from the command line: argv[5] = predicate value.
    if (argc >= 6) {
        predicate_value = static_cast<uint32_t>(std::stoul(argv[5]));
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