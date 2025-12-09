// benchmark.cpp

// Copy the file to the remote server
// scp "/Users/mahikacalyanakoti/Downloads/College/Year4/Year4Sem1/CIS 6500/project/column_sketch/benchmark.cpp" \mahika@biglab.seas.upenn.edu:~/column_sketch/

// Compilation Instructions:
// g++ -O3 -march=native -std=c++17 -DNDEBUG -funroll-loops benchmark.cpp -o bench

// Running Instructions: (Bash)
// Small data
// ./bench all numeric 10000000

// Big data
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

Dataset make_uniform_numeric(std::size_t n, uint32_t min_v = 0, uint32_t max_v = 65535) {
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
    const uint32_t* data = col.data();
    for (std::size_t i = 0; i < n; ++i) {
        if (data[i] < x) {
            ++cnt;
        }
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

// ------------------------------------------------------------
// BitWeaving/V (vertical bit-sliced layout)
// ------------------------------------------------------------

struct BitWeavingV {
    static constexpr int BITS = 16;
    std::size_t n = 0;         // number of tuples
    std::size_t n_words = 0;   // number of 64-row groups
    // words[w][b] holds bit b (0..31) for rows [64*w, 64*w+63]
    std::vector<std::array<std::uint64_t, BITS>> words;
};

BitWeavingV build_bitweaving_v(const std::vector<uint32_t>& base) {
    BitWeavingV bw;
    bw.n = base.size();
    bw.n_words = (bw.n + 63) / 64;
    bw.words.assign(bw.n_words, {}); // zero-initialize all bit-planes

    for (std::size_t i = 0; i < bw.n; ++i) {
        uint32_t v = base[i];
        std::size_t w = i / 64;
        std::size_t bit_idx = i % 64;
        std::uint64_t mask = 1ULL << bit_idx;
        for (int b = 0; b < BitWeavingV::BITS; ++b) {
            if (v & (1u << b)) {
                bw.words[w][b] |= mask;
            }
        }
    }
    return bw;
}

// Bit-sliced comparison for col < x over 64-row groups.
std::size_t bwv_scan_less(const BitWeavingV& bw, uint32_t x) {
    if (bw.n == 0) return 0;

    const int BITS = BitWeavingV::BITS;
    std::size_t total = 0;

    std::size_t last_bits = bw.n % 64;
    std::uint64_t last_mask = last_bits ? ((1ULL << last_bits) - 1ULL) : ~0ULL;

    for (std::size_t w = 0; w < bw.n_words; ++w) {
        std::uint64_t active = (w == bw.n_words - 1) ? last_mask : ~0ULL;
        std::uint64_t lt = 0;
        std::uint64_t eq = active;

        // Process from MSB to LSB.
        for (int b = BITS - 1; b >= 0; --b) {
            std::uint64_t xi = bw.words[w][b];
            std::uint64_t cword = ((x >> b) & 1u) ? active : 0ULL;

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
// Convenience: run all configurations in one go, sweeping selectivities
// ------------------------------------------------------------

void run_all_numeric(std::size_t n) {
    // Selectivities to sweep (approximate fractions of tuples qualifying)
    const std::array<double, 6> selectivities = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};

    // Uniform dataset
    {
        Dataset d = make_uniform_numeric(n);
        const uint32_t min_v = 0;
        const uint32_t max_v = 1'000'000'000;

        std::cout << "=== UNIFORM numeric, n=" << n << " ===\n";

        for (double s : selectivities) {
            // Map selectivity s in [0,1] to a predicate in [min_v, max_v]
            uint32_t predicate_value;
            if (s <= 0.0) {
                predicate_value = min_v;
            } else if (s >= 1.0) {
                predicate_value = max_v;
            } else {
                predicate_value = static_cast<uint32_t>(s * (max_v - min_v));
            }

            std::cout << "\n-- selectivity ~= " << std::fixed << std::setprecision(2)
                      << s << " (predicate_value=" << predicate_value << ") --\n";

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

            // BitWeaving/V
            {
                BitWeavingV bw = build_bitweaving_v(d.base);

                volatile std::size_t warm = bwv_scan_less(bw, predicate_value);
                (void)warm;

                double secs = time_seconds([&]() {
                    volatile std::size_t res = bwv_scan_less(bw, predicate_value);
                    (void)res;
                });

                double tuples_per_sec = static_cast<double>(n) / secs;
                std::cout << "bwv  uniform: "
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
                std::cout << "cs   UNIFORM: "
                          << std::fixed << std::setprecision(6)
                          << secs << " s, "
                          << tuples_per_sec / 1e6 << " Mtuples/s\n";
            }
        }
    }

    std::cout << "\n";

    // Categorical dataset
    {
        const std::size_t num_categories = 10'000;
        Dataset d = make_categorical(n, num_categories);

        std::cout << "=== CATEGORICAL numeric, n=" << n << " ===\n";

        for (double s : selectivities) {
            // Map selectivity s to category predicate in [0, num_categories]
            uint32_t predicate_value;
            if (s <= 0.0) {
                predicate_value = 0;
            } else if (s >= 1.0) {
                predicate_value = static_cast<uint32_t>(num_categories);
            } else {
                predicate_value = static_cast<uint32_t>(s * num_categories);
            }

            std::cout << "\n-- selectivity ~= " << std::fixed << std::setprecision(2)
                      << s << " (predicate_value=" << predicate_value << ") --\n";

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

            // BitWeaving/V
            {
                BitWeavingV bw = build_bitweaving_v(d.base);

                volatile std::size_t warm = bwv_scan_less(bw, predicate_value);
                (void)warm;

                double secs = time_seconds([&]() {
                    volatile std::size_t res = bwv_scan_less(bw, predicate_value);
                    (void)res;
                });

                double tuples_per_sec = static_cast<double>(n) / secs;
                std::cout << "bwv  categorical: "
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
                  << " {scan|cs|bwv} {uniform|categorical} {numeric} N [predicate]\n";
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
    } else if (method == "bwv") {
        std::cout << "Building BitWeaving/V layout...\n";
        BitWeavingV bw = build_bitweaving_v(d.base);

        // warmup
        volatile std::size_t warm = bwv_scan_less(bw, predicate_value);
        (void)warm;

        double secs = time_seconds([&]() {
            volatile std::size_t res = bwv_scan_less(bw, predicate_value);
            (void)res;
        });

        double tuples_per_sec = static_cast<double>(n) / secs;
        std::cout << "BWV scan:  "
                  << std::fixed << std::setprecision(6)
                  << secs << " s, "
                  << tuples_per_sec / 1e6 << " Mtuples/s\n";
    } else {
        std::cerr << "Unknown method: " << method << "\n";
        return 1;
    }

    return 0;
}