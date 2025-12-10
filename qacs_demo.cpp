// qacs_demo.cpp
//
// Query-aware Column Sketch (QACS) demo for categorical equality queries.
// - Generates a skewed categorical column (few hot values are very frequent).
// - Compares three techniques under a skewed workload of hot equality queries:
//     1) Full scan over the base column
//     2) Static CS-like code (one catch-all code)
//     3) QACS with dedicated codes for hot values
// - Reports time/query (ms) and base-data accesses/query for each technique.
//
// Compile:
//   g++ -O3 -march=native -std=c++17 -DNDEBUG qacs_demo.cpp -o qacs_demo
//
// Run (example):
//   ./qacs_demo 10000000    // n = 10M categorical values

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using std::uint32_t;
using std::uint8_t;

// ------------------------------------------------------------
// Skewed categorical dataset
// ------------------------------------------------------------

struct DatasetCat {
    std::vector<uint32_t> base;
};

// Simple skew: with probability p_hot pick from {0,1,2,3}, else from [4, num_categories)
DatasetCat make_skewed_categorical(std::size_t n,
                                   std::size_t num_categories = 10000,
                                   double p_hot = 0.8)
{
    DatasetCat d;
    d.base.resize(n);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<uint32_t> hot_dist(0, 3); // 4 hot values: 0..3
    std::uniform_int_distribution<uint32_t> cold_dist(4, static_cast<uint32_t>(num_categories - 1));

    for (std::size_t i = 0; i < n; ++i) {
        if (prob(rng) < p_hot) {
            d.base[i] = hot_dist(rng);
        } else {
            d.base[i] = cold_dist(rng);
        }
    }
    return d;
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
// Baseline equality scan with instrumentation
// ------------------------------------------------------------

struct EqStats {
    std::size_t matches = 0;
    std::size_t base_accesses = 0;
};

EqStats full_scan_eq(const std::vector<uint32_t>& base, uint32_t x) {
    EqStats s;
    const std::size_t n = base.size();
    for (std::size_t i = 0; i < n; ++i) {
        ++s.base_accesses; // we touch every base element
        if (base[i] == x) ++s.matches;
    }
    return s;
}

// ------------------------------------------------------------
// Static CS-like categorical code
//   codes[i] = 0 for all values (we still have to check base for equality)
// ------------------------------------------------------------

struct StaticCS {
    std::vector<uint8_t> codes; // always 0 in this simple baseline
};

StaticCS build_static_cs(const std::vector<uint32_t>& base) {
    StaticCS cs;
    cs.codes.assign(base.size(), 0);
    (void)base;
    return cs;
}

EqStats static_cs_eq_scan(const StaticCS& cs,
                          const std::vector<uint32_t>& base,
                          uint32_t x)
{
    EqStats s;
    const std::size_t n = base.size();
    (void)cs; // not really used, but kept for symmetry
    for (std::size_t i = 0; i < n; ++i) {
        ++s.base_accesses;      // still check base every time
        if (base[i] == x) ++s.matches;
    }
    return s;
}

// ------------------------------------------------------------
// QACS for equality on categorical data.
//   - We predefine a small set of hot values (0..3).
//   - codes[i] = j if base[i] == hot_values[j]
//   - codes[i] = hot_values.size() for all "cold" values.
//   - For hot equality queries, we can answer purely from codes.
//   - For cold queries, we check base only where codes[i] is "cold".
// ------------------------------------------------------------

struct QACS {
    std::vector<uint8_t> codes;
    std::vector<uint32_t> hot_values;
    std::unordered_map<uint32_t, uint8_t> value_to_code;
};

QACS build_qacs(const std::vector<uint32_t>& base,
                const std::vector<uint32_t>& hot_values)
{
    QACS qs;
    qs.hot_values = hot_values;
    qs.codes.resize(base.size());

    // Map hot value -> dedicated code [0, H-1]
    for (std::size_t j = 0; j < hot_values.size(); ++j) {
        qs.value_to_code[hot_values[j]] = static_cast<uint8_t>(j);
    }
    uint8_t cold_code = static_cast<uint8_t>(hot_values.size());

    for (std::size_t i = 0; i < base.size(); ++i) {
        auto it = qs.value_to_code.find(base[i]);
        if (it != qs.value_to_code.end()) {
            qs.codes[i] = it->second;
        } else {
            qs.codes[i] = cold_code;
        }
    }
    return qs;
}

EqStats qacs_eq_scan(const QACS& qs,
                     const std::vector<uint32_t>& base,
                     uint32_t x)
{
    EqStats s;
    const std::size_t n = base.size();

    auto it = qs.value_to_code.find(x);
    uint8_t cold_code = static_cast<uint8_t>(qs.hot_values.size());
    if (it != qs.value_to_code.end()) {
        // Hot predicate: we never need base.
        uint8_t code_x = it->second;
        for (std::size_t i = 0; i < n; ++i) {
            if (qs.codes[i] == code_x) {
                ++s.matches;
                // base_accesses stays 0 for hot queries
            }
        }
    } else {
        // Cold predicate: we only need to check base when code == cold_code.
        for (std::size_t i = 0; i < n; ++i) {
            if (qs.codes[i] == cold_code) {
                ++s.base_accesses;
                if (base[i] == x) ++s.matches;
            }
        }
    }
    return s;
}

// ------------------------------------------------------------
// Experiment driver
// ------------------------------------------------------------

int main(int argc, char** argv) {
    std::size_t n = 10'000'000;
    if (argc >= 2) {
        n = static_cast<std::size_t>(std::stoull(argv[1]));
    }

    std::cout << "QACS demo: n = " << n << " (skewed categorical)\n";

    DatasetCat d = make_skewed_categorical(n);

    // Define hot values {0,1,2,3}
    // queries will be skewed to these
    std::vector<uint32_t> hot_values = {0, 1, 2, 3};

    StaticCS cs = build_static_cs(d.base);
    QACS qs = build_qacs(d.base, hot_values);

    // Workload: equality queries on hot values (skewed)
    const int num_queries = 10;
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<int> hot_idx_dist(0, static_cast<int>(hot_values.size()) - 1);

    auto run_workload_scan = [&]() {
        EqStats agg;
        for (int q = 0; q < num_queries; ++q) {
            uint32_t x = hot_values[hot_idx_dist(rng)];
            EqStats s = full_scan_eq(d.base, x);
            agg.matches += s.matches;
            agg.base_accesses += s.base_accesses;
        }
        return agg;
    };

    auto run_workload_static_cs = [&]() {
        EqStats agg;
        for (int q = 0; q < num_queries; ++q) {
            uint32_t x = hot_values[hot_idx_dist(rng)];
            EqStats s = static_cs_eq_scan(cs, d.base, x);
            agg.matches += s.matches;
            agg.base_accesses += s.base_accesses;
        }
        return agg;
    };

    auto run_workload_qacs = [&]() {
        EqStats agg;
        for (int q = 0; q < num_queries; ++q) {
            uint32_t x = hot_values[hot_idx_dist(rng)];
            EqStats s = qacs_eq_scan(qs, d.base, x);
            agg.matches += s.matches;
            agg.base_accesses += s.base_accesses;
        }
        return agg;
    };

    // Time each workload
    EqStats scan_stats;
    double scan_secs = time_seconds([&]() {
        scan_stats = run_workload_scan();
    });

    EqStats cs_stats;
    double cs_secs = time_seconds([&]() {
        cs_stats = run_workload_static_cs();
    });

    EqStats qacs_stats;
    double qacs_secs = time_seconds([&]() {
        qacs_stats = run_workload_qacs();
    });

    auto print_results = [&](const std::string& name, double secs, const EqStats& s) {
        double ms_per_query = (secs * 1000.0) / num_queries;
        double base_per_query = static_cast<double>(s.base_accesses) / num_queries;
        std::cout << name << ": "
                  << std::fixed << std::setprecision(3)
                  << ms_per_query << " ms/query, "
                  << base_per_query << " base-accesses/query"
                  << " (total_matches=" << s.matches << ")\n";
    };

    std::cout << "\nSkewed hot equality workload over values {0,1,2,3}:\n";
    print_results("Scan     ", scan_secs, scan_stats);
    print_results("StaticCS ", cs_secs, cs_stats);
    print_results("QACS     ", qacs_secs, qacs_stats);

    return 0;
}


