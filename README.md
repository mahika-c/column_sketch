# column_sketch
Query-Aware Column Sketches

Mahika Calyanakoti

CIS 6500 Final Project

## Problem:

Analytical databases rely heavily on scan performance for predicates such as <, =, and BETWEEN. Existing methods (optimized scans, BitWeaving, Column Imprints) are either bandwidth-bound or limited by clustering. Column Sketches (CS), introduced in SIGMOD’18, use lossy compression to create a compact auxiliary column that speeds up predicate evaluation by reducing bytes moved and cache misses. However, CS are static and do not adapt to query patterns. This project aims to (1) reproduce the performance results from the paper and (2) extend CS with query-awareness to optimize for workload frequency.

## Approach:

### Reproduce SIGMOD’18 Results:

* Implement Column Sketches, BitWeaving/V, and optimized scans (FScan).

* Measure cycles per tuple using Linux perf stat counters on CPU-isolated runs (as done in the paper).

* Compare performance for numeric (4B, 8B) and categorical data under varying selectivities.

* Expect Column Sketch to outperform scans by ~3x–6x and BitWeaving by ~1.4x in cycles/tuple.

### Query-Aware Column Sketch (QACS):

* Adapt code assignments to frequent predicates (“hot values”) to reduce fallback to base data. ***

* Implement some hard-coded query-aware column sketches for some data and associated queries and compare it to a general column sketch for that same data.

* Can implement value heatmaps to reassign unique codes periodically.

* Evaluate QACS vs. static CS under skewed queries.

## Technologies:

C++20 with AVX2/AVX-512 SIMD, perf for cycle counts, OpenMP for parallel scans, raw binary or Arrow column format for controlled in-memory experiments.

## Expected Results:

* Target Goal 1: Recreate cycles per tuple benchmarks for CS, BitWeaving, and full sequential scan on small and large data, and uniform and categorical datasets.


* Target Goal 2: Demonstrate that query aware column sketches improves scan latency and reduces base data accesses in skewed workloads.


* Reach Goal: Prototype Hybrid BitWeaving-in-CS, performing fine-grained pruning inside the sketched column to push the memory-bound limit.

## Outcome:

The project will deliver a verified replication of SIGMOD’18 results and an adaptive sketching method showing measurable gains on query-heavy workloads.



