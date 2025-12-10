#!/usr/bin/env python3
"""
Measure L1 cache misses for scan, BitWeaving/V, and Column Sketch
on a uniform numeric column using perf, and generate a bar chart
similar to Figure 8 in the Column Sketches paper.

Usage (from column_sketch dir):

  python3 plot_cache_misses.py --n 10000000 --output cache_misses_uniform.png

This runs, for each technique:

  perf stat -e L1-dcache-load-misses -x, ./bench <method> uniform numeric N

and plots L1-dcache-load-misses (in millions) as a bar chart.
"""

import argparse
import subprocess
from typing import Tuple, List

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def run_perf(method: str, n: int) -> int:
    """
    Run perf stat for a single method and return L1-dcache-load-misses
    as an integer count.
    """
    cmd = [
        "perf",
        "stat",
        "-e",
        "L1-dcache-load-misses",
        "-x,",
        "./bench",
        method,
        "uniform",
        "numeric",
        str(n),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")

    misses = None
    for line in proc.stderr.splitlines():
        # CSV format: value,unit,event,...
        if "L1-dcache-load-misses" in line:
            parts = line.split(",")
            if parts:
                try:
                    misses = int(parts[0])
                    break
                except ValueError:
                    continue

    if misses is None:
        raise RuntimeError(f"Could not parse L1-dcache-load-misses from perf output:\n{proc.stderr}")

    return misses


def make_plot(results: List[Tuple[str, int]], n: int, output: str) -> None:
    if plt is None:
        print("matplotlib is not installed; cannot generate cache-miss plot.")
        return

    labels = [label for label, _ in results]
    values = [misses / 1e6 for _, misses in results]  # millions

    x = range(len(labels))
    plt.figure()
    plt.bar(x, values, tick_label=labels)

    plt.ylabel("L1 Cache Misses (Millions)")
    plt.title(f"Uniform numeric, n={n}")

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000_000, help="Number of rows (uniform numeric)")
    parser.add_argument(
        "--output",
        type=str,
        default="cache_misses_uniform.png",
        help="Output PNG filename",
    )
    args = parser.parse_args()

    techniques = [("scan", "Optimized Scan"), ("bwv", "BitWeaving/V"), ("cs", "Column Sketch")]

    results: List[Tuple[str, int]] = []
    for method_key, label in techniques:
        print(f"Running perf for {label} ({method_key}) ...")
        misses = run_perf(method_key, args.n)
        print(f"  L1-dcache-load-misses: {misses}")
        results.append((label, misses))

    make_plot(results, args.n, args.output)


if __name__ == "__main__":
    main()


