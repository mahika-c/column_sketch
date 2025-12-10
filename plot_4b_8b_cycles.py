#!/usr/bin/env python3
"""
Compare cycles/tuple for 4B vs 8B uniform numeric data using perf.

For each data width (4B, 8B) and each technique (scan, BitWeaving/V, Column Sketch),
we run:

  perf stat -e cycles -x, ./bench   <method> uniform numeric  N      # 4B
  perf stat -e cycles -x, ./bench8  <method> uniform numeric8 N      # 8B

We then compute cycles/tuple = cycles / N and plot a grouped bar chart:

  x-axis: 4B, 8B
  y-axis: cycles / tuple
  3 bars per group: Optimized Scan, BitWeaving/V, Column Sketch

Usage (from column_sketch dir, on a host where perf, bench, and bench8 exist):

  python3 plot_4b_8b_cycles.py --n 10000000 --output cycles_4b_8b.png
"""

import argparse
import subprocess
from typing import Dict, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def run_perf_cycles(cmd) -> int:
    """
    Run 'perf stat -e cycles -x, <cmd...>' and return total cycles as int.
    cmd should be a list of arguments for the benchmark command.
    """
    full_cmd = ["perf", "stat", "-e", "cycles", "-x,"] + cmd

    proc = subprocess.run(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(full_cmd)}\n{proc.stderr}")

    cycles = None
    for line in proc.stderr.splitlines():
        if "cycles" in line:
            parts = line.split(",")
            if parts:
                try:
                    cycles = int(parts[0])
                    break
                except ValueError:
                    continue

    if cycles is None:
        raise RuntimeError(f"Could not parse cycles from perf output:\n{proc.stderr}")

    return cycles


def collect_cycles(n: int) -> Dict[Tuple[str, str], float]:
    """
    Collect cycles/tuple for each (width, technique) pair.
    width in {"4B", "8B"}
    technique in {"Optimized Scan", "BitWeaving/V", "Column Sketch"}
    """
    methods = [("scan", "Optimized Scan"), ("bwv", "BitWeaving/V"), ("cs", "Column Sketch")]

    result: Dict[Tuple[str, str], float] = {}

    # 4B: use ./bench (numeric)
    for method_key, label in methods:
        cmd = ["./bench", method_key, "uniform", "numeric", str(n)]
        print(f"Running perf for 4B {label} ...")
        cycles = run_perf_cycles(cmd)
        cpt = cycles / float(n)
        print(f"  cycles: {cycles}, cycles/tuple: {cpt:.4f}")
        result[("4B", label)] = cpt

    # 8B: use ./bench8 (numeric8)
    for method_key, label in methods:
        cmd = ["./bench8", method_key, "uniform", "numeric8", str(n)]
        print(f"Running perf for 8B {label} ...")
        cycles = run_perf_cycles(cmd)
        cpt = cycles / float(n)
        print(f"  cycles: {cycles}, cycles/tuple: {cpt:.4f}")
        result[("8B", label)] = cpt

    return result


def make_plot(cycles: Dict[Tuple[str, str], float], output: str) -> None:
    if plt is None:
        print("matplotlib is not installed; cannot generate 4B vs 8B cycles plot.")
        return

    widths = ["4B", "8B"]
    techniques = [("Optimized Scan", "white"), ("BitWeaving/V", "black"), ("Column Sketch", "gray")]

    x_positions = [0, 1]  # 0 -> 4B, 1 -> 8B
    bar_width = 0.25

    plt.figure()

    for idx, (label, color) in enumerate(techniques):
        xs = [x + (idx - 1) * bar_width for x in x_positions]
        ys = [cycles.get((w, label), 0.0) for w in widths]
        plt.bar(xs, ys, width=bar_width, label=label, color=color, edgecolor="black")

    plt.xticks(x_positions, widths)
    plt.xlabel("Data Width")
    plt.ylabel("Cycles / Tuple")
    plt.title("4B vs 8B uniform numeric")
    plt.legend()
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
        default="cycles_4b_8b.png",
        help="Output PNG filename",
    )
    args = parser.parse_args()

    cycles = collect_cycles(args.n)
    make_plot(cycles, args.output)


if __name__ == "__main__":
    main()


