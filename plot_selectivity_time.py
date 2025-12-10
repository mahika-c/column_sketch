#!/usr/bin/env python3
"""
Run the benchmark for a given N and generate plots of
time/query (ms) vs selectivity for each technique (scan, bwv, cs),
for both uniform and categorical datasets.

Usage (rom the column_sketch directory):

  python3 plot_selectivity_time.py --n 10000000   --prefix selectivity

This will:
  - run: ./bench all numeric N
  - parse the output
  - write PNGs:
        selectivity_uniform.png
        selectivity_categorical.png

Each new terminal session:
cd /path/to/column_sketch
source venv/bin/activate
"""

import argparse
import re
import subprocess
from collections import defaultdict

import matplotlib.pyplot as plt


def run_bench(n: int) -> str:
    """Run ./bench all numeric N and return its stdout as a string."""
    proc = subprocess.run(
        ["./bench", "all", "numeric", str(n)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,  # for compatibility with older Python (e.g. 3.6)
    )
    return proc.stdout


def parse_output(text: str):
    """
    Parse the output of run_all_numeric into a structure:
      data[dataset][tech] = list of (selectivity, secs)
    where dataset in {"uniform", "categorical"}
          tech in {"scan", "bwv", "cs"}.
    """
    data = {
        "uniform": defaultdict(list),
        "categorical": defaultdict(list),
    }

    current_dataset = None  # "uniform" or "categorical"
    current_sel = None

    sel_re = re.compile(r"-- selectivity ~= ([0-9.]+)")
    time_re = re.compile(r"([0-9.]+)\s*s,")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("=== UNIFORM"):
            current_dataset = "uniform"
            continue
        if line.startswith("=== CATEGORICAL"):
            current_dataset = "categorical"
            continue

        m_sel = sel_re.search(line)
        if m_sel:
            current_sel = float(m_sel.group(1))
            continue

        if current_dataset is None or current_sel is None:
            continue

        # Lines look like:
        #   scan uniform: 0.213539 s, ...
        #   bwv  uniform: 0.393968 s, ...
        #   cs   UNIFORM: 0.145179 s, ...
        tokens = line.split()
        if not tokens:
            continue

        tech_token = tokens[0].lower()
        if tech_token not in {"scan", "bwv", "cs"}:
            continue

        m_time = time_re.search(line)
        if not m_time:
            continue
        secs = float(m_time.group(1))

        data[current_dataset][tech_token].append((current_sel, secs))

    return data


def make_plots(data, n: int, prefix: str):
    """Generate and save plots for uniform and categorical datasets."""
    if plt is None:
        print("matplotlib is not installed; cannot generate plots.")
        return
    techniques = [("scan", "Optimized Scan"), ("bwv", "BitWeaving/V"), ("cs", "Column Sketch")]

    for dataset in ("uniform", "categorical"):
        plt.figure()
        plt.title(f"{dataset.capitalize()} numeric, n={n}")
        plt.xlabel("Selectivity")
        plt.ylabel("Time per query (ms)")

        for tech_key, tech_label in techniques:
            points = data[dataset].get(tech_key)
            if not points:
                continue
            # Sort by selectivity
            points = sorted(points, key=lambda p: p[0])
            sels = [p[0] for p in points]
            ms = [p[1] * 1000.0 for p in points]
            plt.plot(sels, ms, marker="o", label=tech_label)

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        out_name = f"{prefix}_{dataset}.png"
        plt.savefig(out_name, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1_000_000_000, help="Number of rows (N)")
    parser.add_argument(
        "--prefix",
        type=str,
        default="results",
        help="Prefix for output PNG files",
    )
    args = parser.parse_args()

    print(f"Running ./bench all numeric {args.n} ...")
    out = run_bench(args.n)
    data = parse_output(out)
    make_plots(data, args.n, args.prefix)


if __name__ == "__main__":
    main()


