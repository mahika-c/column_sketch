#!/usr/bin/env python3
"""
Run qacs_demo and plot:
  - Time per query (ms) for Scan, StaticCS, QACS
  - Base accesses per query for Scan, StaticCS, QACS

Usage (from column_sketch dir, with qacs_demo built):

  python3 plot_qacs.py --n 10000000 --output_prefix qacs_skewed
"""

import argparse
import re
import subprocess

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def run_qacs(n: int) -> str:
    proc = subprocess.run(
        ["./qacs_demo", str(n)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )
    return proc.stdout


def parse_output(text: str):
    # Lines look like:
    # Scan     : 0.500 ms/query, 10000000.000 base-accesses/query (total_matches=...)
    pat = re.compile(
        r"^(Scan|StaticCS|QACS)\s*:\s*([0-9.]+)\s+ms/query,\s*([0-9.]+)\s+base-accesses/query"
    )
    results = {}
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            name = m.group(1)
            ms = float(m.group(2))
            base = float(m.group(3))
            results[name] = (ms, base)
    return results


def make_plots(results, prefix: str):
    if plt is None:
        print("matplotlib is not installed; cannot generate QACS plots.")
        return

    labels = ["Scan", "StaticCS", "QACS"]
    ms = [results[l][0] for l in labels]
    base = [results[l][1] for l in labels]

    x = range(len(labels))

    # Time per query
    plt.figure()
    plt.bar(x, ms, tick_label=labels)
    plt.ylabel("Time per query (ms)")
    plt.title("QACS vs Static CS vs Scan (skewed hot equality workload)")
    plt.tight_layout()
    plt.savefig(f"{prefix}_time.png", bbox_inches="tight")
    plt.close()

    # Base accesses per query
    plt.figure()
    plt.bar(x, base, tick_label=labels)
    plt.ylabel("Base accesses per query")
    plt.title("Base data touches per query")
    plt.tight_layout()
    plt.savefig(f"{prefix}_base.png", bbox_inches="tight")
    plt.close()

    print(f"Wrote {prefix}_time.png and {prefix}_base.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--output_prefix", type=str, default="qacs_skewed")
    args = ap.parse_args()

    out = run_qacs(args.n)
    print(out)
    results = parse_output(out)
    make_plots(results, args.output_prefix)


if __name__ == "__main__":
    main()


