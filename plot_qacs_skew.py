#!/usr/bin/env python3
"""
Sweep skew (hot probability) for QACS demo and plot:
  - Time per query (ms) vs p_hot for Scan, StaticCS, QACS
  - Base accesses per query vs p_hot for Scan, StaticCS, QACS

Usage (from column_sketch dir, with qacs_demo built):

  python3 plot_qacs_skew.py --n 10000000 --output_prefix qacs_skew
"""

import argparse
import re
import subprocess
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def run_qacs(n: int, p_hot: float) -> str:
    proc = subprocess.run(
        ["./qacs_demo", str(n), str(p_hot)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )
    return proc.stdout


def parse_output(text: str) -> Dict[str, tuple]:
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


def make_plots(skews: List[float], by_method: Dict[str, List[tuple]], prefix: str):
    if plt is None:
        print("matplotlib is not installed; cannot generate QACS skew plots.")
        return

    labels = ["Scan", "StaticCS", "QACS"]

    # Time per query vs skew
    plt.figure()
    for name in labels:
        ms_vals = [by_method[name][i][0] for i in range(len(skews))]
        plt.plot(skews, ms_vals, marker="o", label=name)
    plt.xlabel("Hot probability (p_hot)")
    plt.ylabel("Time per query (ms)")
    plt.title("QACS vs baselines under varying skew")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_time.png", bbox_inches="tight")
    plt.close()

    # Base accesses per query vs skew
    plt.figure()
    for name in labels:
        base_vals = [by_method[name][i][1] for i in range(len(skews))]
        plt.plot(skews, base_vals, marker="o", label=name)
    plt.xlabel("Hot probability (p_hot)")
    plt.ylabel("Base accesses per query")
    plt.title("Base data touches vs skew")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_base.png", bbox_inches="tight")
    plt.close()

    print(f"Wrote {prefix}_time.png and {prefix}_base.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--output_prefix", type=str, default="qacs_skew")
    args = ap.parse_args()

    # Example skew values suggested in the project notes.
    skews = [0.5, 0.7, 0.9]

    by_method: Dict[str, List[tuple]] = {"Scan": [], "StaticCS": [], "QACS": []}

    for p_hot in skews:
        print(f"Running qacs_demo with n={args.n}, p_hot={p_hot} ...")
        out = run_qacs(args.n, p_hot)
        print(out)
        res = parse_output(out)
        for name in by_method.keys():
            by_method[name].append(res[name])

    make_plots(skews, by_method, args.output_prefix)


if __name__ == "__main__":
    main()


