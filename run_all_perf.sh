#!/usr/bin/env bash
# Be strict about unset vars, but don't exit on first non-zero (perf/grep may).
set -u

# Usage:
#   ./run_all_perf.sh [N]
# Default N is 1000000000 if not provided.

N=${1:-1000000000}

echo "Running perf benchmarks for N=${N}"
echo

for method in scan cs bwv; do
  for dist in uniform categorical; do
    echo "=== ${method} ${dist} numeric ${N} ==="

    # Run the benchmark under perf, capturing perf output separately
    perf stat -e cycles -x, ./bench "${method}" "${dist}" numeric "${N}" 2>perf_out.txt

    # Extract cycles from perf output (first matching line)
    cycles=$(grep 'cycles' perf_out.txt | head -n1 | awk -F',' '{print $1}')

    # If we didn't get a cycles value, show perf output so it's debuggable.
    if [ -z "${cycles}" ]; then
      echo "cycles/tuple: n/a (no cycles line from perf)"
      echo "perf output was:"
      cat perf_out.txt
      echo
    else
      # Compute and print cycles/tuple
      awk -v c="${cycles}" -v N="${N}" 'BEGIN {
        if (c > 0 && N > 0) {
          printf("cycles/tuple: %.2f\n\n", c / N);
        } else {
          print "cycles/tuple: n/a\n";
        }
      }'
    fi
  done
done

rm -f perf_out.txt


