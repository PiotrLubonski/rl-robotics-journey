"""Podsumowanie eksperymentu z gamma."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS = Path("runs/week03_gamma/results.csv")


def main():
    by_gamma = defaultdict(list)
    with RESULTS.open() as f:
        for row in csv.DictReader(f):
            by_gamma[float(row["gamma"])].append(float(row["mean"]))

    print(f"{'gamma':>6} | {'seedy':>5} | {'srednia':>8} | {'min':>7} | {'max':>7}")
    print("-" * 46)
    for gamma in sorted(by_gamma, reverse=True):
        scores = by_gamma[gamma]
        print(
            f"{gamma:>6} | {len(scores):>5} | {statistics.mean(scores):>8.1f} | "
            f"{min(scores):>7.1f} | {max(scores):>7.1f}"
        )


if __name__ == "__main__":
    main()
