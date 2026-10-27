#!/usr/bin/env python3
"""複数テストの method_comparison.csv を集約し、統計とグラフを生成する。"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / "output"
METRICS = (
    "rotation_angle_error_deg",
    "translation_l2_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="方式比較結果を複数テスト間で集約する。")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--first-test", type=int, default=2)
    parser.add_argument("--last-test", type=int, default=11)
    parser.add_argument(
        "--tests",
        type=int,
        nargs="+",
        default=None,
        help="集約するテスト番号。指定時は --first-test/--last-test より優先する。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "test2_to_test11_summary",
    )
    return parser.parse_args()


def read_results(output_root: Path, test_numbers: list[int]) -> list[dict]:
    results = []
    for test_number in test_numbers:
        path = output_root / f"test{test_number}" / "comparison" / "method_comparison.csv"
        if not path.exists():
            raise FileNotFoundError(f"入力がありません: {path}")
        with path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                row = dict(row)
                row["test_name"] = f"test{test_number}"
                results.append(row)
    return results


def statistics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_by_angle(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["target_method"], float(row["nominal_angle_deg"]))].append(row)

    summary = []
    for (method, angle), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        output = {"target_method": method, "nominal_angle_deg": angle}
        for metric in METRICS:
            stats = statistics([float(row[metric]) for row in group])
            output.update({f"{metric}_{name}": value for name, value in stats.items()})
        summary.append(output)
    return summary


def summarize_overall(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["target_method"]].append(row)

    summary = []
    for method, group in sorted(grouped.items()):
        output = {"target_method": method}
        for metric in METRICS:
            stats = statistics([float(row[metric]) for row in group])
            output.update({f"{metric}_{name}": value for name, value in stats.items()})
        summary.append(output)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    summary: list[dict], metric: str, ylabel: str, title: str, output_path: Path
) -> None:
    plt.figure(figsize=(9, 5))
    methods = sorted({row["target_method"] for row in summary})
    for method in methods:
        values = [row for row in summary if row["target_method"] == method]
        angles = np.asarray([row["nominal_angle_deg"] for row in values])
        means = np.asarray([row[f"{metric}_mean"] for row in values])
        stds = np.asarray([row[f"{metric}_std"] for row in values])
        line = plt.plot(angles, means, marker="o", label=method)[0]
        plt.fill_between(
            angles,
            np.maximum(means - stds, 0.0),
            means + stds,
            color=line.get_color(),
            alpha=0.15,
        )
    plt.xlabel("Nominal yaw angle [deg]")
    plt.ylabel(ylabel)
    plt.title(f"{title}: mean +/- standard deviation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    test_numbers = args.tests or list(range(args.first_test, args.last_test + 1))
    if len(set(test_numbers)) != len(test_numbers):
        raise ValueError("テスト番号が重複しています。")
    rows = read_results(args.output_root, test_numbers)
    by_angle = summarize_by_angle(rows)
    overall = summarize_overall(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if test_numbers == list(range(test_numbers[0], test_numbers[-1] + 1)):
        title = f"test{test_numbers[0]}-test{test_numbers[-1]}"
    else:
        title = f"{len(test_numbers)} tests"

    write_csv(args.output_dir / "combined_method_comparison.csv", rows)
    write_csv(args.output_dir / "summary_by_angle.csv", by_angle)
    write_csv(args.output_dir / "summary_overall.csv", overall)
    plot(
        by_angle,
        "rotation_angle_error_deg",
        "Rotation difference [deg]",
        title,
        args.output_dir / "rotation_difference.png",
    )
    plot(
        by_angle,
        "translation_l2_norm",
        "Translation vector difference",
        title,
        args.output_dir / "translation_difference.png",
    )
    tests_text = ", ".join(f"test{number}" for number in test_numbers)
    print(f"集約対象: {tests_text} ({len(rows)} rows)")
    for path in sorted(args.output_dir.iterdir()):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
