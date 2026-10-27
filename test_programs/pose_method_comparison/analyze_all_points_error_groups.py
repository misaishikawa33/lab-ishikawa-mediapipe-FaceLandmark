#!/usr/bin/env python3
"""all_pointsの姿勢誤差で分類し、YOLO補正とeye_onlyを比較する。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon


HERE = Path(__file__).resolve().parent
METRICS = {
    "rotation": ("rotation_angle_error_deg", "Rotation error [deg]"),
    "translation": ("translation_l2_norm", "Translation error"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="all_points誤差の大きさ別にYOLO補正とeye_onlyを比較する。")
    parser.add_argument("--first-test", type=int, default=20)
    parser.add_argument("--last-test", type=int, default=39)
    parser.add_argument("--results-root", type=Path, default=HERE / "output")
    parser.add_argument(
        "--output-dir", type=Path,
        default=HERE / "output" / "test20_to_test39_all_points_error_analysis")
    return parser.parse_args()


def read_samples(args: argparse.Namespace) -> list[dict]:
    samples = []
    for test_number in range(args.first_test, args.last_test + 1):
        path = (
            args.results_root / f"test{test_number}" / "comparison" /
            "method_comparison.csv")
        with path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        grouped: dict[int, dict[str, dict]] = {}
        for row in rows:
            grouped.setdefault(int(row["photo_number"]), {})[row["target_method"]] = row
        for photo_number, methods in sorted(grouped.items()):
            if not {"all_points", "eye_only", "yolo_corrected"} <= methods.keys():
                continue
            sample = {
                "test_name": f"test{test_number}",
                "photo_number": photo_number,
                "nominal_angle_deg": float(methods["all_points"]["nominal_angle_deg"]),
            }
            for metric_name, (column, _) in METRICS.items():
                for method in ("all_points", "eye_only", "yolo_corrected"):
                    sample[f"{method}_{metric_name}_error"] = float(methods[method][column])
            samples.append(sample)
    return samples


def assign_tertiles(samples: list[dict], baseline_key: str, output_key: str) -> tuple[float, float]:
    ordered = sorted(samples, key=lambda row: row[baseline_key])
    groups = np.array_split(np.asarray(ordered, dtype=object), 3)
    for name, group in zip(("good", "middle", "poor"), groups):
        for row in group:
            row[output_key] = name
    return float(groups[0][-1][baseline_key]), float(groups[1][-1][baseline_key])


def safe_wilcoxon(first: np.ndarray, second: np.ndarray) -> float:
    try:
        return float(wilcoxon(first, second).pvalue)
    except ValueError:
        return float("nan")


def summarize(samples: list[dict], metric: str) -> list[dict]:
    group_key = f"{metric}_all_points_group"
    output = []
    for group_name in ("good", "middle", "poor"):
        group = [row for row in samples if row[group_key] == group_name]
        all_error = np.asarray([row[f"all_points_{metric}_error"] for row in group])
        eye_error = np.asarray([row[f"eye_only_{metric}_error"] for row in group])
        yolo_error = np.asarray([row[f"yolo_corrected_{metric}_error"] for row in group])
        yolo_improvement = all_error - yolo_error
        eye_improvement = all_error - eye_error
        advantage = eye_error - yolo_error
        output.append({
            "metric": metric,
            "all_points_group": group_name,
            "count": len(group),
            "all_points_error_mean": float(np.mean(all_error)),
            "all_points_error_std": float(np.std(all_error)),
            "eye_only_error_mean": float(np.mean(eye_error)),
            "eye_only_error_std": float(np.std(eye_error)),
            "yolo_corrected_error_mean": float(np.mean(yolo_error)),
            "yolo_corrected_error_std": float(np.std(yolo_error)),
            "eye_improvement_mean": float(np.mean(eye_improvement)),
            "eye_improved_ratio": float(np.mean(eye_improvement > 0)),
            "yolo_improvement_mean": float(np.mean(yolo_improvement)),
            "yolo_improved_ratio": float(np.mean(yolo_improvement > 0)),
            "yolo_vs_eye_advantage_mean": float(np.mean(advantage)),
            "yolo_better_than_eye_ratio": float(np.mean(advantage > 0)),
            "yolo_vs_eye_wilcoxon_p": safe_wilcoxon(yolo_error, eye_error),
        })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_plot(summary: list[dict], ylabel: str, output_path: Path) -> None:
    labels = ["good", "middle", "poor"]
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(8, 5.5))
    for offset, method, color in (
        (-width, "all_points", "tab:blue"),
        (0.0, "eye_only", "tab:orange"),
        (width, "yolo_corrected", "tab:red"),
    ):
        means = [next(row for row in summary if row["all_points_group"] == name)[f"{method}_error_mean"] for name in labels]
        stds = [next(row for row in summary if row["all_points_group"] == name)[f"{method}_error_std"] for name in labels]
        plt.bar(x + offset, means, width, yerr=stds, capsize=3, color=color, alpha=0.85, label=method)
    plt.xticks(x, labels)
    plt.xlabel("Group classified by all_points error")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def scatter_plot(samples: list[dict], metric: str, ylabel: str, output_path: Path) -> None:
    baseline = np.asarray([row[f"all_points_{metric}_error"] for row in samples])
    plt.figure(figsize=(7.5, 6))
    for method, color in (("eye_only", "tab:orange"), ("yolo_corrected", "tab:red")):
        errors = np.asarray([row[f"{method}_{metric}_error"] for row in samples])
        plt.scatter(baseline, errors, s=22, alpha=0.4, color=color, label=method)
    limit = float(max(
        np.max(baseline),
        max(row[f"eye_only_{metric}_error"] for row in samples),
        max(row[f"yolo_corrected_{metric}_error"] for row in samples)))
    plt.plot([0, limit], [0, limit], color="black", linestyle="--", label="same as all_points")
    plt.xlabel(f"all_points {ylabel.lower()}")
    plt.ylabel(f"Comparison method {ylabel.lower()}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def improvement_scatter_plot(
    samples: list[dict], metric: str, ylabel: str, output_path: Path,
) -> None:
    baseline = np.asarray([row[f"all_points_{metric}_error"] for row in samples])
    yolo = np.asarray([row[f"yolo_corrected_{metric}_error"] for row in samples])
    improvement = baseline - yolo
    rho, p_value = spearmanr(baseline, improvement)
    slope, intercept = np.polyfit(baseline, improvement, 1)
    line_x = np.linspace(float(np.min(baseline)), float(np.max(baseline)), 100)
    plt.figure(figsize=(7.5, 6))
    plt.scatter(baseline, improvement, s=24, alpha=0.5, color="tab:red")
    plt.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.5)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(f"all_points {ylabel.lower()}")
    plt.ylabel(f"YOLO improvement in {ylabel.lower()}")
    plt.title(f"Spearman rho={rho:.3f}, p={p_value:.3g}, n={len(samples)}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_samples(args)
    thresholds = []
    all_summary = []
    correlations = []
    for metric, (_, ylabel) in METRICS.items():
        baseline_key = f"all_points_{metric}_error"
        group_key = f"{metric}_all_points_group"
        good_max, middle_max = assign_tertiles(samples, baseline_key, group_key)
        thresholds.append({
            "metric": metric,
            "good_max": good_max,
            "middle_max": middle_max,
            "poor_min_exclusive": middle_max,
        })
        metric_summary = summarize(samples, metric)
        all_summary.extend(metric_summary)
        baseline = np.asarray([row[baseline_key] for row in samples])
        advantage = np.asarray([
            row[f"eye_only_{metric}_error"] - row[f"yolo_corrected_{metric}_error"]
            for row in samples])
        rho, p_value = spearmanr(baseline, advantage)
        correlations.append({
            "metric": metric,
            "relationship": "all_points_error_vs_yolo_over_eye_advantage",
            "count": len(samples),
            "spearman_rho": float(rho),
            "p_value": float(p_value),
        })
        yolo_error = np.asarray([
            row[f"yolo_corrected_{metric}_error"] for row in samples])
        improvement = baseline - yolo_error
        improvement_rho, improvement_p = spearmanr(baseline, improvement)
        correlations.append({
            "metric": metric,
            "relationship": "all_points_error_vs_yolo_improvement",
            "count": len(samples),
            "spearman_rho": float(improvement_rho),
            "p_value": float(improvement_p),
        })
        group_plot(metric_summary, ylabel, args.output_dir / f"{metric}_error_by_all_points_group.png")
        scatter_plot(samples, metric, ylabel, args.output_dir / f"{metric}_all_points_vs_methods.png")
        improvement_scatter_plot(
            samples, metric, ylabel,
            args.output_dir / f"{metric}_all_points_error_vs_yolo_improvement.png")

    write_csv(args.output_dir / "sample_classification.csv", samples)
    write_csv(args.output_dir / "group_summary.csv", all_summary)
    write_csv(args.output_dir / "group_thresholds.csv", thresholds)
    write_csv(args.output_dir / "correlations.csv", correlations)
    print(f"analyzed: {len(samples)}")
    for row in all_summary:
        print(
            f"{row['metric']:>11} {row['all_points_group']:>6}: "
            f"all={row['all_points_error_mean']:.3f}, "
            f"eye={row['eye_only_error_mean']:.3f}, "
            f"yolo={row['yolo_corrected_error_mean']:.3f}, "
            f"yolo_win={100 * row['yolo_better_than_eye_ratio']:.1f}%, "
            f"p={row['yolo_vs_eye_wilcoxon_p']:.4g}")
    for path in sorted(args.output_dir.iterdir()):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
