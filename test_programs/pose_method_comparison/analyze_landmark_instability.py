#!/usr/bin/env python3
"""MediaPipe輪郭点の不安定度とYOLO補正による姿勢改善量を解析する。"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.stats import spearmanr

from evaluate_pose_methods import (
    EYE_ONLY_LANDMARKS,
    YOLO_TARGET_LANDMARKS,
    yolo_modes_from_angle,
)


HERE = Path(__file__).resolve().parent
NUMBER_RE = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="reference/masked間の輪郭点ずれとYOLO姿勢改善量を解析する。"
    )
    parser.add_argument("--first-test", type=int, default=20)
    parser.add_argument("--last-test", type=int, default=39)
    parser.add_argument("--input-root", type=Path, default=HERE / "input")
    parser.add_argument("--results-root", type=Path, default=HERE / "output")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "output" / "test20_to_test39_direct_instability_analysis",
    )
    parser.add_argument("--yaw-threshold", type=float, default=15.0)
    return parser.parse_args()


def numbered_images(path: Path) -> dict[int, Path]:
    images = {}
    for image_path in sorted(path.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        match = NUMBER_RE.search(image_path.stem)
        if match:
            images[int(match.group(1))] = image_path
    return images


def detect_landmarks(face_mesh, path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None
    return np.asarray(
        [(point.x * width, point.y * height)
         for point in result.multi_face_landmarks[0].landmark[:468]],
        dtype=np.float64,
    )


def read_method_results(path: Path) -> dict[tuple[int, str], dict]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return {
            (int(row["photo_number"]), row["target_method"]): row
            for row in csv.DictReader(csv_file)
        }


def apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return homogeneous @ affine.T


def instability_metrics(
    reference: np.ndarray,
    masked: np.ndarray,
    target_ids: set[int],
) -> dict[str, float] | None:
    stable_ids = np.asarray(sorted(set(EYE_ONLY_LANDMARKS) - target_ids), dtype=int)
    target_index = np.asarray(sorted(target_ids), dtype=int)
    direct_distances = np.linalg.norm(
        masked[target_index] - reference[target_index], axis=1)
    interocular = float(np.linalg.norm(masked[33] - masked[263]))
    if interocular <= 1e-9:
        return None

    affine, inliers = cv2.estimateAffinePartial2D(
        reference[stable_ids], masked[stable_ids], method=cv2.RANSAC,
        ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99,
    )
    if affine is None:
        return None

    aligned_reference = apply_affine(reference, affine)
    distances = np.linalg.norm(
        masked[target_index] - aligned_reference[target_index], axis=1)

    stable_residuals = np.linalg.norm(
        masked[stable_ids] - aligned_reference[stable_ids], axis=1)
    return {
        "target_point_count": len(target_index),
        "interocular_distance_px": interocular,
        "landmark_instability_mean_px": float(np.mean(direct_distances)),
        "landmark_instability_rms_px": float(np.sqrt(np.mean(direct_distances ** 2))),
        "landmark_instability_median_px": float(np.median(direct_distances)),
        "landmark_instability_max_px": float(np.max(direct_distances)),
        "landmark_instability_normalized": float(np.mean(direct_distances) / interocular),
        "aligned_landmark_instability_mean_px": float(np.mean(distances)),
        "aligned_landmark_instability_normalized": float(np.mean(distances) / interocular),
        "alignment_residual_mean_px": float(np.mean(stable_residuals)),
        "alignment_inlier_ratio": (
            float(np.mean(inliers)) if inliers is not None else float("nan")
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def yaw_group(angle: float, threshold: float) -> str:
    if angle < -threshold:
        return "negative_yaw"
    if angle > threshold:
        return "positive_yaw"
    return "front"


def correlation_rows(rows: list[dict], threshold: float) -> list[dict]:
    groups = {"all": rows}
    for name in ("negative_yaw", "front", "positive_yaw"):
        groups[name] = [
            row for row in rows
            if yaw_group(float(row["nominal_angle_deg"]), threshold) == name
        ]
    output = []
    for group_name, group in groups.items():
        x = np.asarray([row["landmark_instability_normalized"] for row in group])
        for metric in (
            "yolo_rotation_improvement_deg",
            "eye_rotation_improvement_deg",
            "yolo_vs_eye_rotation_advantage_deg",
            "yolo_translation_improvement",
            "eye_translation_improvement",
            "yolo_vs_eye_translation_advantage",
        ):
            y = np.asarray([row[metric] for row in group])
            rho, p_value = spearmanr(x, y)
            output.append({
                "yaw_group": group_name,
                "improvement_metric": metric,
                "count": len(group),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
            })
    return output


def instability_bins(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: row["landmark_instability_normalized"])
    split = np.array_split(np.asarray(ordered, dtype=object), 3)
    output = []
    for name, group_array in zip(("stable", "middle", "unstable"), split):
        group = list(group_array)
        output.append({
            "instability_group": name,
            "count": len(group),
            "instability_mean": float(np.mean([
                row["landmark_instability_normalized"] for row in group])),
            "all_points_rotation_error_mean_deg": float(np.mean([
                row["all_points_rotation_error_deg"] for row in group])),
            "yolo_rotation_error_mean_deg": float(np.mean([
                row["yolo_rotation_error_deg"] for row in group])),
            "eye_rotation_error_mean_deg": float(np.mean([
                row["eye_rotation_error_deg"] for row in group])),
            "rotation_improvement_mean_deg": float(np.mean([
                row["yolo_rotation_improvement_deg"] for row in group])),
            "yolo_rotation_improved_ratio": float(np.mean([
                row["yolo_rotation_improvement_deg"] > 0 for row in group])),
            "eye_rotation_improvement_mean_deg": float(np.mean([
                row["eye_rotation_improvement_deg"] for row in group])),
            "eye_rotation_improved_ratio": float(np.mean([
                row["eye_rotation_improvement_deg"] > 0 for row in group])),
            "yolo_rotation_better_than_eye_ratio": float(np.mean([
                row["yolo_vs_eye_rotation_advantage_deg"] > 0 for row in group])),
            "all_points_translation_error_mean": float(np.mean([
                row["all_points_translation_error"] for row in group])),
            "yolo_translation_error_mean": float(np.mean([
                row["yolo_translation_error"] for row in group])),
            "eye_translation_error_mean": float(np.mean([
                row["eye_translation_error"] for row in group])),
            "translation_improvement_mean": float(np.mean([
                row["yolo_translation_improvement"] for row in group])),
            "yolo_translation_improved_ratio": float(np.mean([
                row["yolo_translation_improvement"] > 0 for row in group])),
            "eye_translation_improvement_mean": float(np.mean([
                row["eye_translation_improvement"] for row in group])),
            "eye_translation_improved_ratio": float(np.mean([
                row["eye_translation_improvement"] > 0 for row in group])),
            "yolo_translation_better_than_eye_ratio": float(np.mean([
                row["yolo_vs_eye_translation_advantage"] > 0 for row in group])),
        })
    return output


def scatter_plot(
    rows: list[dict], metric: str, ylabel: str, output_path: Path,
) -> None:
    colors = {"negative_yaw": "tab:blue", "front": "tab:green", "positive_yaw": "tab:orange"}
    plt.figure(figsize=(8, 6))
    x_all = np.asarray([row["landmark_instability_normalized"] for row in rows])
    y_all = np.asarray([row[metric] for row in rows])
    for group_name, color in colors.items():
        group = [row for row in rows if row["yaw_group"] == group_name]
        plt.scatter(
            [row["landmark_instability_normalized"] for row in group],
            [row[metric] for row in group],
            s=26, alpha=0.65, color=color, label=group_name,
        )
    if len(x_all) >= 2:
        slope, intercept = np.polyfit(x_all, y_all, 1)
        line_x = np.linspace(float(np.min(x_all)), float(np.max(x_all)), 100)
        plt.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.5, label="linear trend")
    rho, p_value = spearmanr(x_all, y_all)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Normalized MediaPipe contour landmark instability")
    plt.ylabel(ylabel)
    plt.title(f"Spearman rho={rho:.3f}, p={p_value:.3g}, n={len(rows)}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def method_comparison_plot(
    rows: list[dict], yolo_metric: str, eye_metric: str, ylabel: str, output_path: Path,
) -> None:
    x = np.asarray([row["landmark_instability_normalized"] for row in rows])
    plt.figure(figsize=(8, 6))
    for label, metric, color in (
        ("yolo_corrected", yolo_metric, "tab:red"),
        ("eye_only", eye_metric, "tab:orange"),
    ):
        y = np.asarray([row[metric] for row in rows])
        plt.scatter(x, y, s=20, alpha=0.30, color=color, label=label)
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        rho, _ = spearmanr(x, y)
        plt.plot(
            line_x, slope * line_x + intercept, color=color, linewidth=2,
            label=f"{label} trend (rho={rho:.3f})")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Direct normalized MediaPipe contour landmark displacement")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for test_number in range(args.first_test, args.last_test + 1):
            test_name = f"test{test_number}"
            reference_images = numbered_images(args.input_root / "reference" / test_name)
            masked_images = numbered_images(args.input_root / "masked" / test_name)
            comparison = read_method_results(
                args.results_root / test_name / "comparison" / "method_comparison.csv")
            photo_numbers = sorted({number for number, _ in comparison})
            for photo_number in photo_numbers:
                all_points = comparison.get((photo_number, "all_points"))
                yolo = comparison.get((photo_number, "yolo_corrected"))
                eye = comparison.get((photo_number, "eye_only"))
                reference_path = reference_images.get(photo_number)
                masked_path = masked_images.get(photo_number)
                if not all((all_points, yolo, eye, reference_path, masked_path)):
                    failures.append({"test_name": test_name, "photo_number": photo_number, "reason": "missing_input"})
                    continue
                reference_points = detect_landmarks(face_mesh, reference_path)
                masked_points = detect_landmarks(face_mesh, masked_path)
                if reference_points is None or masked_points is None:
                    failures.append({"test_name": test_name, "photo_number": photo_number, "reason": "face_detection_failed"})
                    continue
                angle = float(all_points["nominal_angle_deg"])
                _, correction_mode = yolo_modes_from_angle(angle, args.yaw_threshold)
                metrics = instability_metrics(
                    reference_points, masked_points, YOLO_TARGET_LANDMARKS[correction_mode])
                if metrics is None:
                    failures.append({"test_name": test_name, "photo_number": photo_number, "reason": "alignment_failed"})
                    continue
                all_rotation = float(all_points["rotation_angle_error_deg"])
                yolo_rotation = float(yolo["rotation_angle_error_deg"])
                eye_rotation = float(eye["rotation_angle_error_deg"])
                all_translation = float(all_points["translation_l2_norm"])
                yolo_translation = float(yolo["translation_l2_norm"])
                eye_translation = float(eye["translation_l2_norm"])
                rows.append({
                    "test_name": test_name,
                    "photo_number": photo_number,
                    "nominal_angle_deg": angle,
                    "yaw_group": yaw_group(angle, args.yaw_threshold),
                    "correction_mode": correction_mode,
                    "reference_image": str(reference_path),
                    "masked_image": str(masked_path),
                    **metrics,
                    "all_points_rotation_error_deg": all_rotation,
                    "yolo_rotation_error_deg": yolo_rotation,
                    "eye_rotation_error_deg": eye_rotation,
                    "yolo_rotation_improvement_deg": all_rotation - yolo_rotation,
                    "eye_rotation_improvement_deg": all_rotation - eye_rotation,
                    "yolo_vs_eye_rotation_advantage_deg": eye_rotation - yolo_rotation,
                    "all_points_translation_error": all_translation,
                    "yolo_translation_error": yolo_translation,
                    "eye_translation_error": eye_translation,
                    "yolo_translation_improvement": all_translation - yolo_translation,
                    "eye_translation_improvement": all_translation - eye_translation,
                    "yolo_vs_eye_translation_advantage": eye_translation - yolo_translation,
                })

    if not rows:
        raise RuntimeError("解析可能なデータがありません。")
    write_csv(args.output_dir / "landmark_instability_analysis.csv", rows)
    correlations = correlation_rows(rows, args.yaw_threshold)
    write_csv(args.output_dir / "correlations.csv", correlations)
    bins = instability_bins(rows)
    write_csv(args.output_dir / "instability_group_summary.csv", bins)
    if failures:
        write_csv(args.output_dir / "failures.csv", failures)
    scatter_plot(
        rows, "yolo_rotation_improvement_deg", "YOLO rotation error improvement [deg]",
        args.output_dir / "instability_vs_rotation_improvement.png")
    scatter_plot(
        rows, "yolo_translation_improvement", "YOLO translation error improvement",
        args.output_dir / "instability_vs_translation_improvement.png")
    method_comparison_plot(
        rows, "yolo_rotation_improvement_deg", "eye_rotation_improvement_deg",
        "Rotation error improvement from all_points [deg]",
        args.output_dir / "instability_vs_rotation_improvement_methods.png")
    method_comparison_plot(
        rows, "yolo_translation_improvement", "eye_translation_improvement",
        "Translation error improvement from all_points",
        args.output_dir / "instability_vs_translation_improvement_methods.png")

    print(f"analyzed: {len(rows)}, failures: {len(failures)}")
    for row in correlations:
        print(
            f"{row['yaw_group']:>12} {row['improvement_metric']}: "
            f"rho={row['spearman_rho']:.4f}, p={row['p_value']:.4g}, n={row['count']}")
    for path in sorted(args.output_dir.iterdir()):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
