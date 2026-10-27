#!/usr/bin/env python3
"""pose_results.csvの各方式を基準方式と比較し、CSVとグラフを生成する。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np


HERE = Path(__file__).resolve().parent

# 普段変更する設定。CLI引数を指定した場合はCLI側を優先する。
DATASET_NAME = "test1"
REFERENCE_INPUT = HERE / "output" / DATASET_NAME / "pose_results.csv"
TARGET_INPUT = HERE / "output" / DATASET_NAME / "pose_results.csv"
OUTPUT_DIR = HERE / "output" / DATASET_NAME / "comparison"
REFERENCE_METHOD = "all_points"
REFERENCE_IMAGE_TYPE = "reference"
TARGET_IMAGE_TYPE = "masked"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="位置姿勢推定方式の結果を比較する。")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="従来形式の指定。このCSVを基準側と比較側の両方に使用する。",
    )
    parser.add_argument(
        "--reference-input",
        type=Path,
        default=REFERENCE_INPUT,
        help="基準結果CSV。",
    )
    parser.add_argument(
        "--target-input",
        type=Path,
        default=TARGET_INPUT,
        help="比較対象結果CSV。",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--reference-method", default=REFERENCE_METHOD)
    parser.add_argument("--reference-image-type", default=REFERENCE_IMAGE_TYPE)
    parser.add_argument("--target-image-type", default=TARGET_IMAGE_TYPE)
    parser.add_argument(
        "--target-methods",
        nargs="+",
        default=None,
        help="比較する方式名。省略時は比較対象CSV内の方式を自動選択する。",
    )
    return parser.parse_args()


def rotation(row: dict) -> np.ndarray:
    return np.array([
        [float(row[f"R_{r}{c}"]) for c in range(3)] for r in range(3)
    ], dtype=np.float64)


def translation(row: dict) -> np.ndarray:
    return np.array([float(row[name]) for name in ("t_x", "t_y", "t_z")])


def rotation_vector(row: dict) -> np.ndarray:
    return np.array([float(row[name]) for name in ("rvec_x", "rvec_y", "rvec_z")])


def rotation_angle_error_deg(reference: np.ndarray, target: np.ndarray) -> float:
    """L2 norm of the relative rotation vector, converted to degrees."""
    relative = target @ reference.T
    relative_rvec, _ = cv2.Rodrigues(relative)
    return float(np.degrees(np.linalg.norm(relative_rvec.reshape(3), ord=2)))


def read_rows(path: Path) -> dict[tuple[int, str, str], dict]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["pose_success"].lower() != "true":
                continue
            rows[(
                int(row["photo_number"]),
                row.get("image_type", ""),
                row["method"],
            )] = row
    return rows


def compute_comparison(
        reference_rows: dict[tuple[int, str, str], dict],
        target_rows: dict[tuple[int, str, str], dict],
        reference_method: str,
        target_methods: list[str] | None = None,
        reference_image_type: str = REFERENCE_IMAGE_TYPE,
        target_image_type: str = TARGET_IMAGE_TYPE) -> list[dict]:
    results = []
    methods = target_methods or sorted({
        method for _, image_type, method in target_rows
        if image_type == target_image_type
    })
    photo_numbers = sorted({
        number for number, image_type, _ in reference_rows
        if image_type == reference_image_type
    })
    for photo_number in photo_numbers:
        reference = reference_rows.get(
            (photo_number, reference_image_type, reference_method))
        if reference is None:
            continue
        reference_r = rotation(reference)
        reference_t = translation(reference)
        reference_rvec = rotation_vector(reference)
        for method in methods:
            target = target_rows.get((photo_number, target_image_type, method))
            if target is None:
                continue
            target_t = translation(target)
            target_r = rotation(target)
            relative_rvec, _ = cv2.Rodrigues(target_r @ reference_r.T)
            rotation_l2_rad = float(np.linalg.norm(relative_rvec.reshape(3), ord=2))
            results.append({
                "photo_number": photo_number,
                "nominal_angle_deg": float(reference["nominal_angle_deg"]),
                "reference_method": reference_method,
                "target_method": method,
                "rotation_angle_error_deg": float(np.degrees(rotation_l2_rad)),
                "relative_rotation_vector_l2_norm_rad": rotation_l2_rad,
                "rotation_matrix_frobenius_norm": float(np.linalg.norm(target_r - reference_r, ord="fro")),
                "rotation_vector_l2_norm": float(np.linalg.norm(rotation_vector(target) - reference_rvec)),
                "translation_l2_norm": float(np.linalg.norm(target_t - reference_t)),
                "delta_t_x": float(target_t[0] - reference_t[0]),
                "delta_t_y": float(target_t[1] - reference_t[1]),
                "delta_t_z": float(target_t[2] - reference_t[2]),
            })
    return results


def plot(results: list[dict], key: str, ylabel: str, output_path: Path):
    plt.figure(figsize=(9, 5))
    methods = sorted({row["target_method"] for row in results})
    for method in methods:
        values = sorted(
            (row for row in results if row["target_method"] == method),
            key=lambda row: row["nominal_angle_deg"])
        plt.plot(
            [row["nominal_angle_deg"] for row in values],
            [row[key] for row in values], marker="o", label=method)
    plt.xlabel("Nominal yaw angle [deg]")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    reference_input = args.input or args.reference_input
    target_input = args.input or args.target_input
    reference_rows = read_rows(reference_input)
    target_rows = read_rows(target_input)
    results = compute_comparison(
        reference_rows,
        target_rows,
        args.reference_method,
        args.target_methods,
        args.reference_image_type,
        args.target_image_type,
    )
    if not results:
        raise RuntimeError(
            "比較可能な結果がありません。基準CSV、比較CSV、写真番号、方式名を確認してください。"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "method_comparison.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"reference input: {reference_input} ({args.reference_method})")
    print(f"target input: {target_input}")
    plot(results, "rotation_angle_error_deg", "Rotation difference [deg]", args.output_dir / "rotation_difference.png")
    plot(results, "translation_l2_norm", "Translation vector difference", args.output_dir / "translation_difference.png")
    print(f"saved: {output_csv}")
    print(f"saved: {args.output_dir / 'rotation_difference.png'}")
    print(f"saved: {args.output_dir / 'translation_difference.png'}")


if __name__ == "__main__":
    main()
