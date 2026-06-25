#!/usr/bin/env python3
"""
pose_yolo_accuracy.csvから、マスクなしYOLOなしを基準にしたyaw/pitch/roll差を可視化する。

比較対象は以下の2条件。
- マスクあり、YOLOなし
- マスクあり、YOLOあり
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

BASELINE_IMAGE_TYPE = "maskless"
BASELINE_CONDITION = "no_yolo_eye_points"
TARGET_CONDITIONS = [
    ("no_yolo_eye_points", "マスクあり・YOLOなし"),
    ("with_yolo_angle_points", "マスクあり・YOLOあり"),
]
ANGLE_KEYS = ["yaw", "pitch", "roll"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot yaw/pitch/roll errors by angle from pose_yolo_accuracy.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="評価結果CSVのパス。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="差分CSVとグラフ画像の出力先。",
    )
    parser.add_argument(
        "--target-image-type",
        default="mask_cloth",
        help="比較対象にするマスクあり画像のimage_type。",
    )
    parser.add_argument(
        "--x-angle-source",
        choices=["ground_truth_yaw", "baseline_yaw", "baseline_yaw_flipped"],
        default="baseline_yaw_flipped",
        help="グラフ横軸に使う角度。",
    )
    return parser.parse_args()


def setup_japanese_font():
    plt.rcParams["font.family"] = [
        "Noto Sans CJK JP",
        "IPAexGothic",
        "IPAGothic",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def to_bool(value: str) -> bool:
    return str(value).lower() == "true"


def safe_float(value: str, default: float = np.nan) -> float:
    if value == "" or value is None:
        return default
    return float(value)


def read_pose_rows(csv_path: Path) -> dict[tuple[int, str, str], dict]:
    rows = {}
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not to_bool(row.get("pose_success", "")):
                continue
            rows[(int(row["image_id"]), row["image_type"], row["condition"])] = row
    return rows


def get_plot_yaw(baseline: dict, x_angle_source: str) -> float:
    if x_angle_source == "baseline_yaw":
        return float(baseline["yaw"])
    if x_angle_source == "baseline_yaw_flipped":
        return -float(baseline["yaw"])
    return safe_float(baseline["ground_truth_yaw"])


def get_x_label(x_angle_source: str) -> str:
    if x_angle_source == "baseline_yaw":
        return "マスクなし・YOLOなしで測定した顔角度 [deg]"
    if x_angle_source == "baseline_yaw_flipped":
        return "マスクなし・YOLOなしで測定した顔角度（反転補正後）[deg]"
    return "正解顔角度 [deg]"


def compute_euler_errors(
        rows: dict[tuple[int, str, str], dict],
        target_image_type: str,
        x_angle_source: str) -> list[dict]:
    results = []
    image_ids = sorted({key[0] for key in rows})
    for image_id in image_ids:
        baseline = rows.get((image_id, BASELINE_IMAGE_TYPE, BASELINE_CONDITION))
        if baseline is None:
            continue

        plot_yaw = get_plot_yaw(baseline, x_angle_source)
        ground_truth_yaw = safe_float(baseline["ground_truth_yaw"])

        for condition, label in TARGET_CONDITIONS:
            target = rows.get((image_id, target_image_type, condition))
            if target is None:
                continue

            row = {
                "image_id": image_id,
                "ground_truth_yaw": ground_truth_yaw,
                "plot_yaw": plot_yaw,
                "x_angle_source": x_angle_source,
                "target_condition": condition,
                "target_label": label,
            }
            for angle_key in ANGLE_KEYS:
                baseline_value = float(baseline[angle_key])
                target_value = float(target[angle_key])
                diff = target_value - baseline_value
                row[f"baseline_{angle_key}"] = baseline_value
                row[f"target_{angle_key}"] = target_value
                row[f"{angle_key}_diff"] = diff
                row[f"{angle_key}_abs_diff"] = abs(diff)

            results.append(row)

    return sorted(results, key=lambda row: (row["plot_yaw"], row["target_condition"]))


def write_error_csv(results: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "ground_truth_yaw",
        "plot_yaw",
        "x_angle_source",
        "target_condition",
        "target_label",
    ]
    for angle_key in ANGLE_KEYS:
        fieldnames.extend([
            f"baseline_{angle_key}",
            f"target_{angle_key}",
            f"{angle_key}_diff",
            f"{angle_key}_abs_diff",
        ])

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_angle_abs_diff(
        results: list[dict],
        angle_key: str,
        output_path: Path,
        x_label: str):
    plt.figure(figsize=(9, 5))
    for condition, label in TARGET_CONDITIONS:
        series = [row for row in results if row["target_condition"] == condition]
        series = sorted(series, key=lambda row: row["plot_yaw"])
        x_values = [row["plot_yaw"] for row in series]
        y_values = [row[f"{angle_key}_abs_diff"] for row in series]
        plt.plot(x_values, y_values, marker="o", linewidth=2, label=label)

    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel(f"{angle_key}の絶対差 [deg]")
    plt.title(f"マスクなし・YOLOなし基準に対する{angle_key}差")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    setup_japanese_font()
    rows = read_pose_rows(args.input)
    results = compute_euler_errors(rows, args.target_image_type, args.x_angle_source)
    if not results:
        raise RuntimeError("比較できるyaw/pitch/roll結果が見つかりません。")

    write_error_csv(results, args.output_dir / "euler_error_by_angle.csv")
    x_label = get_x_label(args.x_angle_source)
    for angle_key in ANGLE_KEYS:
        plot_angle_abs_diff(
            results,
            angle_key,
            args.output_dir / f"{angle_key}_abs_diff_by_angle.png",
            x_label,
        )

    print(f"saved: {args.output_dir / 'euler_error_by_angle.csv'}")
    for angle_key in ANGLE_KEYS:
        print(f"saved: {args.output_dir / f'{angle_key}_abs_diff_by_angle.png'}")


if __name__ == "__main__":
    main()
