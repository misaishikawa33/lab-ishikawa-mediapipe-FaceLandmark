#!/usr/bin/env python3
"""
pose_yolo_accuracy.csvから、マスクなしYOLOなしを基準にした姿勢差を角度ごとに可視化する。

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pose error by angle from pose_yolo_accuracy.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "test_programs" / "output" / "pose_yolo_accuracy.csv",
        help="評価結果CSVのパス。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "test_programs" / "output" / "pose_error_by_angle",
        help="差分CSVとグラフ画像の出力先。",
    )
    parser.add_argument(
        "--target-image-type",
        default="mask_surgical",
        help="比較対象にするマスクあり画像のimage_type。",
    )
    parser.add_argument(
        "--x-angle-source",
        choices=["ground_truth_yaw", "baseline_yaw", "baseline_yaw_flipped"],
        default="ground_truth_yaw",
        help="グラフ横軸に使う角度。baseline_yawはマスクなしYOLOなしの推定yaw。",
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


def read_pose_rows(csv_path: Path) -> dict[tuple[int, str, str], dict]:
    rows = {}
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not to_bool(row.get("pose_success", "")):
                continue
            image_id = int(row["image_id"])
            image_type = row["image_type"]
            condition = row["condition"]
            rows[(image_id, image_type, condition)] = row
    return rows


def row_to_rotation(row: dict) -> np.ndarray:
    return np.array(
        [
            [float(row["R_00"]), float(row["R_01"]), float(row["R_02"])],
            [float(row["R_10"]), float(row["R_11"]), float(row["R_12"])],
            [float(row["R_20"]), float(row["R_21"]), float(row["R_22"])],
        ],
        dtype=np.float64,
    )


def row_to_translation(row: dict) -> np.ndarray:
    return np.array(
        [float(row["t_x"]), float(row["t_y"]), float(row["t_z"])],
        dtype=np.float64,
    )


def safe_float(value: str, default: float = np.nan) -> float:
    if value == "" or value is None:
        return default
    return float(value)


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


def compute_errors(
        rows: dict[tuple[int, str, str], dict],
        target_image_type: str,
        x_angle_source: str) -> list[dict]:
    results = []
    image_ids = sorted({key[0] for key in rows})
    for image_id in image_ids:
        baseline = rows.get((image_id, BASELINE_IMAGE_TYPE, BASELINE_CONDITION))
        if baseline is None:
            continue

        baseline_r = row_to_rotation(baseline)
        baseline_t = row_to_translation(baseline)
        ground_truth_yaw = safe_float(baseline["ground_truth_yaw"])
        plot_yaw = get_plot_yaw(baseline, x_angle_source)

        for condition, label in TARGET_CONDITIONS:
            target = rows.get((image_id, target_image_type, condition))
            if target is None:
                continue

            target_r = row_to_rotation(target)
            target_t = row_to_translation(target)
            r_norm = float(np.linalg.norm(target_r - baseline_r, ord="fro"))
            t_norm = float(np.linalg.norm(target_t - baseline_t))

            results.append(
                {
                    "image_id": image_id,
                    "ground_truth_yaw": ground_truth_yaw,
                    "plot_yaw": plot_yaw,
                    "x_angle_source": x_angle_source,
                    "target_condition": condition,
                    "target_label": label,
                    "r_frobenius_norm": r_norm,
                    "t_l2_norm": t_norm,
                    "baseline_yaw": float(baseline["yaw"]),
                    "target_yaw": float(target["yaw"]),
                    "baseline_t_x": float(baseline["t_x"]),
                    "baseline_t_y": float(baseline["t_y"]),
                    "baseline_t_z": float(baseline["t_z"]),
                    "target_t_x": float(target["t_x"]),
                    "target_t_y": float(target["t_y"]),
                    "target_t_z": float(target["t_z"]),
                }
            )
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
        "r_frobenius_norm",
        "t_l2_norm",
        "baseline_yaw",
        "target_yaw",
        "baseline_t_x",
        "baseline_t_y",
        "baseline_t_z",
        "target_t_x",
        "target_t_y",
        "target_t_z",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_metric(
        results: list[dict],
        metric_key: str,
        ylabel: str,
        title: str,
        output_path: Path,
        x_label: str):
    plt.figure(figsize=(9, 5))
    for condition, label in TARGET_CONDITIONS:
        series = [row for row in results if row["target_condition"] == condition]
        series = sorted(series, key=lambda row: row["plot_yaw"])
        x_values = [row["plot_yaw"] for row in series]
        y_values = [row[metric_key] for row in series]
        plt.plot(x_values, y_values, marker="o", linewidth=2, label=label)

    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.title(title)
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
    results = compute_errors(rows, args.target_image_type, args.x_angle_source)
    if not results:
        raise RuntimeError("比較できる姿勢推定結果が見つかりません。")

    csv_path = args.output_dir / "pose_error_by_angle.csv"
    r_plot_path = args.output_dir / "rotation_norm_by_angle.png"
    t_plot_path = args.output_dir / "translation_norm_by_angle.png"

    write_error_csv(results, csv_path)
    plot_metric(
        results,
        "r_frobenius_norm",
        "回転行列Rのノルム差",
        "マスクなし・YOLOなし基準に対する回転姿勢の差",
        r_plot_path,
        get_x_label(args.x_angle_source),
    )
    plot_metric(
        results,
        "t_l2_norm",
        "並進ベクトルtのノルム差",
        "マスクなし・YOLOなし基準に対する位置の差",
        t_plot_path,
        get_x_label(args.x_angle_source),
    )

    print(f"saved: {csv_path}")
    print(f"saved: {r_plot_path}")
    print(f"saved: {t_plot_path}")


if __name__ == "__main__":
    main()
