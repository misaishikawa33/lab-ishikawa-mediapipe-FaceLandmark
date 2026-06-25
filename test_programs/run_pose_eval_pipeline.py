#!/usr/bin/env python3
"""
姿勢推定評価をまとめて実行するプログラム。

使い方。
1. 下の「設定」だけを必要に応じて変更する。
2. 次のコマンドを実行する。

   cd /home/misa/lab/mediapipe/FaceLandmark
   MPLCONFIGDIR=/tmp /home/misa/miniconda/envs/facelandmark310/bin/python test_programs/run_pose_eval_pipeline.py

この1回の実行で、以下を出力する。
- ground_truth_yaw基準でYOLO使用を判定した評価結果。
- initial_yawを画像反転に合わせて符号反転した角度基準でYOLO使用を判定した評価結果。
- 各評価結果に対するRとtのノルム差CSV。
- 各評価結果に対するRとtのノルム差グラフ。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


# =========================
# 設定。別データセットではここだけ変更する。
# =========================

DATASET_NAME = "0622_surgical"

MASKLESS_DIR = REPO_ROOT / "test_programs" / "input" / "0622" # マスクなし画像のディレクトリ。サブディレクトリの構成は、maskless/{dataset_name}/*.jpg とすること。
MASK_DIR = REPO_ROOT / "test_programs" / "input" / "0622" / "mask" / "images" / "surgical" # マスク画像のディレクトリ。サブディレクトリの構成は、mask/images/{dataset_name}/*.png とすること。
TEXTURE_IMAGE = REPO_ROOT / "test_programs" / "input" / "0622" / "face1.jpg" # テクスチャ画像は、masklessの画像の中から1枚選ぶ。姿勢推定のレンダリングに使用する。
MASK_IMAGE_TYPE = "mask_surgical" # マスク画像の種類を表す文字列。pose_eval_pipeline.pyの中で、マスク画像のファイル名からこの文字列を探して、マスク画像の種類を判定するために使用する。

OUTPUT_ROOT = REPO_ROOT / "test_programs" / "output" / "pose_eval_pipeline" # 出力先のルートディレクトリ。評価結果は、{OUTPUT_ROOT}/{DATASET_NAME}/{run_name}/ に出力される。

YOLO_MODEL = REPO_ROOT / "yolofolder" / "best.pt" 
YAW_THRESHOLD = 20.0

# 入力画像が左右反転している場合は、initial_yawの符号を反転して判定する。
INITIAL_YAW_SWITCH_SOURCE = "estimated_flipped"

# ノルム差グラフの横軸。0622では角度を画像から測定するため、マスクなしYOLOなしの推定yawを反転して使う。
X_ANGLE_SOURCE = "baseline_yaw_flipped"


RUN_CONFIGS = [
    {
        "name": "initial_yaw_flipped",
        "switch_yaw_source": INITIAL_YAW_SWITCH_SOURCE,
    },
    {
        "name": "tracked_yaw",
        "switch_yaw_source": "tracked_yaw",
    },
]


def run_command(command: list[str]):
    print("")
    print("run:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_evaluation(config: dict):
    run_name = config["name"]
    run_dir = OUTPUT_ROOT / DATASET_NAME / run_name
    pose_csv = run_dir / "pose_yolo_accuracy.csv"
    yolo_summary_csv = run_dir / "yolo_usage_summary.csv"
    render_dir = run_dir / "rendered_pose"
    yolo_debug_dir = run_dir / "yolo_debug"
    error_dir = run_dir / "pose_error_by_angle"

    evaluate_command = [
        sys.executable,
        str(REPO_ROOT / "test_programs" / "evaluate_pose_yolo_accuracy.py"),
        "--maskless-dir",
        str(MASKLESS_DIR),
        "--mask-dir",
        str(MASK_DIR),
        "--mask-image-type",
        MASK_IMAGE_TYPE,
        "--texture",
        str(TEXTURE_IMAGE),
        "--yolo-model",
        str(YOLO_MODEL),
        "--yaw-threshold",
        str(YAW_THRESHOLD),
        "--switch-yaw-source",
        config["switch_yaw_source"],
        "--output",
        str(pose_csv),
        "--yolo-summary-output",
        str(yolo_summary_csv),
        "--render-output-dir",
        str(render_dir),
        "--yolo-debug-output-dir",
        str(yolo_debug_dir),
    ]
    run_command(evaluate_command)

    plot_command = [
        sys.executable,
        str(REPO_ROOT / "test_programs" / "plot_pose_error_by_angle.py"),
        "--input",
        str(pose_csv),
        "--output-dir",
        str(error_dir),
        "--target-image-type",
        MASK_IMAGE_TYPE,
        "--x-angle-source",
        X_ANGLE_SOURCE,
    ]
    run_command(plot_command)

    return {
        "run_name": run_name,
        "pose_csv": pose_csv,
        "yolo_summary_csv": yolo_summary_csv,
        "render_dir": render_dir,
        "yolo_debug_dir": yolo_debug_dir,
        "error_dir": error_dir,
    }


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    results = []
    print("Pose evaluation pipeline")
    print(f"dataset: {DATASET_NAME}")
    print(f"maskless: {MASKLESS_DIR}")
    print(f"mask: {MASK_DIR}")
    print(f"texture: {TEXTURE_IMAGE}")
    print(f"output: {OUTPUT_ROOT / DATASET_NAME}")

    for config in RUN_CONFIGS:
        results.append(run_evaluation(config))

    print("")
    print("Done.")
    for result in results:
        print(f"[{result['run_name']}]")
        print(f"  pose_csv: {result['pose_csv']}")
        print(f"  yolo_summary_csv: {result['yolo_summary_csv']}")
        print(f"  rendered_pose: {result['render_dir']}")
        print(f"  yolo_debug: {result['yolo_debug_dir']}")
        print(f"  pose_error_graphs: {result['error_dir']}")


if __name__ == "__main__":
    main()
