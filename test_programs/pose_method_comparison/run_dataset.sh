#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "使い方: $0 DATASET_NAME" >&2
    echo "例: $0 test2" >&2
    exit 2
fi

dataset_name="$1"
if [[ ! "$dataset_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "エラー: データセット名には英数字、ピリオド、アンダースコア、ハイフンだけを使用してください。" >&2
    exit 2
fi

repo_root="/home/misa/lab/mediapipe/FaceLandmark"
python_bin="/home/misa/miniconda/envs/facelandmark310/bin/python"
program_root="$repo_root/test_programs/pose_method_comparison"
reference_dir="$program_root/input/reference/$dataset_name"
masked_dir="$program_root/input/masked/$dataset_name"
result_dir="$program_root/output/$dataset_name"

if [[ ! -d "$reference_dir" ]]; then
    echo "エラー: マスクなし画像フォルダーがありません: $reference_dir" >&2
    exit 1
fi
if [[ ! -d "$masked_dir" ]]; then
    echo "エラー: マスクあり画像フォルダーがありません: $masked_dir" >&2
    exit 1
fi
if [[ ! -x "$python_bin" ]]; then
    echo "エラー: Pythonが実行できません: $python_bin" >&2
    exit 1
fi

cd "$repo_root"
export MPLCONFIGDIR=/tmp

echo "[$dataset_name] 位置姿勢推定を開始します。"
"$python_bin" -u \
    "$program_root/evaluate_pose_methods.py" \
    --dataset "$dataset_name"

echo "[$dataset_name] 方式間比較を開始します。"
"$python_bin" \
    "$program_root/compare_results.py" \
    --input "$result_dir/pose_results.csv" \
    --output-dir "$result_dir/comparison"

echo "[$dataset_name] 完了しました。"
echo "結果: $result_dir"
