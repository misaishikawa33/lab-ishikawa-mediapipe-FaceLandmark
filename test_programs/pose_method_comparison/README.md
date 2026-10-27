# 位置姿勢推定方式の比較

## 一括実行

データセット名を1回指定するだけで、姿勢推定と方式間比較を続けて実行できます。

```bash
cd /home/misa/lab/mediapipe/FaceLandmark
test_programs/pose_method_comparison/run_dataset.sh test2
```

入力は`input/reference/test2`と`input/masked/test2`から読み込み、結果は`output/test2`へ保存されます。`test3`を処理する場合は、最後の引数を`test3`へ変更します。

`-45`度から`+45`度まで5度刻みで撮影した、マスクなし・マスクあり各19枚の連番画像に対して位置姿勢を推定します。マスクなし画像は全対応点で推定して基準とし、マスクあり画像を次の4方式で評価します。

- `all_points`: 全体対応点
- `eye_only`: 指定した目元のみの固定対応点
- `prior_research`: 指定した先行研究の固定対応点
- `yolo_corrected`: YOLO輪郭補正後の対応点

## 入力

画像を`input/reference/`と`input/masked/`へ配置します。両方のファイル名の末尾に同じ連番が必要です。

```text
input/reference/face01.jpg  -> -45度（マスクなし）
input/reference/face02.jpg  -> -40度（マスクなし）
...
input/reference/face10.jpg  ->   0度（マスクなし）
...
input/reference/face19.jpg  -> +45度（マスクなし）

input/masked/face01.jpg     -> -45度（マスクあり）
...
input/masked/face19.jpg     -> +45度（マスクあり）
```

複数の撮影セットは、現在の`test1`のようにデータセット用サブフォルダーへ配置できます。

```text
input/
├── reference/test1/face01.jpg ... face19.jpg
└── masked/test1/face01_surgical_manual.jpg ... face19_surgical_manual.jpg
```

現在の既定データセットは`test1`です。別のセットは`--dataset データセット名`で選択できます。

番号の開始値は1以外でも構いませんが、両フォルダーに同じ番号の19枚が欠番なく必要です。標準ではマスクなしの0度画像を3D対応点生成用テクスチャとして使用します。

## 評価の実行

リポジトリのルートから実行します。

```bash
/home/misa/miniconda/envs/facelandmark310/bin/python \
  test_programs/pose_method_comparison/evaluate_pose_methods.py
```

上記を実行すると、`input/reference/test1`と`input/masked/test1`を読み込み、`output/test1`へ結果を保存します。

方式を限定する場合は次のように指定します。

```bash
/home/misa/miniconda/envs/facelandmark310/bin/python \
  test_programs/pose_method_comparison/evaluate_pose_methods.py \
  --methods all_points eye_only prior_research
```

別の入力、出力、テクスチャを使用する例です。

```bash
/home/misa/miniconda/envs/facelandmark310/bin/python \
  test_programs/pose_method_comparison/evaluate_pose_methods.py \
  --reference-input-dir /path/to/maskless_images \
  --masked-input-dir /path/to/masked_images \
  --output-dir /path/to/output \
  --texture /path/to/frontal.jpg
```

## YOLO方式

既知の撮影角度を使い、現在の`Application.py`と`YoloRinkakuCorrector.py`に合わせて次のように切り替えます。

| 撮影角度 | YOLO補正 | PnP対応点 |
|---|---|---|
| `-15 <= angle <= 15` | 左右両方 (`both`) | 目元対応点 (`datalist3`) |
| `angle < -15` | 右側 (`right`) | 右顔対応点 (`datalist1`) |
| `angle > 15` | 左側 (`left`) | 左顔対応点 (`datalist2`) |

YOLO推論に失敗した画像ではFace Meshの未補正座標を使ってPnPを実行し、CSVの`yolo_success`を`False`として記録します。
YOLO入力領域には、Face Mesh外接矩形の最大辺に対して35%の余白を上下左右へ追加します。

## 出力

```text
output/
├── pose_results.csv
├── pose_norm_differences.csv
└── debug_points/
    ├── reference/all_points/
    └── masked/
        ├── all_points/
        ├── eye_only/
        ├── prior_research/
        └── yolo_corrected/
```

`pose_results.csv`にはOpenGL座標系へ変換した回転ベクトル、OpenCVから得た変換前の回転ベクトル、並進ベクトル、回転行列、Euler角、対応点方式、YOLO成否などを共通形式で保存します。

`pose_norm_differences.csv`には、同じ写真番号のマスクなし・全対応点を基準として、マスクあり4方式との差を保存します。回転差は`||R_masked - R_reference||_F`、並進差は`||t_masked - t_reference||_2`です。

デバッグ画像では、使用した対応点を緑、YOLOで補正された対応点を赤で示します。YOLO方式ではセグメンテーション輪郭と切り出し矩形も重ねます。全点の番号も表示したい場合は`--draw-landmark-ids`を付けます。

## 方式間比較

評価後に次を実行します。

```bash
MPLCONFIGDIR=/tmp /home/misa/miniconda/envs/facelandmark310/bin/python \
  test_programs/pose_method_comparison/compare_results.py
```

標準では`all_points`を基準として、次を出力します。

```text
output/comparison/method_comparison.csv
output/comparison/rotation_difference.png
output/comparison/translation_difference.png
```

回転差グラフには回転行列間の相対回転角、並進差グラフには並進ベクトルのL2距離を使用します。基準を変更する場合は、例えば`--reference-method eye_points`を指定します。

基準CSVと比較対象CSVを分ける場合は、次のように指定します。

```bash
MPLCONFIGDIR=/tmp /home/misa/miniconda/envs/facelandmark310/bin/python \
  test_programs/pose_method_comparison/compare_results.py \
  --reference-input /path/to/reference/pose_results.csv \
  --reference-method all_points \
  --target-input /path/to/target/pose_results.csv \
  --reference-image-type reference \
  --target-image-type masked \
  --target-methods all_points eye_only prior_research yolo_corrected
```

引数を使わず、`compare_results.py`冒頭の`REFERENCE_INPUT`、`TARGET_INPUT`、`REFERENCE_METHOD`を編集して指定することもできます。両CSVは`photo_number`によって対応付けられます。
