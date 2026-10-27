# マスクあり・なしCGモデル作成手順

## 目的

同じ人物をマスクなし・マスクありで連続撮影し、マスクあり画像を手動で位置合わせする。
マスクなし画像から作った同一の468頂点・903面形状に2種類のテクスチャを貼り、
Blenderで同一角度の比較画像を生成する。

## 使用環境

作業ディレクトリへ移動する。

```bash
cd /home/misa/lab/mediapipe/FaceLandmark
```

撮影とMediaPipe処理には、`mp.solutions`を利用できるPython 3.10環境を使用する。

```bash
PYTHON_CG=/home/misa/miniconda/envs/facelandmark310/bin/python
```

## 1. 新規撮影・手動位置合わせ・モデル生成

`CG03`部分は撮影ごとに変更する。

```bash
MPLCONFIGDIR=/tmp/matplotlib-cg \
$PYTHON_CG test_programs/CG/capture_and_build_mask_pair.py \
  --name CG03 \
  --camera -1 \
  --width 640 \
  --height 480 \
  --alignment manual
```

`--camera -1`はカメラID 0～4を自動探索する。カメラIDが分かっている場合は、
例えば`--camera 0`を指定する。

### 撮影操作

1. マスクなしで正面を向き、`Space`を押す。
2. マスクを装着する。
3. 半透明のマスクなし画像に目・眉・輪郭を合わせ、`Space`を押す。
4. 手動位置合わせ画面で微調整する。

撮影を中止する場合は`Q`または`Esc`を押す。

### 手動位置合わせ操作

| キー | 操作 |
|---|---|
| `A` | 左へ1 px移動 |
| `D` | 右へ1 px移動 |
| `W` | 上へ1 px移動 |
| `S` | 下へ1 px移動 |
| `Q` | 0.25度回転 |
| `E` | -0.25度回転 |
| `Z` | 0.25%縮小 |
| `X` | 0.25%拡大 |
| `R` | 調整をリセット |
| `Space` / `Enter` | 確定してモデル生成へ進む |
| `Esc` | 中止 |

目、眼鏡、眉、髪の生え際、顔の外輪郭が重なるように調整する。

## 2. 出力結果の確認

`CG03`の場合、以下へ保存される。

```text
test_programs/CG/captures/CG03/
├── source/
│   ├── nomask.jpg
│   └── mask_raw.jpg
├── aligned/
│   ├── mask_aligned.png
│   ├── alignment_comparison.png
│   ├── alignment_affine.csv
│   └── alignment_quality.txt
├── landmarks/
│   ├── nomask_landmarks.csv
│   └── mask_raw_landmarks.csv
└── models/
    ├── model_nomask.mqo
    ├── model_nomask.obj
    ├── model_nomask.mtl
    ├── model_masked.mqo
    ├── model_masked.obj
    └── model_masked.mtl
```

最初に次の画像を開き、位置合わせを確認する。

```bash
xdg-open test_programs/CG/captures/CG03/aligned/alignment_comparison.png
```

手動補正値は次で確認できる。

```bash
cat test_programs/CG/captures/CG03/aligned/alignment_quality.txt
```

## 3. 撮影済み画像の位置合わせだけをやり直す

元の結果を保存する場合は、最初にバックアップする。

```bash
cp -a \
  test_programs/CG/captures/CG03 \
  test_programs/CG/captures/CG03_before_realign
```

撮影済みの`source/nomask.jpg`と`source/mask_raw.jpg`を再利用する。

```bash
MPLCONFIGDIR=/tmp/matplotlib-cg \
$PYTHON_CG test_programs/CG/capture_and_build_mask_pair.py \
  --name CG03 \
  --reuse-captured \
  --alignment manual
```

自動位置合わせを使用する場合だけ、`--alignment auto`を指定する。

## 4. マスクなし・ありのBlendファイルを作成

### マスクなし

```bash
blender --background \
  --python test_programs/CG/setup_blender_scene.py -- \
  --obj test_programs/CG/captures/CG03/models/model_nomask.obj \
  --output test_programs/CG/captures/CG03/models/face_nomask.blend \
  --width 1280 --height 720 \
  --fx 1500 --fy 1500 \
  --principal-x 640 --principal-y 360 \
  --camera-distance 700 \
  --subdivision-level 0 \
  --light-energy 3000 \
  --world-strength 1.0 \
  --exposure 1.0
```

### マスクあり

```bash
blender --background \
  --python test_programs/CG/setup_blender_scene.py -- \
  --obj test_programs/CG/captures/CG03/models/model_masked.obj \
  --output test_programs/CG/captures/CG03/models/face_masked.blend \
  --width 1280 --height 720 \
  --fx 1500 --fy 1500 \
  --principal-x 640 --principal-y 360 \
  --camera-distance 700 \
  --subdivision-level 0 \
  --light-energy 3000 \
  --world-strength 1.0 \
  --exposure 1.0
```

## 5. Blenderでモデルを確認

```bash
blender test_programs/CG/captures/CG03/models/face_nomask.blend
```

```bash
blender test_programs/CG/captures/CG03/models/face_masked.blend
```

Blender内では、`ビュー → カメラ → アクティブカメラ`を選択し、`F12`で確認する。

## 6. -45度から+45度まで5度刻みで出力

### マスクなし画像

```bash
blender --background \
  test_programs/CG/captures/CG03/models/face_nomask.blend \
  --python test_programs/CG/render_z_angle_sequence.py -- \
  --output-dir test_programs/CG/captures/CG03/output/reference \
  --start -45 --end 45 --step 5
```

### マスクあり画像

```bash
blender --background \
  test_programs/CG/captures/CG03/models/face_masked.blend \
  --python test_programs/CG/render_z_angle_sequence.py -- \
  --output-dir test_programs/CG/captures/CG03/output/masked \
  --start -45 --end 45 --step 5
```

両方で次の19枚が生成される。

```text
face01.jpg ... face19.jpg
```

出力スクリプトは既存の姿勢比較データと向きを合わせるため、公称角度とBlenderのZ回転の
符号を内部で反転する。したがって、出力は次の対応になる。

```text
face01.jpg：公称-45度（Blender Z=+45度）
face10.jpg：公称  0度（Blender Z=  0度）
face19.jpg：公称+45度（Blender Z=-45度）
```

`render_manifest.csv`には`angle_deg`（評価用の公称角度）と
`blender_z_angle_deg`（実際に設定した角度）の両方が記録される。

## 7. 姿勢推定用フォルダへコピー

```bash
mkdir -p test_programs/pose_method_comparison/input/reference/CG03
mkdir -p test_programs/pose_method_comparison/input/masked/CG03

cp test_programs/CG/captures/CG03/output/reference/face*.jpg \
  test_programs/pose_method_comparison/input/reference/CG03/

cp test_programs/CG/captures/CG03/output/masked/face*.jpg \
  test_programs/pose_method_comparison/input/masked/CG03/
```

## 8. 姿勢推定を実行

```bash
MPLCONFIGDIR=/tmp/matplotlib-cg \
$PYTHON_CG test_programs/pose_method_comparison/evaluate_pose_methods.py \
  --dataset CG03
```

結果は次へ保存される。

```text
test_programs/pose_method_comparison/output/CG03/
```

主なファイル：

```text
pose_results.csv
pose_norm_differences.csv
debug_points/
```

## 9. 角度方向の確認

評価前に`face01`、`face10`、`face19`を確認する。現在のスクリプトは既存データに合わせて
符号反転済みである。BlenderのZ角度と公称角度を同じ符号にしたい特殊な場合だけ、
レンダリングコマンドへ`--blender-angle-sign 1`を追加する。

以前の`debug_points`が残っていると、顔検出失敗時に古い画像と混同する可能性がある。
判断には必ず最新の`pose_results.csv`の`face_detected`と`error`列も使用する。

## 補足

- 手動位置合わせでは、MediaPipeランドマークを画像変換に使用しない。
- MediaPipeランドマークは、3次元形状の生成と記録には使用する。
- マスクなし・ありモデルは同じ468頂点・903面を使用し、テクスチャだけが異なる。
- `Subdivision Surface`は形状を変えるため、厳密な比較では`--subdivision-level 0`を使用する。
