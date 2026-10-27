# CG顔モデルの準備

## 1. マスク領域を切り取らないMQOを生成

```bash
python3 test_programs/CG/generate_full_face_mqo.py
```

`mqodata/model/model.mqo` が生成されます。既存の468頂点、`mesh.dat` の全面メッシュ、
最新の `mqodata/landmark/landmark_*.dat` のUV座標から復元するため、MediaPipeの再実行は不要です。
このモデルは468頂点・903面で、
マスク領域を含む顔全面のモデルです。既存の `model_cut.mqo` は601面であり、
マスク領域が除外されているためCG用には使用しません。

## 2. 全面MQOをOBJへ変換

リポジトリのルートで実行します。

```bash
python3 test_programs/CG/convert_mqo_to_obj.py
```

以下が `test_programs/CG/generated/` に作られます。

- `model.obj`
- `model.mtl`
- `nomask_edge_extended.png`

既定値以外のファイルを使う場合：

```bash
python3 test_programs/CG/convert_mqo_to_obj.py \
  --input mqodata/model/model.mqo \
  --output test_programs/CG/generated/model.obj
```

## 3. 実験条件を設定したBlendファイルを生成

```bash
blender --background \
  --python test_programs/CG/setup_blender_scene.py -- \
  --obj test_programs/CG/generated/model.obj \
  --output test_programs/CG/generated/face_experiment.blend \
  --width 1280 --height 720 \
  --fx 1500 --fy 1500 --principal-x 640 --principal-y 360 \
  --camera-distance 700
```

この時点で `Shade Smooth` が自動適用されます。468頂点の位置は変更しません。

見た目をさらに滑らかに確認したい場合だけ、次を追加できます。

```bash
  --subdivision-level 1
```

ただしSubdivisionはレンダリングされる表面形状を変えるため、468個の3次元点との厳密な対応確認では
`--subdivision-level 0`（既定値）を使用してください。

明るさは次の3引数で変更できます。

```text
--light-energy 3000    正面ライトの強さ
--world-strength 1.0   背景から回り込む光
--exposure 1.0         画像全体の露出（+1で約2倍）
```

暗い場合は、まず `--exposure 1.5`、それでも暗ければ `--light-energy 5000`を試してください。

このコマンドは次を自動設定します。

- 解像度：1280 x 720
- OpenCV内部パラメータ：fx=fy=1500、cx=640、cy=360
- Blenderカメラ：sensor width=36 mm、lens=42.1875 mm
- レンズ歪み：なし（画像生成後の歪み処理を行わない）
- カメラ距離：700（モデル座標をmmとして扱う）
- 背景、Area Light、EEVEE、PNG出力
- `HeadRoot` の下に `FaceMesh` を配置
- 顔はX-Z平面で直立し、カメラを正面のY軸側へ配置（角度変更はZ軸）

## 4. Blender GUIで確認

```bash
blender test_programs/CG/generated/face_experiment.blend
```

確認する項目：

1. Outlinerに `HeadRoot`、`FaceMesh`、`ExperimentCamera`、`KeyLight` がある。
2. テンキー0でカメラ表示にし、顔が画面内にある。
3. Material Previewで顔画像が正しく貼られている。
4. テクスチャが上下反転していない。

テクスチャが上下反転した場合だけ、変換を次でやり直します。

```bash
python3 test_programs/CG/convert_mqo_to_obj.py --no-flip-v
```

顔が裏向きで表示される場合だけ、次でやり直します。

```bash
python3 test_programs/CG/convert_mqo_to_obj.py --keep-winding
```

## 固定した実験条件

| 項目 | 値 |
|---|---:|
| 画像幅 | 1280 px |
| 画像高さ | 720 px |
| fx, fy | 1500 px |
| cx, cy | 640, 360 px |
| 歪み係数 | 0 |
| カメラ距離 | 700 mm |
| yaw | -45度から+45度、5度間隔（次工程） |
| pitch | 0度 |
| roll | 0度 |

焦点距離は、sensor widthを36 mmとしたとき次で換算しています。

```text
lens_mm = fx_px * sensor_width_mm / image_width_px
        = 1500 * 36 / 1280
        = 42.1875 mm
```

`fx`、画像幅、Blenderのlensのうち一つだけをGUIから変更すると、評価コードと条件が一致しなくなります。条件を変更するときは、このセットアップスクリプトの引数からBlendファイルを再生成してください。

## 5. Z軸を5度刻みで自動レンダリング

```bash
blender --background test_programs/CG/generated/face_experiment.blend \
  --python test_programs/CG/render_z_angle_sequence.py -- \
  --output-dir test_programs/CG/output/reference/testCG01 \
  --start -45 --end 45 --step 5
```

`HeadRoot` のZ軸だけを回転し、-45度から+45度まで19枚を保存します。
既存の姿勢比較データと向きを合わせるため、公称角度に対するBlender Z回転の符号は
自動的に反転します。

```text
test_programs/CG/output/reference/testCG01/
├── face01.jpg   # -45度
├── face02.jpg   # -40度
├── ...
├── face19.jpg   # +45度
└── render_manifest.csv
```

CSVには各画像の角度、回転軸、回転行列を保存します。処理後は `HeadRoot` を実行前の角度へ戻します。
