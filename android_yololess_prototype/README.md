# Android YOLOなしプロトタイプ構成案

このフォルダは、現在の `main` 側の実装には影響を与えず、Android StudioでYOLOなし版を作成するための構成案とコード雛形をまとめたものである。

最初にAndroidで実装する処理は以下である。

```text
1. カメラ画像を取得する。
2. MediaPipe FaceMeshまたはFace Landmarkerで顔ランドマークを検出する。
3. 検出した2Dランドマークと、事前に用意した3Dランドマークを対応付ける。
4. OpenCV solvePnPで回転Rと並進tを求める。
5. 求めたRとtを使って3Dモデルをカメラ画像上に描画する。
```

## 推奨する役割分担

Android側で毎フレーム行う処理は以下である。

```text
CameraX
  ↓
MediaPipe FaceMesh
  ↓
2Dランドマーク取得
  ↓
PnPによるR/t推定
  ↓
OpenGL ESまたはSceneViewなどで3Dモデル描画
```

YOLOはこの段階では使わない。まず、FaceMesh、PnP、3Dモデル描画だけがAndroid上で成立するか確認する。

## モデル作成について

Android上でMQOモデルを毎回生成するのは避ける方がよい。

推奨は以下である。

```text
PC側でface1.jpgから3Dモデルを作成する。
↓
Androidで扱いやすい形式に変換する。
↓
Androidアプリのassetsに入れて読み込む。
```

Androidで扱いやすい形式は以下である。

```text
glTF / GLB
OBJ
```

研究用の最初の確認では、まずOBJまたはGLBで固定モデルを読み込む構成がよい。

## assetsに入れるデータ

Androidアプリには、少なくとも以下を持たせる。

```text
assets/face_model.glb
  事前生成した顔3Dモデル。

assets/face_3d_points.csv
  PnPに使う3Dランドマーク座標。
  landmark_id,x,y,z の形式にする。
```

`face_3d_points.csv` の例。

```csv
landmark_id,x,y,z
0,12.3,4.5,-2.1
1,10.8,3.9,-1.8
33,-42.0,10.2,5.3
263,42.0,10.4,5.1
```

## このフォルダ内のコード

```text
app/src/main/java/com/example/facelandmarkar/PoseEstimator.kt
  OpenCV solvePnPでRとtを求める処理。

app/src/main/java/com/example/facelandmarkar/LandmarkRepository.kt
  assets/face_3d_points.csvを読み込む処理。

app/src/main/java/com/example/facelandmarkar/FacePosePipeline.kt
  FaceMeshの2D点と3D点を対応付け、姿勢推定を呼び出す処理。

app/src/main/java/com/example/facelandmarkar/RendererNotes.kt
  R/tを描画に渡すときの考え方。
```

## 実装順序

最初から3Dモデル描画まで完全に作るより、以下の順で確認するとよい。

```text
1. Androidでカメラプレビューを表示する。
2. FaceMeshランドマークを画面上に点で描画する。
3. assetsから3DランドマークCSVを読み込む。
4. solvePnPでR/tを求め、yaw/pitch/rollを画面に表示する。
5. 3Dモデルを固定位置で表示する。
6. R/tを使って3Dモデルを顔に重ねる。
```

## 注意点

Android版では、Pythonの `create_MQO.py` をそのまま移植する必要はない。

モデル生成はPC側で行い、Android側では生成済みモデルと3Dランドマーク座標を読み込む方が実装しやすい。

また、PnPでは初期値の影響を受けるため、固定初期値から直接 `SOLVEPNP_ITERATIVE` を行うのではなく、EPNPまたはSQPNPで初期解を求めてからITERATIVEで補正する構成が望ましい。
