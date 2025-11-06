## 概要
本プロジェクトは、MediaPipeを用いた顔認識と3Dモデリングを組み合わせ、顔の3D再現を行うものです。
特に、マスクなどで顔の一部が隠れている状態でも、3Dモデルを用いて顔全体の再現を目指すことを目的としています。
MediaPipeで検出した顔の特徴点に顔の3Dモデルをマッピングすることで、リアルタイムに立体的な顔の再構成が可能になります。

## 実行方法(main)
```bash
python main.py [--texture nomask.jpg] [--draw_landmark]
python3.10 main.py
```

### 引数
  - texture	    使用するテクスチャ画像のファイルパスを指定します（省略可）
  - draw_landmark	ランドマークを描画する場合に指定します（省略可）
  - use_facedetector Face Detector機能を起動時にONにします（省略可）

### コントロールキー
 - **f**: Face Detecter機能のON/OFF（耳の位置検出を有効化）
 - **l**: Face Landmarker機能のON/OFF（高精度顔ランドマーク検出、マゼンタ色で描画）
 - **v**: Face Landmarker描画モードの切り替え（点のみ → 線とメッシュ → 点+番号 → OFF）
 - **g**: Face Landmarker PnP姿勢推定のON/OFF（Face Landmarkerのランドマークを使用したPnP姿勢推定）
 - **d**: 位置調整を有効化（ランドマーク234と454を耳の位置に移動、顔の端を固定）※Face Detecter有効時のみ
 - **a**: 自動スケール調整のON/OFF（顔のサイズを耳の距離から自動調整）※Face Detecter有効時のみ
 - **m**: FaceMesh描画のON/OFF（478個のランドマーク描画）
 - **n**: モデル描画のON/OFF（3Dモデルの表示・非表示）
 - **p**: 対応点モードを変更（All Points → Upper Points → Selected Points）
 - **c**: 姿勢比較モードのON/OFF（PnP方式とFace Landmarker方式の同時実行・比較）
 - **b**: Face Landmarker直接姿勢推定のON/OFF（Face Landmarkerの変換行列から直接カメラ姿勢を推定）
 - **t**: ステータス表示モードの切り替え（コンパクト表示 → 詳細表示 → コンソール表示のみ）
 - **1-9**: Face Landmarkerのスケール調整（手動調整、0.05～1.0倍）
 - **0**: Face Landmarkerスケールを自動調整に戻す

### 画面表示
実行中、画面右上に現在の機能状態が表示されます:
- **[M] FaceMesh Draw**: ON/OFF（緑=有効、灰色=無効）
- **[F] Face Detecter**: ON/OFF（緑=有効、灰色=無効）
- **[L] Face Landmarker**: ON/OFF（緑=有効、灰色=無効）
- **[V] FL Draw Mode**: Face Landmarker描画モード（Points/Mesh/Points+Num/OFF）
- **[G] FL PnP Pose**: Face Landmarker PnP姿勢推定のON/OFF（緑=有効、灰色=無効）
- **[D] Position Adjust**: 位置調整のON/OFF（緑=有効、灰色=無効、オレンジ=要Face Detecter）
- **[A] Auto Scale**: 自動スケールのON/OFF（緑=有効、灰色=無効、オレンジ=要Face Detecter）
- **[N] Model Draw**: モデル描画のON/OFF（緑=有効、灰色=無効）
- **[P] Point Mode**: 現在の対応点モード（All Points/Upper Points/Selected Points）
- **[C] Pose Comparison**: 姿勢比較モードのON/OFF（緑=有効、灰色=無効）
- **[B] FL Direct Pose**: Face Landmarker直接姿勢推定のON/OFF（緑=有効、灰色=無効）
- **[T] Status Display**: 表示モード（コンパクト/詳細/コンソール）
- **FL Scale**: Face Landmarkerスケール係数と調整モード（Manual/Auto）

#### ステータス表示モード
- **コンパクト**: 有効な機能のみ表示（例：ON: FaceMesh, Face Detector, FL Direct Pose）
- **詳細**: 全機能の状態を詳細表示
- **コンソール**: 画面表示なし、コンソール出力のみ

#### 姿勢比較機能
**C**キーで有効化すると、従来のPnP方式とFace Landmarker方式の両方で同時に姿勢推定を実行し、結果を比較できます:
- リアルタイムで両方式の推定結果を表示
- 比較結果は自動的にファイル保存（`output/pose_comparison_YYYYMMDD_N.txt`）
- ファイルには各方式のオイラー角、成功/失敗状態、角度差分などの詳細情報を記録

#### Face Landmarker機能
**L**キーで有効化すると、MediaPipeの高精度Face Landmarkerによる468点のランドマーク検出が利用できます:

##### Face Landmarker描画モード（**V**キー）
- **点のみ**: ランドマークを点（丸）のみで表示
- **線とメッシュ**: FaceMeshのような線とメッシュで表示、点のサイズは半分
- **点+番号**: ランドマークに20個おきに番号を表示
- **OFF**: ランドマークを非表示（機能は動作）

##### Face Landmarker PnP姿勢推定（**G**キー）
Face Landmarkerが検出したランドマークを使用してPnP問題を解決し、カメラ姿勢を推定:
- Face Landmarkerの468点ランドマークから対応点を抽出
- 従来のPnP手法（OpenCVのsolvePnP）を使用して姿勢推定
- FaceMeshの3D座標データと対応点リストを利用
- 対応点モード（**P**キー）の設定に従って使用するランドマークを選択

##### Face Landmarker直接姿勢推定（**B**キー）
Face Landmarkerから取得した変換行列を直接使用してカメラの位置・姿勢を推定:
- 従来のPnP方式を使わず、Face Landmarkerの変換行列から直接姿勢を計算
- 高精度な顔追跡が可能（Face Landmarker機能がONの場合のみ利用可能）
- スケール係数は変換行列の大きさから自動推定
- **1-9**キーで手動スケール調整、**0**キーで自動調整に戻す

#### モデル描画制御（**N**キー）
3Dモデルの表示・非表示を制御:
- OFF時は姿勢推定のみ実行し、モデルは表示しない
- ランドマーク表示のみ確認したい場合や処理負荷軽減に有効

#### 姿勢推定の優先順位
複数の姿勢推定機能が有効な場合の実行優先順位:
1. **姿勢比較モード**（**C**キー）
2. **Face Landmarker PnP姿勢推定**（**G**キー）
3. **Face Landmarker直接姿勢推定**（**B**キー）
4. **従来のPnP方式**（デフォルト）

## 実行方法(create_MQO)
```bash
python3 create_MQO.py [model/nomask.jpg]
python3.10 create_MQO.py model/nomask.jpg
```

### 引数
  - 画像の名前は適宜変更してください

## 動作環境
- OS: Microsoft Windows 11 Pro
- バージョン: 10.0.26100 (ビルド 26100)
- Python: 3.9.13

## 主なライブラリ
- glfw==2.7.0
- mediapipe==0.10.18
- numpy==1.26.4
- opencv-contrib-python==4.10.0.84
- opencv-python==4.10.0.84
- PyOpenGL==3.1.7
- PyOpenGL-accelerate==3.1.7
- PySimpleGUI==5.0.10

## 必要なモデルファイル
- **face_landmarker.task**: MediaPipe Face Landmarker用モデルファイル（プロジェクトルートに配置）
  - 自動ダウンロードスクリプト: `python3 download_model.py`

## ファイル構成

```text
│  Application.py       # アプリの大部分の処理
│  create_MQO.py        # モデル生成
│  GLWindow.py          # glウィンドウ関連の関数
│  main.py              # メインプログラム
│  PoseEstimation.py    # カメラ姿勢推定
│  USBCamera.py         # カメラ関連の処理
│
├─data                  # 推定に使用する点のデータ
│
├─mqodata
│  │  mask.jpg
│  │  nomask.jpg        # テクスチャ画像
│  │  mesh.dat          # メッシュデータ
│  │
│  ├─landmark　         # 全特徴点保存用フォルダ
│  ├─landmark3d         # 全特徴点（正規化後）保存用フォルダ
│  ├─mesh               # メッシュデータ保存用フォルダ
│  └─model              # 生成したモデル保存用フォルダ
│      
├─mqoloader             # mqoデータ読み込みプログラム群
│
├─output　              # 出力動画像保存フォルダ
│  ├─images
│  └─videos
│
├─results　             # 結果保存フォルダ
│   culc.py
│   landmark_mask.jpg　 # ランドマーク推定結果
│
└─test_programs         # 顔認識関連のテスト用プログラム群

```