## 概要
本プロジェクトは、MediaPipeを用いた顔認識と3Dモデリングを組み合わせ、顔の3D再現を行うものです。
特に、マスクなどで顔の一部が隠れている状態でも、3Dモデルを用いて顔全体の再現を目指すことを目的としています。
MediaPipeで検出した顔の特徴点に顔の3Dモデルをマッピングすることで、リアルタイムに立体的な顔の再構成が可能になります。

## 推奨環境
このリポジトリでは、`main.py` と `yolofolder/chin_detection.py` を同じ Python 3.10 環境で動かすために、`facelandmark310` 環境を使用します。

環境定義:
- [environment.facelandmark310.yml](/home/misa/lab/mediapipe/FaceLandmark/environment.facelandmark310.yml)

作成手順:
```bash
conda env create -f environment.facelandmark310.yml
conda activate facelandmark310
```

## 実行方法(main)
```bash
conda activate facelandmark310
python main.py [--texture nomask.jpg] [--draw_landmark]
```

## 実行方法(create_MQO)
```bash
conda activate facelandmark310
python create_MQO.py [model/nomask.jpg]
```

## 引数
  - texture	    使用するテクスチャ画像のファイルパスを指定します（省略可）
  - use_facedetector Face Detector機能を起動時にONにします（省略可）
  - --movie :動画を入力とする。例;mqodata/input/video/mask_test4.mp4
  - --record:動画を自動保存
  - --record_format mp4:保存する動画の形式を指定(mp4,avi)
  ```bash
  python main.py --movie mqodata/input/video/mask_test4.mp4 --texture mqodata/nomasktest4.jpg --record --record_format mp4
   ```
   


## コントロールキー
 - **n**: モデル描画のON/OFF（3Dモデルの表示・非表示）
 - **p**: 対応点モードを変更（All Points → Upper Points → Selected Points）
 - **t**: ステータス表示モードの切り替え（コンパクト表示 → 詳細表示 → コンソール表示のみ）
 
## 画面表示
実行中、画面右上に現在の機能状態が表示されます:
- **[N] Model Draw**: モデル描画のON/OFF（緑=有効、灰色=無効）
- **[P] Point Mode**: 現在の対応点モード


### モデル描画制御（**N**キー）
3Dモデルの表示・非表示を制御:
- OFF時は姿勢推定のみ実行し、モデルは表示しない
- ランドマーク表示のみ確認したい場合や処理負荷軽減に有効


## 動作環境
- OS: Ubuntu 22.04 系 / X11 実行
- Python: 3.10.18
- conda env: `facelandmark310`

## 主なライブラリ
- glfw==2.10.0
- mediapipe==0.10.21
- numpy==1.26.4
- opencv-contrib-python==4.11.0.86
- PyOpenGL==3.1.10
- PyOpenGL-accelerate==3.1.10
- TkEasyGUI==1.0.41
- ultralytics==8.3.203
- torch==2.8.0
- torchvision==0.23.0

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
