#!/usr/bin/env python3
"""
MediaPipe Face Landmarkerモデルファイルをダウンロードするスクリプト
"""

import urllib.request
import os

def download_face_landmarker_model():
    """Face Landmarkerモデルファイルをダウンロード"""
    # 複数のURLを試行
    urls = [
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker.task/float16/1/face_landmarker.task",
        "https://github.com/google/mediapipe/raw/master/mediapipe/modules/face_landmark/face_landmark.tflite",
        "https://storage.googleapis.com/mediapipe-assets/face_landmarker.task"
    ]
    
    filename = "face_landmarker.task"
    
    for i, url in enumerate(urls):
        print(f"試行 {i+1}: {url}")
        try:
            # ダウンロード実行
            urllib.request.urlretrieve(url, filename)
            
            # ファイルサイズを確認
            size = os.path.getsize(filename)
            print(f"ダウンロード完了: {filename} ({size} bytes)")
            
            # サイズが1KB以上なら成功とみなす
            if size > 1024:
                print("モデルファイルのダウンロードが成功しました！")
                return True
            else:
                print("ファイルサイズが小さすぎます。次のURLを試行します...")
                os.remove(filename)
                
        except Exception as e:
            print(f"エラー: {e}")
            if os.path.exists(filename):
                os.remove(filename)
    
    print("すべてのURLでダウンロードに失敗しました。")
    print("手動でモデルファイルを取得する必要があります。")
    return False

if __name__ == "__main__":
    download_face_landmarker_model()
