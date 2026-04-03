#!/usr/bin/env python3
"""
マスク特徴点検出プログラム
マスクの輪郭から4つの特徴点（顎、鼻、左端、右端）を検出・描画

【使用方法】
1. 通常実行（特徴点のみ）
   python chin_detection.py

2. 右端から顎までの輪郭を出力
   python chin_detection.py --rinkaku

3. 鼻先から右頬までの輪郭を出力
   python chin_detection.py --nose

4. 両方の輪郭を出力
   python chin_detection.py --rinkaku --nose
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import glob
import os
import argparse
import csv

SCRIPT_DIR = Path(__file__).resolve().parent

def find_mask_keypoints(contour):
    """
    マスクの輪郭から4つの特徴点を検出
    
    Args:
        contour: OpenCVの輪郭データ
        
    Returns:
        dict: 4つの特徴点座標と輪郭データ
    """
    
    points = []
    for point in contour:
        x, y = point[0][0], point[0][1]
        points.append((x, y))
    
    if not points:
        return None
    
    # Y座標でソート（上から下へ）
    points_by_y = sorted(points, key=lambda p: p[1])
    
    # 1. 顎座標（最下点）
    chin_point = points_by_y[-1]  # Y座標最大
    
    # 2. 鼻座標（最上点）
    nose_point = points_by_y[0]   # Y座標最小
    
    # 上部の点群から左右端を検出（上位20%の点を対象）
    upper_points = points_by_y[:max(1, len(points_by_y) // 5)]
    
    # 3. マスク上部左端(上部点群のX座標最大
    left_point = max(upper_points, key=lambda p: p[0]) 
    
    # 4. マスク上部右端（上部点群のX座標最小）
    right_point = min(upper_points, key=lambda p: p[0])
    
    return {
        'chin': chin_point,
        'nose': nose_point, 
        'left_edge': left_point,
        'right_edge': right_point,
        'all_points': points,
        'contour': contour
    }

def extract_contour_between_points(points, chin_point, right_point):
    """
    右端から顎までの輪郭部分を抽出（逆方向）
    
    Args:
        points: 輪郭上の全ての点
        chin_point: 顎座標
        right_point: 右端座標
        
    Returns:
        list: 右端から顎までの輪郭点
    """
    # 輪郭上で顎と右端の位置を探す
    chin_idx = None
    right_idx = None
    
    for i, p in enumerate(points):
        if p == chin_point:
            chin_idx = i
        if p == right_point:
            right_idx = i
    
    if chin_idx is None or right_idx is None:
        return []
    
    # 右端から顎までの点を取得（逆方向）
    if right_idx < chin_idx:
        contour_part = points[right_idx:chin_idx+1]
    else:
        contour_part = points[right_idx:] + points[:chin_idx+1]
    
    return contour_part

def extract_contour_right_to_nose(points, right_point, nose_point):
    """
    鼻先から右頬までの輪郭部分を抽出
    
    Args:
        points: 輪郭上の全ての点
        right_point: 右頬座標
        nose_point: 鼻先座標
        
    Returns:
        list: 鼻先から右頬までの輪郭点
    """
    # 輪郭上で右頬と鼻先の位置を探す
    right_idx = None
    nose_idx = None
    
    for i, p in enumerate(points):
        if p == right_point:
            right_idx = i
        if p == nose_point:
            nose_idx = i
    
    if right_idx is None or nose_idx is None:
        return []
    
    # 鼻先から右頬までの点を取得（逆方向）
    if nose_idx < right_idx:
        contour_part = points[nose_idx:right_idx+1]
    else:
        contour_part = points[nose_idx:] + points[:right_idx+1]
    
    return contour_part

def process_image_for_chin(model, image_path, output_dir, use_rinkaku=False, use_nose=False):
    """
    画像からマスクの特徴点を検出・描画
    
    Args:
        model: YOLOモデル
        image_path: 入力画像パス
        output_dir: 出力ディレクトリ
    """
    img_name = Path(image_path).stem
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"❌ 画像読み込み失敗: {image_path}")
        return None
    
    print(f"🖼️  処理中: {os.path.basename(image_path)}")
    
    # 画像を640×480にリサイズ
    target_size = (640, 480)
    resized_image = cv2.resize(original_image, target_size)
    
    # YOLO推論（リサイズした画像で実行）
    results = model(resized_image, max_det=1)[0]
    
    if results.masks is None:
        print(f"  ⚠️  マスクは検出されませんでした: {img_name}")
        return None
    
    chin_points = []
    
    # 各マスクインスタンスを処理
    for i, mask in enumerate(results.masks.data):
        # マスクを640×480サイズに合わせてリサイズ
        mask_resized = cv2.resize(mask.cpu().numpy(), target_size)
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 最大の輪郭を選択（メインのマスク領域）
            main_contour = max(contours, key=cv2.contourArea)
            
            # マスクの特徴点を検出
            keypoints = find_mask_keypoints(main_contour)
            
            if keypoints:
                chin_points.append({
                    'instance_id': i,
                    'keypoints': keypoints,
                    'contour_area': cv2.contourArea(main_contour)
                })
                
                print(f"  ✅ 特徴点検出: インスタンス{i}")
                print(f"    顎: {keypoints['chin']}")
                print(f"    鼻: {keypoints['nose']}")
                print(f"    左端: {keypoints['left_edge']}")
                print(f"    右端: {keypoints['right_edge']}")
    
    if not chin_points:
        print(f"  ⚠️  特徴点の検出に失敗しました: {img_name}")
        return None
    
    # 描画用画像を作成（リサイズした画像を使用）
    result_image = resized_image.copy()
    
    # 各特徴点を描画・保存
    for point_data in chin_points:
        keypoints = point_data['keypoints']
        instance_id = point_data['instance_id']
        
        # 各特徴点の描画設定
        point_configs = [
            {'name': 'Chin', 'point': keypoints['chin'], 'color': (0, 0, 255)},      # 赤：顎
            {'name': 'Nose', 'point': keypoints['nose'], 'color': (0, 255, 0)},      # 緑：鼻
            {'name': 'Left', 'point': keypoints['left_edge'], 'color': (255, 0, 0)}, # 青：左端
            {'name': 'Right', 'point': keypoints['right_edge'], 'color': (0, 255, 255)} # 黄：右端
        ]
        
        # 各特徴点を描画
        for config in point_configs:
            x, y = config['point']
            color = config['color']
            name = config['name']
            
            # 特徴点に円を描画
            cv2.circle(result_image, (x, y), 6, color, -1)
            
            # 座標テキストを表示
            cv2.putText(result_image, f"{name}({x},{y})", 
                       (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # --rinkaku オプションの処理
        rinkaku_points = []
        if use_rinkaku:
            chin_point = keypoints['chin']
            right_point = keypoints['right_edge']
            all_points = keypoints['all_points']
            
            # 顎から右端までの輪郭を抽出
            rinkaku_points = extract_contour_between_points(all_points, chin_point, right_point)
            
            if rinkaku_points:
                # 輪郭を結果画像に描画（紫色）
                rinkaku_array = np.array(rinkaku_points, dtype=np.int32)
                cv2.polylines(result_image, [rinkaku_array], False, (255, 0, 255), 2)
        
        # --nose オプションの処理
        nose_points = []
        if use_nose:
            right_point = keypoints['right_edge']
            nose_point = keypoints['nose']
            all_points = keypoints['all_points']
            
            # 右頬から鼻先までの輪郭を抽出
            nose_points = extract_contour_right_to_nose(all_points, right_point, nose_point)
            
            if nose_points:
                # 輪郭を結果画像に描画（青白色）
                nose_array = np.array(nose_points, dtype=np.int32)
                cv2.polylines(result_image, [nose_array], False, (255, 255, 0), 2)
        
        # 座標データをテキストファイルに保存
        coord_file = f"{output_dir}/{img_name}_inst{instance_id:02d}_keypoints.txt"
        with open(coord_file, 'w', encoding='utf-8') as f:
            f.write(f"マスク特徴点検出結果\n")
            f.write(f"画像: {os.path.basename(image_path)}\n")
            f.write(f"インスタンス: {instance_id}\n")
            f.write(f"顎座標 (x, y): {keypoints['chin']}\n")
            f.write(f"鼻座標 (x, y): {keypoints['nose']}\n")
            f.write(f"左端座標 (x, y): {keypoints['left_edge']}\n")
            f.write(f"右端座標 (x, y): {keypoints['right_edge']}\n")
            f.write(f"マスク面積: {point_data['contour_area']:.2f}px²\n")
            
            # --rinkaku オプションで輪郭座標を追加
            if use_rinkaku and rinkaku_points:
                f.write(f"\n【右端から顎までの輪郭座標】\n")
                for idx, point in enumerate(rinkaku_points):
                    f.write(f"点{idx}: {point}\n")
                
                # CSV形式で出力
                rinkaku_csv = f"{output_dir}/{img_name}_inst{instance_id:02d}_rinkaku.csv"
                with open(rinkaku_csv, 'w', newline='', encoding='utf-8') as csv_f:
                    writer = csv.writer(csv_f)
                    writer.writerow(['番号', 'x座標', 'y座標'])
                    for idx, point in enumerate(rinkaku_points):
                        writer.writerow([idx, point[0], point[1]])
            
            # --nose オプションで輪郭座標を追加
            if use_nose and nose_points:
                f.write(f"\n【鼻先から右頬までの輪郭座標】\n")
                for idx, point in enumerate(nose_points):
                    f.write(f"点{idx}: {point}\n")
                
                # CSV形式で出力
                nose_csv = f"{output_dir}/{img_name}_inst{instance_id:02d}_nose.csv"
                with open(nose_csv, 'w', newline='', encoding='utf-8') as csv_f:
                    writer = csv.writer(csv_f)
                    writer.writerow(['番号', 'x座標', 'y座標'])
                    for idx, point in enumerate(nose_points):
                        writer.writerow([idx, point[0], point[1]])
    
    # 結果画像を保存
    result_file = f"{output_dir}/{img_name}_keypoints_detection.jpg"
    cv2.imwrite(result_file, result_image)
    
    return chin_points

def main():
    print("🎯 マスク特徴点検出プログラム")
    print("=" * 40)
    
    # コマンドラインパーサー
    parser = argparse.ArgumentParser(description="マスク特徴点検出プログラム")
    parser.add_argument("--rinkaku", action="store_true", help="右端から顎までの輪郭座標を出力")
    parser.add_argument("--nose", action="store_true", help="右頬から鼻先までの輪郭座標を出力")
    args = parser.parse_args()
    
    # 設定
    model_path = SCRIPT_DIR / "best.pt"
    input_path = SCRIPT_DIR / "testdata" / "masked4_face_up.jpg"
    output_dir = SCRIPT_DIR / "testdata"
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # モデル読み込み
    print(f"📦 モデル読み込み: {model_path}")
    try:
        model = YOLO(str(model_path))
        print("✅ モデル読み込み成功")
    except Exception as e:
        print(f"❌ モデル読み込み失敗: {e}")
        return
    
    # 画像ファイル取得（単一ファイル指定とディレクトリ指定の両方に対応）
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []

    if input_path.is_file():
        image_files = [str(input_path)]
    elif input_path.is_dir():
        for ext in image_extensions:
            image_files.extend(glob.glob(str(input_path / ext)))
            image_files.extend(glob.glob(str(input_path / ext.upper())))
    
    if not image_files:
        print(f"❌ {input_path} に画像ファイルが見つかりません")
        return
    
    print(f"\n📁 処理対象: {len(image_files)}枚の画像")
    print(f"💾 出力先: {output_dir}")
    if args.rinkaku:
        print(f"🔴 輪郭出力: ON（右端から顎までの輪郭座標を出力）")
    if args.nose:
        print(f"🔵 輪郭出力: ON（右頬から鼻先までの輪郭座標を出力）")
    print()
    
    # 処理統計
    total_processed = 0
    successful_detections = 0
    all_chin_points = []
    
    # 各画像を処理
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] ", end="")
        
        chin_points = process_image_for_chin(model, image_path, output_dir, use_rinkaku=args.rinkaku, use_nose=args.nose)
        total_processed += 1
        
        if chin_points:
            successful_detections += 1
            all_chin_points.extend(chin_points)
    
    # 結果統計表示
    print(f"\n🎉 処理完了!")
    print(f"📊 処理統計:")
    print(f"  - 処理画像数: {total_processed}")
    print(f"  - 検出成功: {successful_detections}")
    print(f"  - 検出率: {successful_detections/total_processed*100:.1f}%")
    print(f"  - 検出された特徴点セット: {len(all_chin_points)}個")
    
    # ファイル生成統計
    result_images = len(glob.glob(str(output_dir / "*_keypoints_detection.jpg")))
    coord_files = len(glob.glob(str(output_dir / "*_keypoints.txt")))
    
    print(f"\n📁 生成ファイル:")
    print(f"  - 結果画像: {result_images}")
    print(f"  - 座標テキスト: {coord_files}")
    print(f"\n💾 保存先: {output_dir}")

if __name__ == "__main__":
    main()
