# Mediapipの座標を変換ツール
# XとYのスケール変換と平行移動を行う

"""
    P: 変換したい点群（MediaPipeの片側輪郭）
    p1: MediaPipe側アンカー（顎）
    p2: MediaPipe側アンカー（頬側）
    p3: MediaPipe側アンカー（鼻根本）
    q1: YOLO側アンカー（顎）
    q2: YOLO側アンカー（頬側）
    q3: YOLO側アンカー（鼻根本）
    [📍] 152番→234番の輪郭範囲: [234, 227, 137, 132, 58, 172, 136, 150, 149, 176, 148, 152]
    [📍] 234番→197番の輪郭範囲: [234, 227, 137, 132, 58, 172, 136, 150, 149, 176, 148, 152,116, 117, 118, 119, 120, 121,114, 196,197]
    [📍] 454番→197番の輪郭範囲: [227, 137, 132, 58, 172, 136, 150, 149, 176, 148, 152,340,343,345,350,359,419,449,449,450,454,197]

    顎：152番
    鼻先：197番
    左頬：
    右頬：234 番

    実行方法
    ・2点で相似変換：python withmediapipe.py --anchors 2
    ・3点で相似変換：python withmediapipe.py --anchors 3

"""
import cv2
import mediapipe as mp
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="MediaPipeランドマークの相似変換")
parser.add_argument("--anchors", type=int, choices=[2, 3], default=2, help="アンカー点数（2 or 3）")
args = parser.parse_args()

id = "masked4_face_up"  #写真名（拡張子なし）/input_picture/test0115へのパス

#入出力設定
image_path = f"./input_picture/test0115/{id}.jpg"  # 画像のパス
json_path = f"./input/test0115/input_json/{id}_landmarks.json"  # JSONファイルのパス
#output_path = f"./output_picture/test0115/{id}_with_landmarks.jpg"  # 出力画像のパス

# JSONファイルからMediaPipeのランドマークを読み込む
with open(json_path, 'r') as f:
    landmarks_data = json.load(f)

# MediaPipeのランドマーク座標を取得（画像IDをキーとして使用）
landmarks = landmarks_data[id]  # ランドマークのリスト

# 画像サイズ（640×480）
img_width = 640
img_height = 480



#Mediapipe側のアンカー（顎:152番、右頬:234番）
# 正規化座標を画像座標に変換
p1 = np.array([landmarks[152]['x'] * img_width, landmarks[152]['y'] * img_height])  # MediaPipe側アンカー（顎）
p2 = np.array([landmarks[234]['x'] * img_width, landmarks[234]['y'] * img_height])  # MediaPipe側アンカー（頬側）
p3 = np.array([landmarks[197]['x'] * img_width, landmarks[197]['y'] * img_height])  # MediaPipe側アンカー（鼻根本）

#Yolo側のアンカー(640×480画像における座標)
q1 = np.array([266, 437])  # YOLO側アンカー（顎）
q2 = np.array([162, 279])  # YOLO側アンカー（頬側）
q3 = np.array([196, 262])  # YOLO側アンカー（鼻根本）

# 2点/3点を選択（2 or 3）
num_anchors = args.anchors


# ==========================================
# 相似変換の計算
# ==========================================

if num_anchors == 3:
    # --- 3点の相似変換（SVDによる最小二乗） ---
    def compute_similarity(src_points, dst_points):
        src = np.asarray(src_points, dtype=np.float64)
        dst = np.asarray(dst_points, dtype=np.float64)
        if src.shape[0] < 3 or dst.shape[0] < 3:
            raise ValueError("3点相似変換は3点以上必要です。")

        src_mean = src.mean(axis=0) #重心の計算（mediapipe側）
        dst_mean = dst.mean(axis=0) #重心の計算（yolo側）
        src_centered = src - src_mean #重心を原点に移動
        dst_centered = dst - dst_mean #重心を原点に移動

        cov = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(cov)
        R = Vt.T @ U.T #回転行列

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T #反転を防ぐ

        scale = np.sum(S) / np.sum(src_centered ** 2) #スケール計算
        t_vec = dst_mean - scale * R @ src_mean #平行移動ベクトル計算
        return scale, R, t_vec

    src_points = np.stack([p1, p2, p3])
    dst_points = np.stack([q1, q2, q3])
    s, R, t_vec = compute_similarity(src_points, dst_points)
    # 回転角度 θ（ラジアン）
    theta = np.arctan2(R[1, 0], R[0, 0]) # 回転行列から角度を抽出


elif num_anchors == 2:
    # --- 2点の相似変換（従来の方法） ---
    v_mp = p2 - p1  # MediaPipe側のベクトル（顎→頬）
    v_yolo = q2 - q1  # YOLO側のベクトル（顎→頬）

    # スケール s = ||v_yolo|| / ||v_mp||
    s = np.linalg.norm(v_yolo) / np.linalg.norm(v_mp)

    # 回転角度 θ（ラジアン）
    theta_mp = np.arctan2(v_mp[1], v_mp[0])  # MediaPipe側の角度
    theta_yolo = np.arctan2(v_yolo[1], v_yolo[0])  # YOLO側の角度
    theta = theta_yolo - theta_mp  # 回転角度差

    # 回転行列
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # 平行移動ベクトルの計算
    t_vec = q1 - s * R @ p1
else:
    raise ValueError("num_anchors は 2 または 3 を指定してください。")

print(f"相似変換パラメータ:")
print(f"  スケール s = {s:.6f}")
print(f"  回転角度 θ = {np.degrees(theta):.2f}° ({theta:.6f} rad)")
print(f"  平行移動 t = ({t_vec[0]:.2f}, {t_vec[1]:.2f})")

# ==========================================
# 指定ランドマークに対して顎固定の相似変換を適用
# ==========================================

# 変換対象のランドマーク（順序を保持）
# target_landmarks = [234, 227, 137, 132, 58, 172, 136, 150, 149, 176, 148, 152,116, 117, 118, 119, 120, 121,114, 196,197]
target_landmarks = [234, 227, 137, 132, 58, 172, 136, 150, 149, 176, 148, 152]

# 変換結果を表示

print(f"{'Index':<6} {'YOLO X':<12} {'YOLO Y':<12}")
print("=" * 30)

blended_points = {}

for i, idx in enumerate(target_landmarks):
    # MediaPipe座標を取得（正規化座標から画像座標に変換）
    p_i = np.array([landmarks[idx]['x'] * img_width, landmarks[idx]['y'] * img_height])
    
    # 相似変換を適用（標準形式）: p_yolo = s*R*p_i + t
    p_yolo = s * R @ p_i + t_vec

     # 重み（頬→顎で増加）
    N = len(target_landmarks)
    gamma = 2.0
    w_max = 0.9
    t_weight = i / (N - 1)
    w = min(w_max, t_weight ** gamma)


    if idx == 152:  # 顎
        w = 1.0

    # 部分補正（x, y両方）
    p_blend = (1 - w) * p_i + w * p_yolo
    
    blended_points[idx] = p_blend

    print(f"{idx:<6} 相似変換=({p_yolo[0]:.2f},{p_yolo[1]:.2f}) "
          f"相似変換(重みあり)=({p_blend[0]:.2f},{p_blend[1]:.2f})")

print(f"\n✅ 変換完了")






   