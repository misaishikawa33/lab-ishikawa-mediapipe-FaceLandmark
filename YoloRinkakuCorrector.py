import csv
import os

import cv2
import numpy as np


class YoloRinkakuCorrector:
    def __init__(
            self,
            width,
            height,
            enabled=True,
            update_interval=5,
            model_path='yolofolder/best.pt',
            export_csv=False,
            csv_path='mqodata/input/yolooutput.csv',
            use_outlier_filter=True,
            border_margin_ratio=0.05,
            yaw_threshold_neg=-20,
            yaw_threshold_pos=20,
            target_landmarks_right=None,
            target_landmarks_left=None,
            draw_debug_overlay=True):
        self.width = width
        self.height = height
        self.enabled = enabled
        self.update_interval = update_interval
        self.model_path = model_path
        self.export_csv = export_csv
        self.csv_path = csv_path
        self.use_outlier_filter = use_outlier_filter
        self.border_margin_ratio = border_margin_ratio
        self.yaw_threshold_neg = yaw_threshold_neg
        self.yaw_threshold_pos = yaw_threshold_pos
        self.draw_debug_overlay = draw_debug_overlay

        self.target_landmarks_right = target_landmarks_right or [
            111, 116, 123, 147, 213, 192, 138, 135, 169, 150, 149, 176, 148, 152
        ]
        self.target_landmarks_left = target_landmarks_left or [
            340, 345, 352, 376, 411, 427, 416, 434, 364, 394, 369, 400, 377, 152
        ]
        self.mode_config = {
            'right': {
                'start_key': 'right_edge',
                'avoid_key': None,
                'target_landmarks': self.target_landmarks_right,
            },
            'left': {
                'start_key': 'left_edge',
                'avoid_key': 'right_edge',
                'target_landmarks': self.target_landmarks_left,
            },
        }

        self.model = None
        self.available = False
        self.latest_yolo_keypoints = None
        self.latest_rinkaku_points = []
        self.landmark_overrides_px = {}
        self.landmark_overrides_loaded = False

        self.initialize_model()

    def initialize_model(self):
        if not self.enabled:
            return

        if not os.path.exists(self.model_path):
            print(f"YOLOモデルが見つかりません: {self.model_path}")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.available = True
            print(f"YOLO輪郭補正を有効化: interval={self.update_interval}")
        except Exception as e:
            self.model = None
            self.available = False
            print(f"YOLO初期化に失敗しました: {e}")

    def get_rinkaku_mode_from_yaw(self, yaw):
        if yaw is None:
            return None
        if yaw >= self.yaw_threshold_pos:
            return 'left'
        if yaw <= self.yaw_threshold_neg:
            return 'right'
        return None

    def should_update(self, frame_count):
        return (
            not self.landmark_overrides_loaded
            or (frame_count % max(1, self.update_interval) == 0)
        )

    def is_yolo_keypoints_reliable(self, keypoints, rinkaku_mode):
        if not keypoints:
            return False

        mode_config = self.mode_config.get(rinkaku_mode, self.mode_config['right'])
        border_margin = max(1, int(min(self.width, self.height) * self.border_margin_ratio))

        required_keys = ['chin', mode_config['start_key']]
        if mode_config.get('avoid_key'):
            required_keys.append(mode_config['avoid_key'])

        for key in required_keys:
            point = keypoints.get(key)
            if point is None:
                return False

            x, y = point
            if x <= border_margin or x >= (self.width - border_margin):
                print(f"YOLO結果を破棄: {key} が画面端に近すぎます ({x:.1f}, {y:.1f})")
                return False
            if y <= border_margin or y >= (self.height - border_margin):
                print(f"YOLO結果を破棄: {key} が画面端に近すぎます ({x:.1f}, {y:.1f})")
                return False

        return True

    def find_mask_keypoints(self, contour):
        points = []
        for point in contour:
            x, y = point[0][0], point[0][1]
            points.append((x, y))

        if not points:
            return None

        points_by_y = sorted(points, key=lambda p: p[1])
        chin_point = points_by_y[-1]
        nose_point = points_by_y[0]
        upper_points = points_by_y[:max(1, len(points_by_y) // 5)]
        left_point = max(upper_points, key=lambda p: p[0])
        right_point = min(upper_points, key=lambda p: p[0])

        return {
            'chin': chin_point,
            'nose': nose_point,
            'left_edge': left_point,
            'right_edge': right_point,
            'all_points': points,
        }

    def extract_contour_between_points(self, points, chin_point, right_point):
        chin_idx = None
        right_idx = None

        for i, p in enumerate(points):
            if p == chin_point:
                chin_idx = i
            if p == right_point:
                right_idx = i

        if chin_idx is None or right_idx is None:
            return []

        if right_idx < chin_idx:
            return points[right_idx:chin_idx + 1]
        return points[right_idx:] + points[:chin_idx + 1]

    def extract_contour_between_points_avoiding(self, points, start_point, end_point, avoid_point):
        if not points:
            return []

        start_idx = None
        end_idx = None
        for i, p in enumerate(points):
            if p == start_point:
                start_idx = i
            if p == end_point:
                end_idx = i

        if start_idx is None or end_idx is None:
            return []

        n = len(points)

        forward_path = []
        i = start_idx
        while True:
            forward_path.append(points[i])
            if i == end_idx:
                break
            i = (i + 1) % n

        backward_path = []
        i = start_idx
        while True:
            backward_path.append(points[i])
            if i == end_idx:
                break
            i = (i - 1 + n) % n

        if avoid_point is None or avoid_point == start_point or avoid_point == end_point:
            return forward_path

        forward_has_avoid = avoid_point in forward_path
        backward_has_avoid = avoid_point in backward_path

        if forward_has_avoid and not backward_has_avoid:
            return backward_path
        if backward_has_avoid and not forward_has_avoid:
            return forward_path

        return forward_path

    def save_rinkaku_points_to_csv(self, points):
        try:
            dir_path = os.path.dirname(self.csv_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(self.csv_path, 'w', newline='', encoding='utf-8') as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(['番号', 'x座標', 'y座標'])
                for idx, point in enumerate(points):
                    writer.writerow([idx, point[0], point[1]])
        except Exception as e:
            print(f"輪郭CSV保存エラー: {e}")

    def draw_overlay(self, image):
        if image is None:
            return

        if self.latest_rinkaku_points:
            rinkaku_array = np.array(self.latest_rinkaku_points, dtype=np.int32)
            cv2.polylines(image, [rinkaku_array], False, (255, 255, 0), 2)

        if not self.latest_yolo_keypoints:
            return

        chin = self.latest_yolo_keypoints.get('chin')
        left = self.latest_yolo_keypoints.get('left_edge')
        right = self.latest_yolo_keypoints.get('right_edge')

        if chin is not None:
            cx, cy = int(chin[0]), int(chin[1])
            cv2.circle(image, (cx, cy), 7, (255, 0, 0), -1)
            cv2.putText(image, f"chin({cx},{cy})", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if left is not None:
            lx, ly = int(left[0]), int(left[1])
            cv2.circle(image, (lx, ly), 7, (0, 255, 0), -1)
            cv2.putText(image, f"left({lx},{ly})", (lx + 8, ly - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if right is not None:
            rx, ry = int(right[0]), int(right[1])
            cv2.circle(image, (rx, ry), 7, (0, 255, 255), -1)
            cv2.putText(image, f"right({rx},{ry})", (rx + 8, ry - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def update_landmark_overrides_from_yolo(self, bgr_image, rinkaku_mode='right'):
        if not self.available or self.model is None or bgr_image is None:
            self.latest_yolo_keypoints = None
            self.latest_rinkaku_points = []
            return False, {}

        if bgr_image.shape[1] != self.width or bgr_image.shape[0] != self.height:
            target_image = cv2.resize(bgr_image, (self.width, self.height))
        else:
            target_image = bgr_image

        try:
            results = self.model(target_image, max_det=1, verbose=False)[0]
        except Exception as e:
            print(f"YOLO推論エラー: {e}")
            self.latest_yolo_keypoints = None
            self.latest_rinkaku_points = []
            return False, {}

        if results.masks is None:
            self.latest_yolo_keypoints = None
            self.latest_rinkaku_points = []
            return False, {}

        best_contour = None
        best_area = 0.0

        for mask in results.masks.data:
            mask_resized = cv2.resize(mask.cpu().numpy(), (self.width, self.height))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            if area > best_area:
                best_area = area
                best_contour = main_contour

        if best_contour is None:
            self.latest_yolo_keypoints = None
            self.latest_rinkaku_points = []
            return False, {}

        keypoints = self.find_mask_keypoints(best_contour)
        if not keypoints:
            self.latest_yolo_keypoints = None
            self.latest_rinkaku_points = []
            return False, {}

        mode_config = self.mode_config.get(rinkaku_mode, self.mode_config['right'])
        contour_start_point = keypoints.get(mode_config['start_key'])
        avoid_key = mode_config['avoid_key']
        avoid_point = keypoints.get(avoid_key) if avoid_key else None
        target_landmarks = mode_config['target_landmarks']

        if contour_start_point is None:
            self.latest_yolo_keypoints = keypoints
            self.latest_rinkaku_points = []
            return False, {}

        if self.use_outlier_filter and not self.is_yolo_keypoints_reliable(keypoints, rinkaku_mode):
            self.latest_yolo_keypoints = keypoints
            self.latest_rinkaku_points = []
            return False, {}

        if avoid_point is not None:
            rinkaku_points = self.extract_contour_between_points_avoiding(
                keypoints['all_points'],
                contour_start_point,
                keypoints['chin'],
                avoid_point
            )
        else:
            rinkaku_points = self.extract_contour_between_points(
                keypoints['all_points'],
                keypoints['chin'],
                contour_start_point
            )
        if len(rinkaku_points) < 2:
            self.latest_yolo_keypoints = keypoints
            self.latest_rinkaku_points = []
            return False, {}

        self.latest_yolo_keypoints = keypoints
        self.latest_rinkaku_points = rinkaku_points

        if self.export_csv:
            self.save_rinkaku_points_to_csv(rinkaku_points)

        success, overrides = self.build_landmark_overrides_from_points(
            rinkaku_points,
            target_landmarks
        )
        self.landmark_overrides_loaded = success
        if success:
            self.landmark_overrides_px.update(overrides)

        return success, overrides

    def build_landmark_overrides_from_points(self, points, target_landmarks):
        if len(points) < 2 or not target_landmarks:
            return False, {}

        distances = [0.0]
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            distances.append(distances[-1] + np.sqrt(dx * dx + dy * dy))

        total_distance = distances[-1]
        if total_distance <= 0:
            return False, {}

        target_count = len(target_landmarks)
        overrides = {}
        for idx, landmark_id in enumerate(target_landmarks):
            if target_count > 1:
                target_distance = (idx / (target_count - 1)) * total_distance
            else:
                target_distance = 0.0

            for i in range(len(distances) - 1):
                if distances[i] <= target_distance <= distances[i + 1]:
                    segment = distances[i + 1] - distances[i]
                    ratio = 0.0 if segment <= 0 else (target_distance - distances[i]) / segment
                    x_interp = points[i][0] + ratio * (points[i + 1][0] - points[i][0])
                    y_interp = points[i][1] + ratio * (points[i + 1][1] - points[i][1])
                    overrides[landmark_id] = (x_interp, y_interp)
                    break

        return bool(overrides), overrides

    def apply_landmark_overrides(self, face_landmarks):
        try:
            total = len(face_landmarks.landmark)
            for idx, (x_px, y_px) in self.landmark_overrides_px.items():
                if idx < 0 or idx >= total:
                    continue
                face_landmarks.landmark[idx].x = x_px / self.width
                face_landmarks.landmark[idx].y = y_px / self.height
        except Exception as e:
            print(f"ランドマーク上書き適用エラー: {e}")
