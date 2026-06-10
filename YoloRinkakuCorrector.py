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
            draw_debug_overlay=True,
            use_landmark_crop=False, # YOLO入力をランドマークに基づいて切り抜き
            crop_margin_ratio=0.35,
            use_mask_edge_inpaint=False,# inpaintでマスクの端をなじませる
            edge_inpaint_erode=30,
            edge_inpaint_radius=1,
            edge_color_blend_alpha=1,
            edge_color_feather=0):
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
        self.use_landmark_crop = use_landmark_crop
        self.crop_margin_ratio = crop_margin_ratio
        self.use_mask_edge_inpaint = use_mask_edge_inpaint
        self.edge_inpaint_erode = edge_inpaint_erode
        self.edge_inpaint_radius = edge_inpaint_radius
        self.edge_color_blend_alpha = edge_color_blend_alpha
        self.edge_color_feather = edge_color_feather

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
        self.latest_crop_rect = None
        self.latest_mask_contour = None
        self.latest_rinkaku_mode = None
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

        if self.latest_crop_rect is not None:
            x1, y1, x2, y2 = self.latest_crop_rect
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 128, 0), 2)

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

    def get_landmark_crop_rect(self, face_landmarks):
        if face_landmarks is None:
            return None

        xs = []
        ys = []
        for landmark in face_landmarks.landmark:
            x = landmark.x * self.width
            y = landmark.y * self.height
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)

        if not xs or not ys:
            return None

        x_min = max(0.0, min(xs))
        x_max = min(float(self.width - 1), max(xs))
        y_min = max(0.0, min(ys))
        y_max = min(float(self.height - 1), max(ys))
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        if crop_w <= 1 or crop_h <= 1:
            return None

        margin = max(crop_w, crop_h) * self.crop_margin_ratio
        x1 = int(max(0, np.floor(x_min - margin)))
        y1 = int(max(0, np.floor(y_min - margin)))
        x2 = int(min(self.width, np.ceil(x_max + margin)))
        y2 = int(min(self.height, np.ceil(y_max + margin)))

        return x1, y1, x2, y2

    def prepare_yolo_input(self, bgr_image, face_landmarks):
        if bgr_image.shape[1] != self.width or bgr_image.shape[0] != self.height:
            full_image = cv2.resize(bgr_image, (self.width, self.height))
        else:
            full_image = bgr_image

        if not self.use_landmark_crop:
            self.latest_crop_rect = None
            return full_image, (0, 0, self.width, self.height)

        crop_rect = self.get_landmark_crop_rect(face_landmarks)
        if crop_rect is None:
            self.latest_crop_rect = None
            return full_image, (0, 0, self.width, self.height)

        x1, y1, x2, y2 = crop_rect
        self.latest_crop_rect = crop_rect
        return full_image[y1:y2, x1:x2], crop_rect

    def reset_latest_detection(self, keep_crop=False):
        self.latest_yolo_keypoints = None
        self.latest_rinkaku_points = []
        self.latest_mask_contour = None
        self.latest_rinkaku_mode = None
        if not keep_crop:
            self.latest_crop_rect = None

    def apply_edge_inpaint(self, rgb_image, edge_rgb=None):
        if (
                not self.use_mask_edge_inpaint
                or rgb_image is None
                or self.latest_mask_contour is None):
            return rgb_image

        h, w = rgb_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        contour = np.array(self.latest_mask_contour, dtype=np.int32)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        if self.edge_inpaint_erode > 0:
            kernel_size = self.edge_inpaint_erode * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            inner_mask = cv2.erode(mask, kernel, iterations=1)
            edge_mask = cv2.subtract(mask, inner_mask)
        else:
            edge_mask = mask

        if np.count_nonzero(edge_mask) == 0:
            return rgb_image

        inpainted = cv2.inpaint(
            rgb_image,
            edge_mask,
            self.edge_inpaint_radius,
            cv2.INPAINT_TELEA)

        if edge_rgb is None or self.edge_color_blend_alpha <= 0:
            return inpainted

        alpha = min(1.0, max(0.0, self.edge_color_blend_alpha))
        alpha_mask = edge_mask.astype(np.float32) / 255.0
        if self.edge_color_feather > 0:
            kernel_size = self.edge_color_feather * 2 + 1
            alpha_mask = cv2.GaussianBlur(alpha_mask, (kernel_size, kernel_size), 0)
        alpha_mask = (alpha_mask * alpha)[..., None]

        if isinstance(edge_rgb, dict):
            left_rgb = edge_rgb.get('left')
            right_rgb = edge_rgb.get('right')
            if left_rgb is None and right_rgb is None:
                return inpainted
            if left_rgb is None:
                left_rgb = right_rgb
            if right_rgb is None:
                right_rgb = left_rgb

            x_values = np.linspace(0.0, 1.0, w, dtype=np.float32)
            x_values = np.tile(x_values[None, :, None], (h, 1, 1))
            left_rgb = np.asarray(left_rgb, dtype=np.float32)
            right_rgb = np.asarray(right_rgb, dtype=np.float32)
            edge_rgb = left_rgb * (1.0 - x_values) + right_rgb * x_values
            if self.edge_color_feather > 0:
                kernel_size = self.edge_color_feather * 2 + 1
                edge_rgb = cv2.GaussianBlur(edge_rgb, (kernel_size, kernel_size), 0)
        else:
            edge_rgb = np.asarray(edge_rgb, dtype=np.float32)

        blended = inpainted.astype(np.float32)
        blended = blended * (1.0 - alpha_mask) + edge_rgb * alpha_mask
        return np.clip(blended, 0, 255).astype(np.uint8)

    def update_landmark_overrides_from_yolo(self, bgr_image, rinkaku_mode='right', face_landmarks=None):
        if not self.available or self.model is None or bgr_image is None:
            self.reset_latest_detection()
            return False, {}
        self.latest_rinkaku_mode = rinkaku_mode

        target_image, crop_rect = self.prepare_yolo_input(bgr_image, face_landmarks)
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_rect
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        try:
            results = self.model(target_image, max_det=1, verbose=False)[0]
        except Exception as e:
            print(f"YOLO推論エラー: {e}")
            self.reset_latest_detection(keep_crop=True)
            return False, {}

        if results.masks is None:
            self.reset_latest_detection(keep_crop=True)
            return False, {}

        best_contour = None
        best_area = 0.0

        for mask in results.masks.data:
            mask_resized = cv2.resize(mask.cpu().numpy(), (crop_w, crop_h))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            main_contour = max(contours, key=cv2.contourArea)
            if crop_x1 or crop_y1:
                main_contour = main_contour.copy()
                main_contour[:, 0, 0] += crop_x1
                main_contour[:, 0, 1] += crop_y1
            area = cv2.contourArea(main_contour)
            if area > best_area:
                best_area = area
                best_contour = main_contour

        if best_contour is None:
            self.reset_latest_detection(keep_crop=True)
            return False, {}

        keypoints = self.find_mask_keypoints(best_contour)
        if not keypoints:
            self.reset_latest_detection(keep_crop=True)
            return False, {}
        self.latest_mask_contour = best_contour.copy()

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
