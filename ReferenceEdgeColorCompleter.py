import os

import cv2
import mediapipe as mp
import numpy as np


class ReferenceEdgeColorCompleter:
    face_inner_227_to_175_ccw = [227, 137, 177, 215, 138, 135, 169, 170, 140, 171, 175]
    face_inner_447_to_175_cw = [447, 366, 401, 435, 367, 364, 394, 395, 369, 396, 175]
    face_outer_234_to_152_ccw = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
    face_outer_454_to_152_cw = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]

    def __init__(
            self,
            enabled=True,
            left_reference_path='mqodata/face7.jpg',
            right_reference_path='mqodata/face22.jpg',
            sample_radius=4,
            paint_radius=4,
            blend_alpha=1.0,
            source_mode='outer',
            draw_debug_landmarks=False,
            debug_output_dir='output/debug'):
        self.enabled = enabled
        self.left_reference_path = left_reference_path
        self.right_reference_path = right_reference_path
        self.sample_radius = max(1, int(sample_radius))
        self.paint_radius = max(1, int(paint_radius))
        self.blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        self.source_mode = source_mode
        self.draw_debug_landmarks = draw_debug_landmarks
        self.debug_output_dir = debug_output_dir
        self._reference_cache = {}

    def complete_texture(self, target_img, target_landmarks, texture_path):
        if not self.enabled or target_img is None or target_landmarks is None:
            return target_img, texture_path

        completed_img = target_img.copy()
        completion_specs = self._build_completion_specs()

        color_updates = {}
        debug_references = []
        side_color_updates = []
        for side_name, reference_path, source_landmark_ids, target_landmark_ids in completion_specs:
            reference = self._load_reference(reference_path)
            if reference is None:
                continue

            side_count = 0
            used_landmark_ids = []
            side_updates = []
            for source_landmark_id, target_landmark_id in zip(source_landmark_ids, target_landmark_ids):
                if (
                        source_landmark_id >= len(reference['landmarks'].landmark)
                        or target_landmark_id >= len(target_landmarks.landmark)):
                    continue

                reference_color = self._sample_landmark_bgr(
                    reference['image'],
                    reference['landmarks'].landmark[source_landmark_id])
                if reference_color is None:
                    continue

                color_updates.setdefault(target_landmark_id, []).append(reference_color)
                side_updates.append((target_landmark_id, reference_color))
                used_landmark_ids.append(source_landmark_id)
                side_count += 1

            if side_count > 0:
                print(f"参照色補完({side_name})を適用: {reference_path}")
                debug_references.append((side_name, reference_path, reference, used_landmark_ids))
                side_color_updates.append((side_name, side_updates))

        updated_count = 0
        target_debug_points = []
        for side_name, side_updates in side_color_updates:
            self._paint_side_segments(completed_img, target_landmarks, side_updates)

        for landmark_id, colors in color_updates.items():
            target_xy = self._landmark_to_pixel(
                target_landmarks.landmark[landmark_id],
                completed_img.shape)
            reference_color = np.mean(np.asarray(colors, dtype=np.float32), axis=0)
            self._paint_color(completed_img, target_xy, reference_color)
            target_debug_points.append((landmark_id, target_xy))
            updated_count += 1

        if updated_count == 0:
            return target_img, texture_path

        output_path = self._build_output_path(texture_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, completed_img)
        print(f"参照輪郭色補完テクスチャを保存しました: {output_path} ({updated_count}点)")

        if self.draw_debug_landmarks:
            self._write_debug_images(
                target_img,
                completed_img,
                target_debug_points,
                debug_references,
                texture_path)

        return completed_img, output_path

    def _load_reference(self, image_path):
        if image_path in self._reference_cache:
            return self._reference_cache[image_path]

        if not os.path.exists(image_path):
            print(f"参照色補完画像が見つかりません: {image_path}")
            self._reference_cache[image_path] = None
            return None

        img = cv2.imread(image_path)
        if img is None:
            print(f"参照色補完画像を読み込めません: {image_path}")
            self._reference_cache[image_path] = None
            return None

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb_img)

        if not results.multi_face_landmarks:
            print(f"参照色補完画像から顔ランドマークを検出できません: {image_path}")
            self._reference_cache[image_path] = None
            return None

        reference = {
            'image': img,
            'landmarks': results.multi_face_landmarks[0],
        }
        self._reference_cache[image_path] = reference
        return reference

    def _build_completion_specs(self):
        if self.source_mode == 'inner':
            right_source = self.face_inner_227_to_175_ccw
            left_source = self.face_inner_447_to_175_cw
        elif self.source_mode == 'outer':
            right_source = self.face_outer_234_to_152_ccw
            left_source = self.face_outer_454_to_152_cw
        else:
            raise ValueError(f"Unsupported reference edge color source mode: {self.source_mode}")

        return [
            (
                'right',
                self.left_reference_path,
                right_source,
                self.face_outer_234_to_152_ccw,
            ),
            (
                'left',
                self.right_reference_path,
                left_source,
                self.face_outer_454_to_152_cw,
            ),
        ]

    def _sample_landmark_bgr(self, img, landmark):
        h, w = img.shape[:2]
        x, y = self._landmark_to_pixel(landmark, img.shape)
        x1 = max(0, x - self.sample_radius)
        x2 = min(w, x + self.sample_radius + 1)
        y1 = max(0, y - self.sample_radius)
        y2 = min(h, y + self.sample_radius + 1)
        if x1 >= x2 or y1 >= y2:
            return None

        patch = img[y1:y2, x1:x2, :3].astype(np.float32)
        return np.median(patch.reshape(-1, 3), axis=0)

    def _paint_color(self, img, center_xy, bgr_color):
        h, w = img.shape[:2]
        x, y = center_xy
        x1 = max(0, x - self.paint_radius)
        x2 = min(w, x + self.paint_radius + 1)
        y1 = max(0, y - self.paint_radius)
        y2 = min(h, y + self.paint_radius + 1)
        if x1 >= x2 or y1 >= y2:
            return

        yy, xx = np.ogrid[y1:y2, x1:x2]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= self.paint_radius ** 2
        patch = img[y1:y2, x1:x2, :3].astype(np.float32)
        patch[mask] = (
            (1.0 - self.blend_alpha) * patch[mask]
            + self.blend_alpha * np.asarray(bgr_color, dtype=np.float32))
        img[y1:y2, x1:x2, :3] = np.clip(patch, 0, 255).astype(np.uint8)

    def _paint_side_segments(self, img, target_landmarks, side_updates):
        if len(side_updates) < 2:
            return

        thickness = max(1, self.paint_radius * 2 + 1)
        for (start_id, start_color), (end_id, end_color) in zip(side_updates[:-1], side_updates[1:]):
            if start_id >= len(target_landmarks.landmark) or end_id >= len(target_landmarks.landmark):
                continue

            start_xy = self._landmark_to_pixel(target_landmarks.landmark[start_id], img.shape)
            end_xy = self._landmark_to_pixel(target_landmarks.landmark[end_id], img.shape)
            segment_color = np.mean(
                np.asarray([start_color, end_color], dtype=np.float32),
                axis=0)
            self._paint_line(img, start_xy, end_xy, segment_color, thickness)

    def _paint_line(self, img, start_xy, end_xy, bgr_color, thickness):
        color = tuple(int(v) for v in np.clip(bgr_color, 0, 255))
        if self.blend_alpha >= 1.0:
            cv2.line(img, start_xy, end_xy, color, thickness, cv2.LINE_AA)
            return

        overlay = img.copy()
        cv2.line(overlay, start_xy, end_xy, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, self.blend_alpha, img, 1.0 - self.blend_alpha, 0, dst=img)

    def _write_debug_images(
            self,
            target_img,
            completed_img,
            target_debug_points,
            debug_references,
            texture_path):
        os.makedirs(self.debug_output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(texture_path))[0]

        target_debug = target_img.copy()
        completed_debug = completed_img.copy()
        for landmark_id, xy in target_debug_points:
            self._draw_landmark_marker(target_debug, xy, landmark_id, (0, 255, 255))
            self._draw_landmark_marker(completed_debug, xy, landmark_id, (0, 255, 255))

        target_path = os.path.join(
            self.debug_output_dir,
            f"{base_name}_reference_edge_target_landmarks.png")
        completed_path = os.path.join(
            self.debug_output_dir,
            f"{base_name}_reference_edge_completed_landmarks.png")
        cv2.imwrite(target_path, target_debug)
        cv2.imwrite(completed_path, completed_debug)

        for side_name, reference_path, reference, landmark_ids in debug_references:
            reference_debug = reference['image'].copy()
            for landmark_id in landmark_ids:
                xy = self._landmark_to_pixel(
                    reference['landmarks'].landmark[landmark_id],
                    reference_debug.shape)
                self._draw_landmark_marker(reference_debug, xy, landmark_id, (0, 128, 255))

            reference_base = os.path.splitext(os.path.basename(reference_path))[0]
            output_path = os.path.join(
                self.debug_output_dir,
                f"{base_name}_reference_edge_{side_name}_{reference_base}_landmarks.png")
            cv2.imwrite(output_path, reference_debug)

        print(f"参照輪郭色補完デバッグ画像を保存しました: {self.debug_output_dir}")

    @staticmethod
    def _draw_landmark_marker(img, xy, landmark_id, color):
        x, y = xy
        cv2.circle(img, (x, y), 7, (255, 255, 255), 1)
        cv2.putText(
            img,
            str(landmark_id),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA)

    @staticmethod
    def _landmark_to_pixel(landmark, image_shape):
        h, w = image_shape[:2]
        x = int(np.clip(round(landmark.x * w), 0, w - 1))
        y = int(np.clip(round(landmark.y * h), 0, h - 1))
        return x, y

    @staticmethod
    def _build_output_path(texture_path):
        base_name = os.path.splitext(os.path.basename(texture_path))[0]
        return f"mqodata/model/{base_name}_reference_edge_color.png"
