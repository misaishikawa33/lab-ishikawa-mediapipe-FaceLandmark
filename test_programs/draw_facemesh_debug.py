#!/usr/bin/env python3
"""
静止画像に対するMediaPipe FaceMeshの検出結果を描画する。

姿勢推定のずれが、FaceMesh検出点そのもののずれによるものか確認するための
デバッグ画像を出力する。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import mediapipe as mp


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ID_RE = re.compile(r"face(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw MediaPipe FaceMesh debug images.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "test_programs" / "input" / "0622",
        help="入力画像フォルダ。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "test_programs"
            / "output"
            / "pose_eval_pipeline"
            / "0622_surgical"
            / "initial_yaw_flipped"
            / "mediapipe_debug"
            / "maskless"
        ),
        help="FaceMesh描画画像の出力先。",
    )
    parser.add_argument(
        "--draw-ids",
        action="store_true",
        help="ランドマークIDも描画する。全点に番号を出すため確認用画像は見づらくなる。",
    )
    return parser.parse_args()


def collect_images(input_dir: Path) -> list[Path]:
    paths = []
    for path in input_dir.glob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        if IMAGE_ID_RE.search(path.name):
            paths.append(path)
    return sorted(paths, key=lambda path: int(IMAGE_ID_RE.search(path.name).group(1)))


def draw_facemesh(image_bgr, face_landmarks, draw_ids: bool):
    output = image_bgr.copy()
    height, width = output.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    drawing.draw_landmarks(
        image=output,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
    )

    for landmark_id, landmark in enumerate(face_landmarks.landmark):
        x = int(round(landmark.x * width))
        y = int(round(landmark.y * height))
        cv2.circle(output, (x, y), 2, (0, 255, 0), -1, cv2.LINE_AA)
        if draw_ids:
            cv2.putText(
                output,
                str(landmark_id),
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    return output


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"画像が見つかりません: {args.input_dir}")

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"skip: {image_path}")
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(image_rgb)
            output_path = args.output_dir / image_path.name
            if not result.multi_face_landmarks:
                failed = image_bgr.copy()
                cv2.putText(
                    failed,
                    "FaceMesh not detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(output_path), failed)
                print(f"not detected: {output_path}")
                continue

            debug = draw_facemesh(image_bgr, result.multi_face_landmarks[0], args.draw_ids)
            cv2.imwrite(str(output_path), debug)
            print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
