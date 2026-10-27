#!/usr/bin/env python3
"""連番画像に対し、選択した対応点方式で顔の位置・姿勢を推定する。"""

from __future__ import annotations

import argparse
import copy
import csv
import re
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from YoloRinkakuCorrector import YoloRinkakuCorrector  # noqa: E402
from test_programs.evaluate_pose_yolo_accuracy import (  # noqa: E402
    estimate_with_mode,
    load_point_data,
)


# 普段変更する設定。CLI引数を指定した場合はCLI側を優先する。
INPUT_ROOT = Path(__file__).resolve().parent / "input"
REFERENCE_INPUT_DIR = INPUT_ROOT / "reference"
MASKED_INPUT_DIR = INPUT_ROOT / "masked"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATASET_NAME = "test1"
AVAILABLE_METHODS = ["all_points", "eye_only", "prior_research", "yolo_corrected"]
METHODS = list(AVAILABLE_METHODS)
ANGLE_START = -45.0
ANGLE_END = 45.0
ANGLE_STEP = 5.0
FOCAL_LENGTH = 1500.0
YAW_THRESHOLD = 15.0
YOLO_MODEL = REPO_ROOT / "yolofolder" / "best.pt"

# YOLOで座標を置き換える輪郭ランドマーク。基準画像のデバッグ表示でも、
# 補正対象が分かるようにこれらの点を赤色で描画する。
YOLO_TARGET_LANDMARKS = {
    "right": {111, 116, 123, 147, 213, 192, 138, 135, 169, 150, 149, 176, 148, 152},
    "left": {340, 345, 352, 376, 411, 427, 416, 434, 364, 394, 369, 400, 377, 152},
    "both": {
        234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
        454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    },
}

METHOD_TO_POINT_MODE = {
    "all_points": 0,
    "eye_only": 4,
    "prior_research": 5,
}

# ユーザー指定の「目元のみ」対応点。現在のdatalist3とは別に固定する。
EYE_ONLY_LANDMARKS = [
    6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 46, 52, 53,
    54, 55, 56, 63, 65, 66, 67, 68, 69, 70, 71, 103, 104, 105, 107, 108, 109,
    110, 112, 113, 122, 124, 127, 130, 133, 139, 143, 144, 145, 151, 153, 154, 155,
    156, 157, 158, 159, 160, 161, 162, 163, 168, 173, 189, 190, 193, 221, 222,
    223, 224, 225, 226, 243, 244, 245, 246, 247, 249, 251, 252, 253, 254, 255, 256,
    257, 258, 259, 260, 263, 264, 265, 276, 282, 283, 284, 285, 286, 293, 295, 296,
    297, 298, 299, 300, 301, 332, 333, 334, 336, 337, 338, 339, 341, 342, 351, 353,
    356, 359, 362, 368, 372, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388,
    389, 390, 398, 413, 414, 417, 441, 442, 443, 444, 445, 446, 463, 464, 465, 466, 467,
]

# ユーザー指定の先行研究対応点。
PRIOR_RESEARCH_LANDMARKS = [
    3, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33,
    46, 47, 51, 52, 53, 54, 55, 56, 63, 65, 66, 67, 68, 69, 70, 71, 103, 104,
    105, 107, 108, 109, 110, 112, 113, 114, 120, 121, 122, 124, 128, 130, 133,
    144, 145, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 168,
    173, 174, 188, 189, 190, 193, 195, 196, 197, 217, 221, 222, 223, 224,
    225, 226, 228, 229, 230, 231, 232, 233, 236, 243, 244, 245, 246, 247, 248,
    249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 276, 277,
    281, 282, 283, 284, 285, 286, 293, 295, 296, 297, 298, 299, 300, 301, 332,
    333, 334, 336, 337, 338, 339, 341, 342, 343, 349, 350, 351, 353, 357, 359,
    362, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 398,
    399, 412, 413, 414, 417, 419, 437, 441, 442, 443, 444, 445, 446, 448, 449,
    450, 451, 452, 453, 456, 463, 464, 465, 466, 467,
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="マスクなし基準とマスクあり4方式の位置姿勢推定結果を評価する。"
    )
    parser.add_argument("--reference-input-dir", type=Path, default=REFERENCE_INPUT_DIR)
    parser.add_argument("--masked-input-dir", type=Path, default=MASKED_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--dataset",
        default=DATASET_NAME,
        help="入力ルート直下のデータセット名。空文字なら入力ディレクトリを直接使用する。",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(AVAILABLE_METHODS),
        default=METHODS,
        help="実行する方式。複数指定可能。",
    )
    parser.add_argument(
        "--texture",
        type=Path,
        default=None,
        help="3D対応点生成用の正面画像。省略時はマスクなし0度画像を使用する。",
    )
    parser.add_argument("--yolo-model", type=Path, default=YOLO_MODEL)
    parser.add_argument("--focal-length", type=float, default=FOCAL_LENGTH)
    parser.add_argument("--angle-start", type=float, default=ANGLE_START)
    parser.add_argument("--angle-end", type=float, default=ANGLE_END)
    parser.add_argument("--angle-step", type=float, default=ANGLE_STEP)
    parser.add_argument("--yaw-threshold", type=float, default=YAW_THRESHOLD)
    parser.add_argument(
        "--draw-landmark-ids",
        action="store_true",
        help="デバッグ画像の全対応点にランドマーク番号も描画する。",
    )
    return parser.parse_args()


def expected_angles(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("angle-stepは正の値にしてください。")
    count_float = (end - start) / step
    count = int(round(count_float))
    if count < 0 or not np.isclose(count_float, count):
        raise ValueError("angle-startからangle-endまでをangle-stepで分割できません。")
    return [start + index * step for index in range(count + 1)]


def collect_numbered_images(input_dir: Path, angles: list[float]) -> list[tuple[int, float, Path]]:
    numbered = []
    for path in input_dir.iterdir() if input_dir.exists() else []:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = NUMBER_RE.search(path.stem)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])

    if len(numbered) != len(angles):
        raise ValueError(
            f"入力画像は{len(angles)}枚必要ですが、{len(numbered)}枚見つかりました: {input_dir}"
        )
    numbers = [number for number, _ in numbered]
    expected_numbers = list(range(numbers[0], numbers[0] + len(numbers))) if numbers else []
    if numbers != expected_numbers:
        raise ValueError(f"写真番号が連番ではありません: {numbers}")
    return [(number, angle, path) for (number, path), angle in zip(numbered, angles)]


def yolo_modes_from_angle(angle: float, threshold: float) -> tuple[int, str]:
    """公称角度を、符号が逆の推定yawに対応する補正モードへ変換する。"""
    if -threshold <= angle <= threshold:
        return 3, "both"
    if angle < -threshold:
        return 2, "left"
    return 1, "right"


def add_custom_point_modes(point_data: dict) -> dict:
    """全点3D座標から、目元のみと先行研究の対応点セットを構築する。"""
    id_to_3d = dict(zip(point_data[0]["datalist"], point_data[0]["point_3d"]))
    for mode, name, landmark_ids in (
            (4, "eye_only", EYE_ONLY_LANDMARKS),
            (5, "prior_research", PRIOR_RESEARCH_LANDMARKS)):
        missing = [landmark_id for landmark_id in landmark_ids if landmark_id not in id_to_3d]
        if missing:
            raise ValueError(f"{name}の3D座標がありません: {missing}")
        point_data[mode] = {
            "name": name,
            "datalist": list(landmark_ids),
            "point_3d": np.asarray([id_to_3d[landmark_id] for landmark_id in landmark_ids]),
        }
    return point_data


def load_point_data_headless(texture: Path):
    """GUIなしOpenCVでもCreateMQOの後処理で失敗しないように読み込む。"""
    original_destroy_all_windows = cv2.destroyAllWindows
    cv2.destroyAllWindows = lambda: None
    try:
        return load_point_data(texture)
    finally:
        cv2.destroyAllWindows = original_destroy_all_windows


def make_corrector(width: int, height: int, args: argparse.Namespace) -> YoloRinkakuCorrector:
    return YoloRinkakuCorrector(
        width=width,
        height=height,
        enabled=True,
        update_interval=1,
        model_path=str(args.yolo_model),
        export_csv=False,
        yaw_threshold_neg=-args.yaw_threshold,
        yaw_threshold_pos=args.yaw_threshold,
        use_front_correction=True,
        draw_debug_overlay=False,
        use_landmark_crop=True,
        use_mask_edge_inpaint=False,
    )


def draw_used_points(
        image: np.ndarray,
        face_landmarks,
        landmark_ids: list[int],
        output_path: Path,
        title: str,
        corrected_ids: set[int] | None = None,
        corrector: YoloRinkakuCorrector | None = None,
        draw_ids: bool = False):
    output = image.copy()
    height, width = output.shape[:2]
    corrected_ids = corrected_ids or set()

    if corrector is not None and corrector.latest_mask_contour is not None:
        cv2.drawContours(output, [corrector.latest_mask_contour], -1, (255, 255, 0), 2)
    if corrector is not None and corrector.latest_crop_rect is not None:
        x1, y1, x2, y2 = corrector.latest_crop_rect
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 128, 0), 1)

    for landmark_id in landmark_ids:
        landmark = face_landmarks.landmark[landmark_id]
        x = int(round(landmark.x * width))
        y = int(round(landmark.y * height))
        color = (0, 0, 255) if landmark_id in corrected_ids else (0, 255, 0)
        cv2.circle(output, (x, y), 3, color, -1, cv2.LINE_AA)
        if draw_ids:
            cv2.putText(
                output, str(landmark_id), (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    cv2.rectangle(output, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(
        output, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise OSError(f"デバッグ画像を保存できませんでした: {output_path}")


def empty_result(
        photo_number: int, angle: float, path: Path, image_type: str, method: str) -> dict:
    return {
        "photo_number": photo_number,
        "nominal_angle_deg": angle,
        "image_path": str(path),
        "image_type": image_type,
        "method": method,
        "face_detected": False,
        "pose_success": False,
        "point_mode": "",
        "point_mode_name": "",
        "used_point_count": 0,
        "rinkaku_mode": "",
        "yolo_success": False,
        "corrected_point_count": 0,
        "rvec_x": "",
        "rvec_y": "",
        "rvec_z": "",
        "opencv_rvec_x": "",
        "opencv_rvec_y": "",
        "opencv_rvec_z": "",
        "t_x": "",
        "t_y": "",
        "t_z": "",
        "yaw_deg": "",
        "pitch_deg": "",
        "roll_deg": "",
        **{f"R_{row}{col}": "" for row in range(3) for col in range(3)},
        "debug_image_path": "",
        "error": "",
    }


def fill_pose(result: dict, pose: dict | None):
    if pose is None:
        result["error"] = "pose_estimation_failed"
        return
    result["pose_success"] = True
    result["rvec_x"], result["rvec_y"], result["rvec_z"] = pose["rvec"].tolist()
    result["opencv_rvec_x"], result["opencv_rvec_y"], result["opencv_rvec_z"] = pose["raw_rvec"].tolist()
    result["t_x"], result["t_y"], result["t_z"] = pose["t"].tolist()
    result["yaw_deg"] = pose["yaw"]
    result["pitch_deg"] = pose["pitch"]
    result["roll_deg"] = pose["roll"]
    for row in range(3):
        for col in range(3):
            result[f"R_{row}{col}"] = pose["R"][row, col]


def evaluate_method(
        method: str,
        photo_number: int,
        angle: float,
        image_path: Path,
        image_type: str,
        bgr_image: np.ndarray,
        base_landmarks,
        point_data,
        correctors: dict[tuple[int, int], YoloRinkakuCorrector],
        args: argparse.Namespace) -> dict:
    result = empty_result(photo_number, angle, image_path, image_type, method)
    result["face_detected"] = True
    landmarks = copy.deepcopy(base_landmarks)
    point_mode = METHOD_TO_POINT_MODE.get(method)
    rinkaku_mode = ""
    corrected_ids: set[int] = set()
    corrector = None

    if method == "yolo_corrected":
        point_mode, rinkaku_mode = yolo_modes_from_angle(angle, args.yaw_threshold)
        height, width = bgr_image.shape[:2]
        corrector_key = (width, height)
        if corrector_key not in correctors:
            correctors[corrector_key] = make_corrector(width, height, args)
        corrector = correctors[corrector_key]
        if not corrector.available:
            result["error"] = "yolo_model_unavailable"
        else:
            # 静止画像同士で前画像の補正座標を持ち越さない。
            corrector.landmark_overrides_px = {}
            corrector.landmark_overrides_loaded = False
            yolo_success, overrides = corrector.update_landmark_overrides_from_yolo(
                bgr_image, rinkaku_mode=rinkaku_mode, face_landmarks=landmarks)
            result["yolo_success"] = yolo_success
            result["corrected_point_count"] = len(overrides)
            if yolo_success:
                corrector.apply_landmark_overrides(landmarks)
                corrected_ids = set(overrides)

    mode_data = point_data[point_mode]
    result["point_mode"] = point_mode
    result["point_mode_name"] = mode_data["name"]
    result["used_point_count"] = len(mode_data["datalist"])
    result["rinkaku_mode"] = rinkaku_mode
    pose = estimate_with_mode(
        landmarks, bgr_image.shape, point_mode, point_data, args.focal_length)
    fill_pose(result, pose)

    debug_path = (
        args.output_dir / "debug_points" / image_type / method /
        f"photo_{photo_number:03d}_angle_{angle:+05.1f}.jpg"
    )
    title = (
        f"method={method} angle={angle:+.1f} mode={mode_data['name']} "
        f"points={len(mode_data['datalist'])}"
    )
    if method == "yolo_corrected":
        title += f" yolo={rinkaku_mode} success={result['yolo_success']}"
    display_corrected_ids = corrected_ids
    if image_type == "reference" and method == "all_points":
        _, display_mode = yolo_modes_from_angle(angle, args.yaw_threshold)
        display_corrected_ids = YOLO_TARGET_LANDMARKS[display_mode]
    draw_used_points(
        bgr_image,
        landmarks,
        mode_data["datalist"],
        debug_path,
        title,
        corrected_ids=display_corrected_ids,
        corrector=corrector,
        draw_ids=args.draw_landmark_ids,
    )
    result["debug_image_path"] = str(debug_path)
    return result


def write_results(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pose_errors(rows: list[dict]) -> list[dict]:
    """各マスクあり姿勢と、同じ写真番号のマスクなし全点姿勢との差を求める。"""
    successful = {
        (row["photo_number"], row["image_type"], row["method"]): row
        for row in rows if row["pose_success"]
    }
    errors = []
    for row in rows:
        if row["image_type"] != "masked" or not row["pose_success"]:
            continue
        reference = successful.get((row["photo_number"], "reference", "all_points"))
        if reference is None:
            continue
        reference_r = np.array([
            [float(reference[f"R_{r}{c}"]) for c in range(3)] for r in range(3)
        ])
        target_r = np.array([
            [float(row[f"R_{r}{c}"]) for c in range(3)] for r in range(3)
        ])
        reference_t = np.array([reference[name] for name in ("t_x", "t_y", "t_z")], dtype=float)
        target_t = np.array([row[name] for name in ("t_x", "t_y", "t_z")], dtype=float)
        relative_rvec, _ = cv2.Rodrigues(target_r @ reference_r.T)
        rotation_l2_rad = float(np.linalg.norm(relative_rvec.reshape(3), ord=2))
        errors.append({
            "photo_number": row["photo_number"],
            "nominal_angle_deg": row["nominal_angle_deg"],
            "reference_image_path": reference["image_path"],
            "masked_image_path": row["image_path"],
            "reference_method": "all_points",
            "target_method": row["method"],
            "rotation_angle_error_deg": float(np.degrees(rotation_l2_rad)),
            "relative_rotation_vector_l2_norm_rad": rotation_l2_rad,
            "rotation_matrix_frobenius_norm": float(
                np.linalg.norm(target_r - reference_r, ord="fro")),
            "translation_l2_norm": float(np.linalg.norm(target_t - reference_t)),
            "reference_rvec_x": reference["rvec_x"],
            "reference_rvec_y": reference["rvec_y"],
            "reference_rvec_z": reference["rvec_z"],
            "target_rvec_x": row["rvec_x"],
            "target_rvec_y": row["rvec_y"],
            "target_rvec_z": row["rvec_z"],
            "reference_t_x": reference["t_x"],
            "reference_t_y": reference["t_y"],
            "reference_t_z": reference["t_z"],
            "target_t_x": row["t_x"],
            "target_t_y": row["t_y"],
            "target_t_z": row["t_z"],
            "yolo_success": row["yolo_success"],
        })
    return errors


def main():
    args = parse_args()
    if args.dataset:
        args.reference_input_dir = args.reference_input_dir / args.dataset
        args.masked_input_dir = args.masked_input_dir / args.dataset
        args.output_dir = args.output_dir / args.dataset
    angles = expected_angles(args.angle_start, args.angle_end, args.angle_step)
    reference_images = collect_numbered_images(args.reference_input_dir, angles)
    masked_images = collect_numbered_images(args.masked_input_dir, angles)
    reference_numbers = [number for number, _, _ in reference_images]
    masked_numbers = [number for number, _, _ in masked_images]
    if reference_numbers != masked_numbers:
        raise ValueError(
            "マスクなし画像とマスクあり画像の写真番号が一致しません: "
            f"reference={reference_numbers}, masked={masked_numbers}"
        )
    zero_candidates = [path for _, angle, path in reference_images if np.isclose(angle, 0.0)]
    texture = args.texture or zero_candidates[0]
    point_data, _, _ = load_point_data_headless(texture)
    point_data = add_custom_point_modes(point_data)
    rows = []
    correctors: dict[tuple[int, int], YoloRinkakuCorrector] = {}

    print(f"reference input: {args.reference_input_dir}")
    print(f"masked input: {args.masked_input_dir}")
    print(f"texture: {texture}")
    print(f"masked methods: {', '.join(args.methods)}")

    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        image_groups = [
            ("reference", reference_images, ["all_points"]),
            ("masked", masked_images, args.methods),
        ]
        for image_type, images, methods in image_groups:
            for photo_number, angle, image_path in images:
                bgr_image = cv2.imread(str(image_path))
                if bgr_image is None:
                    for method in methods:
                        row = empty_result(photo_number, angle, image_path, image_type, method)
                        row["error"] = "image_read_failed"
                        rows.append(row)
                    continue
                result = face_mesh.process(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
                if not result.multi_face_landmarks:
                    for method in methods:
                        row = empty_result(photo_number, angle, image_path, image_type, method)
                        row["error"] = "face_not_detected"
                        rows.append(row)
                    continue
                for method in methods:
                    rows.append(evaluate_method(
                        method, photo_number, angle, image_path, image_type, bgr_image,
                        result.multi_face_landmarks[0], point_data, correctors, args))

    output_csv = args.output_dir / "pose_results.csv"
    write_results(rows, output_csv)
    errors = pose_errors(rows)
    error_csv = args.output_dir / "pose_norm_differences.csv"
    if errors:
        write_results(errors, error_csv)
    print(f"saved: {output_csv}")
    print(f"saved: {error_csv}")
    print(f"saved: {args.output_dir / 'debug_points'}")


if __name__ == "__main__":
    main()
