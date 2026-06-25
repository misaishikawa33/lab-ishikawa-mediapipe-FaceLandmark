#!/usr/bin/env python3
"""
YOLO輪郭補正の有無で、同じ静止画像に対する位置姿勢推定結果を比較する。

出力するCSVには、各画像、各条件の回転行列R、並進ベクトルt、推定角度を保存する。
確認用に、推定姿勢で3Dモデルを入力画像上へ再投影した画像も保存する。
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

import cv2
import mediapipe as mp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from YoloRinkakuCorrector import YoloRinkakuCorrector  # noqa: E402
from create_MQO import CreateMQO  # noqa: E402


DATALIST1 = [
    116, 123, 187, 207, 192, 214, 170, 176,
    148, 152, 6, 9, 10, 151, 168, 249,
    251, 252, 253, 254, 255, 256, 257, 258,
    259, 260, 263, 264, 265, 276, 282, 283,
    284, 285, 286, 293, 295, 296, 297, 298,
    299, 300, 301, 332, 333, 334, 336, 337,
    338, 339, 341, 342, 351, 353, 356, 359,
    362, 368, 372, 373, 374, 380, 381, 382,
    383, 384, 385, 386, 387, 388, 389, 390,
    398, 413, 414, 417, 441, 442, 443, 444,
    445, 446, 463, 464, 465, 466, 467
]

DATALIST2 = [
    345, 352, 376, 433, 367, 364, 378, 400,
    377, 152, 6, 7, 9, 10, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30,
    33, 34, 35, 46, 52, 53, 54, 55,
    56, 63, 65, 66, 67, 68, 69, 70,
    71, 103, 104, 105, 107, 108, 109, 110,
    112, 113, 122, 124, 127, 130, 133, 139,
    143, 144, 145, 151, 153, 154, 155, 156,
    157, 158, 159, 160, 161, 162, 163, 168,
    173, 189, 190, 193, 221, 222, 223, 224,
    225, 226, 243, 244, 245, 246, 247
]

DATALIST3 = [
    6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    33, 34, 35, 46, 52, 53, 54, 55, 56, 63, 65, 66, 67, 68,
    69, 70, 71, 103, 104, 105, 107, 108, 109, 110, 112, 113,
    122, 124, 127, 130, 133, 139, 143, 144, 145, 151, 153, 154,
    155, 156, 157, 158, 159, 160, 161, 162, 163, 168, 173, 189,
    190, 193, 221, 222, 223, 224, 225, 226, 243, 244, 245, 246,
    247, 249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
    263, 264, 265, 276, 282, 283, 284, 285, 286, 293, 295, 296,
    297, 298, 299, 300, 301, 332, 333, 334, 336, 337, 338, 339,
    341, 342, 351, 353, 356, 359, 362, 368, 372, 373, 374, 380,
    381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 398, 413,
    414, 417, 441, 442, 443, 444, 445, 446, 463, 464, 465, 466,
    467
]


POINT_MODES = {
    0: "all_points",
    1: "right_face",
    2: "left_face",
    3: "eye_points",
}

IMAGE_ID_RE = re.compile(r"face(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pose estimation results with and without YOLO contour correction."
    )
    parser.add_argument(
        "--maskless-dir",
        type=Path,
        default=REPO_ROOT / "test_programs" / "input" / "maskless",
        help="マスクなし画像のディレクトリ。",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=REPO_ROOT / "test_programs" / "input" / "mask" / "images" / "surgical",
        help="マスクあり画像のディレクトリ。",
    )
    parser.add_argument(
        "--mask-image-type",
        default="mask_surgical",
        help="CSVや出力フォルダー名に使うマスクあり画像の種類名。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "test_programs" / "output" / "pose_yolo_accuracy.csv",
        help="評価結果CSVの出力先。",
    )
    parser.add_argument(
        "--render-output-dir",
        type=Path,
        default=REPO_ROOT / "test_programs" / "output" / "rendered_pose",
        help="推定姿勢で3Dモデルを再投影した確認画像の出力先ディレクトリ。",
    )
    parser.add_argument(
        "--yolo-debug-output-dir",
        type=Path,
        default=REPO_ROOT / "test_programs" / "output" / "yolo_debug",
        help="YOLO検出結果を重ねた確認画像の出力先ディレクトリ。",
    )
    parser.add_argument(
        "--texture",
        type=Path,
        default=REPO_ROOT / "test_programs" / "input" / "maskless" / "face1.jpg",
        help="3Dモデル基準として使う固定テクスチャ画像のパス。",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=REPO_ROOT / "yolofolder" / "best.pt",
        help="YOLOモデルのパス。",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=1500.0,
        help="PnPで用いる仮想カメラの焦点距離。",
    )
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=20.0,
        help="YOLOあり条件で対応点を切り替えるyaw角度しきい値。",
    )
    parser.add_argument(
        "--switch-yaw-source",
        choices=["ground_truth", "estimated", "estimated_flipped", "tracked_yaw"],
        default="ground_truth",
        help="YOLOあり条件の対応点切り替えに使う角度。tracked_yawは前画像の推定角度を使う。",
    )
    parser.add_argument(
        "--angle-start",
        type=float,
        default=None,
        help="face1に対応する正解角度。指定した場合は単純な連番角度としてCSVに入れる。",
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        default=5.0,
        help="--angle-start指定時に使う画像番号ごとの正解角度の刻み。",
    )
    parser.add_argument(
        "--no-landmark-crop",
        action="store_true",
        help="YOLO入力のランドマーク基準切り抜きを無効にする。",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="確認用のモデル再投影画像を保存しない。",
    )
    parser.add_argument(
        "--no-yolo-debug",
        action="store_true",
        help="YOLO検出結果の確認画像を保存しない。",
    )
    parser.add_argument(
        "--maskless-no-yolo-mode",
        choices=["all", "eye"],
        default="all",
        help="マスクなしYOLOなし条件の姿勢推定に使う対応点。allは全対応点、eyeは目元対応点。",
    )
    parser.add_argument(
        "--yolo-summary-output",
        type=Path,
        default=REPO_ROOT / "test_programs" / "output" / "yolo_usage_summary.csv",
        help="YOLO使用有無と結果をまとめたCSVの出力先。",
    )
    return parser.parse_args()


def collect_images(directory: Path) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        match = IMAGE_ID_RE.search(path.name)
        if match:
            images[int(match.group(1))] = path
    return images


def load_point_data(texture_path: Path):
    mqo = CreateMQO(str(texture_path), use_edge_texture_extension=False)
    all_point_3d = np.asarray(mqo.data, dtype=np.float64)
    sources = {
        0: mqo.datalist,
        1: mqo.datalist1,
        2: mqo.datalist2,
        3: mqo.datalist3,
    }
    modes = {}
    for mode, datalist in sources.items():
        point_3d = np.array([all_point_3d[landmark_id] for landmark_id in datalist], dtype=np.float64)
        modes[mode] = {
            "name": POINT_MODES[mode],
            "datalist": datalist,
            "point_3d": point_3d,
        }
    model_points = all_point_3d
    model_mesh = np.asarray(mqo.mesh_cut if len(mqo.mesh_cut) else mqo.mesh, dtype=np.int32)
    return modes, model_points, model_mesh


def landmarks_to_2d(face_landmarks, datalist: list[int], width: int, height: int) -> np.ndarray:
    points = []
    landmark_count = len(face_landmarks.landmark)
    for landmark_id in datalist:
        if landmark_id >= landmark_count:
            raise ValueError(f"FaceMesh結果にランドマーク{landmark_id}がありません。")
        landmark = face_landmarks.landmark[landmark_id]
        points.append([landmark.x * width, landmark.y * height])
    return np.array(points, dtype=np.float64)


def solve_pose(point_3d: np.ndarray, point_2d: np.ndarray, width: int, height: int, focal_length: float):
    camera_matrix = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeff = np.zeros((4, 1), dtype=np.float64)

    initial_flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)
    success, rvec, tvec = cv2.solvePnP(
        point_3d,
        point_2d,
        camera_matrix,
        dist_coeff,
        useExtrinsicGuess=False,
        flags=initial_flag,
    )
    if not success:
        return None

    success, rvec, tvec = cv2.solvePnP(
        point_3d,
        point_2d,
        camera_matrix,
        dist_coeff,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    axis_transform = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    rotation = axis_transform @ cv2.Rodrigues(rvec)[0]
    translation = axis_transform @ tvec
    transformed_rvec = axis_transform @ rvec
    projection = np.hstack((rotation, translation.reshape(3, 1)))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection)
    yaw = float(euler_angles[1, 0])
    pitch = float(euler_angles[0, 0])
    roll = float(euler_angles[2, 0])

    return {
        "R": rotation,
        "t": translation.reshape(3),
        "rvec": transformed_rvec.reshape(3),
        "raw_rvec": rvec.reshape(3),
        "raw_tvec": tvec.reshape(3),
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }


def estimate_with_mode(face_landmarks, image_shape, mode: int, point_data, focal_length: float):
    height, width = image_shape[:2]
    mode_data = point_data[mode]
    point_2d = landmarks_to_2d(face_landmarks, mode_data["datalist"], width, height)
    return solve_pose(mode_data["point_3d"], point_2d, width, height, focal_length)


def get_point_mode_from_yaw(yaw: float | None, threshold: float) -> int:
    if yaw is None:
        return 3
    if yaw >= threshold:
        return 1
    if yaw <= -threshold:
        return 2
    return 3


def get_rinkaku_mode_from_point_mode(point_mode: int) -> str | None:
    if point_mode == 1:
        return "right"
    if point_mode == 2:
        return "left"
    return None


def make_empty_row(image_id: int, image_type: str, condition: str, image_path: Path, args):
    return {
        "image_id": image_id,
        "ground_truth_yaw": get_ground_truth_yaw(image_id, args),
        "image_type": image_type,
        "condition": condition,
        "image_path": str(image_path),
        "texture_path": str(args.texture),
        "render_path": "",
        "yolo_debug_path": "",
        "face_detected": False,
        "pose_success": False,
        "initial_yaw": "",
        "switch_yaw_source": "",
        "switch_yaw": "",
        "point_mode": "",
        "point_mode_name": "",
        "rinkaku_mode": "",
        "yolo_used": False,
        "yolo_success": False,
        "override_count": 0,
        "yaw": "",
        "pitch": "",
        "roll": "",
        "t_x": "",
        "t_y": "",
        "t_z": "",
        "rvec_x": "",
        "rvec_y": "",
        "rvec_z": "",
        **{f"R_{row}{col}": "" for row in range(3) for col in range(3)},
    }


def get_ground_truth_yaw(image_id: int, args) -> str:
    if args.angle_start is None:
        if image_id == 1:
            return 0.0
        if 2 <= image_id <= 13:
            return (image_id - 1) * 5.0
        if 14 <= image_id <= 25:
            return -(image_id - 13) * 5.0
        return ""
    return args.angle_start + (image_id - 1) * args.angle_step


def get_switch_yaw_from_ground_truth(image_id: int, args) -> float | None:
    ground_truth_yaw = get_ground_truth_yaw(image_id, args)
    if ground_truth_yaw == "":
        return None
    return float(ground_truth_yaw)


def get_switch_yaw(image_id: int, initial_yaw: float | None, args) -> float | None:
    if args.switch_yaw_source == "estimated":
        return initial_yaw
    if args.switch_yaw_source == "estimated_flipped":
        return None if initial_yaw is None else -initial_yaw
    return get_switch_yaw_from_ground_truth(image_id, args)


def get_tracked_switch_yaw(image_id: int, image_type: str, tracked_yaws: dict) -> float | None:
    state = tracked_yaws.setdefault(
        image_type,
        {"anchor": None, "positive": None, "negative": None},
    )
    if image_id == 1:
        return None
    if 2 <= image_id <= 13:
        return state["positive"] if state["positive"] is not None else state["anchor"]
    if 14 <= image_id <= 25:
        return state["negative"] if state["negative"] is not None else state["anchor"]
    return state["positive"]


def update_tracked_yaw(image_id: int, image_type: str, yaw: float | None, tracked_yaws: dict):
    if yaw is None:
        return
    state = tracked_yaws.setdefault(
        image_type,
        {"anchor": None, "positive": None, "negative": None},
    )
    if image_id == 1:
        state["anchor"] = yaw
        state["positive"] = yaw
        state["negative"] = yaw
    elif 2 <= image_id <= 13:
        state["positive"] = yaw
    elif 14 <= image_id <= 25:
        state["negative"] = yaw
    else:
        state["positive"] = yaw


def fill_pose_row(row: dict, pose: dict | None, point_mode: int, point_data, initial_yaw=""):
    row["face_detected"] = True
    row["initial_yaw"] = initial_yaw
    row["point_mode"] = point_mode
    row["point_mode_name"] = point_data[point_mode]["name"]
    if pose is None:
        row["pose_success"] = False
        return row

    row["pose_success"] = True
    row["yaw"] = pose["yaw"]
    row["pitch"] = pose["pitch"]
    row["roll"] = pose["roll"]
    row["t_x"], row["t_y"], row["t_z"] = pose["t"].tolist()
    row["rvec_x"], row["rvec_y"], row["rvec_z"] = pose["rvec"].tolist()
    for r in range(3):
        for c in range(3):
            row[f"R_{r}{c}"] = pose["R"][r, c]
    return row


def render_projected_model(
        bgr_image: np.ndarray,
        pose: dict | None,
        model_points: np.ndarray,
        model_mesh: np.ndarray,
        output_path: Path,
        focal_length: float,
        label: str,
        angle_label: str = ""):
    if pose is None:
        return ""

    height, width = bgr_image.shape[:2]
    camera_matrix = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeff = np.zeros((4, 1), dtype=np.float64)
    projected, _ = cv2.projectPoints(
        model_points,
        pose["raw_rvec"].reshape(3, 1),
        pose["raw_tvec"].reshape(3, 1),
        camera_matrix,
        dist_coeff,
    )
    projected = projected.reshape(-1, 2)

    output = bgr_image.copy()
    overlay = output.copy()
    max_index = len(projected) - 1
    for triangle in model_mesh:
        if len(triangle) < 3:
            continue
        i0, i1, i2 = int(triangle[0]), int(triangle[1]), int(triangle[2])
        if i0 > max_index or i1 > max_index or i2 > max_index:
            continue
        pts = projected[[i0, i1, i2]]
        if not np.all(np.isfinite(pts)):
            continue
        pts_i = np.round(pts).astype(np.int32)
        cv2.polylines(overlay, [pts_i], True, (0, 255, 255), 1, cv2.LINE_AA)

    output = cv2.addWeighted(overlay, 0.85, output, 0.15, 0.0)
    cv2.putText(
        output,
        label,
        (10, max(25, height - 48)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"yaw={pose['yaw']:.2f}, pitch={pose['pitch']:.2f}, roll={pose['roll']:.2f}",
        (10, max(52, height - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if angle_label:
        cv2.putText(
            output,
            angle_label,
            (10, max(79, height - 75)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    return str(output_path)


def render_pose_result(
        bgr_image,
        pose,
        image_id,
        image_type,
        condition,
        model_points,
        model_mesh,
        args,
        pose_point_mode_name="",
        angle_yaw=None,
        angle_point_mode_name=""):
    if args.no_render:
        return ""
    output_path = args.render_output_dir / image_type / condition / f"face{image_id:02d}.jpg"
    point_text = f" pose_points={pose_point_mode_name}" if pose_point_mode_name else ""
    label = f"face{image_id} {image_type} {condition}{point_text}"
    angle_label = ""
    if angle_yaw is not None and angle_point_mode_name:
        angle_label = f"angle_yaw={angle_yaw:.2f} by {angle_point_mode_name}"
    return render_projected_model(
        bgr_image,
        pose,
        model_points,
        model_mesh,
        output_path,
        args.focal_length,
        label,
        angle_label,
    )


def save_yolo_debug_image(
        bgr_image,
        corrector,
        overrides,
        image_id,
        image_type,
        rinkaku_mode,
        yolo_success,
        args):
    if args.no_yolo_debug:
        return ""

    output_path = args.yolo_debug_output_dir / image_type / str(rinkaku_mode) / f"face{image_id:02d}.jpg"
    debug_image = bgr_image.copy()

    if corrector.latest_crop_rect is not None:
        x1, y1, x2, y2 = corrector.latest_crop_rect
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            debug_image,
            "YOLO crop",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    if corrector.latest_mask_contour is not None:
        cv2.drawContours(debug_image, [corrector.latest_mask_contour], -1, (255, 255, 0), 2)

    if corrector.latest_yolo_keypoints:
        keypoint_colors = {
            "chin": (0, 0, 255),
            "nose": (255, 0, 255),
            "left_edge": (0, 255, 0),
            "right_edge": (255, 0, 0),
        }
        for key, color in keypoint_colors.items():
            point = corrector.latest_yolo_keypoints.get(key)
            if point is None:
                continue
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.circle(debug_image, (x, y), 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                debug_image,
                key,
                (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    for landmark_id, (x_px, y_px) in overrides.items():
        x, y = int(round(x_px)), int(round(y_px))
        cv2.circle(debug_image, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            debug_image,
            str(landmark_id),
            (x + 4, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    status = f"YOLO {rinkaku_mode} success={yolo_success} overrides={len(overrides)}"
    cv2.putText(
        debug_image,
        status,
        (10, max(25, debug_image.shape[0] - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if yolo_success else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug_image)
    return str(output_path)


def get_corrector(correctors, width: int, height: int, args):
    key = (width, height)
    if key not in correctors:
        correctors[key] = YoloRinkakuCorrector(
            width=width,
            height=height,
            enabled=True,
            update_interval=1,
            model_path=str(args.yolo_model),
            export_csv=False,
            draw_debug_overlay=False,
            use_landmark_crop=not args.no_landmark_crop,
            use_mask_edge_inpaint=False,
        )
    return correctors[key]


def evaluate_image(
        image_id: int,
        image_type: str,
        image_path: Path,
        face_mesh,
        point_data,
        model_points,
        model_mesh,
        correctors,
        tracked_yaws,
        args):
    bgr_image = cv2.imread(str(image_path))
    rows = []
    no_yolo_row = make_empty_row(image_id, image_type, "no_yolo_eye_points", image_path, args)
    yolo_row = make_empty_row(image_id, image_type, "with_yolo_angle_points", image_path, args)

    if bgr_image is None:
        no_yolo_row["error"] = "image_read_failed"
        yolo_row["error"] = "image_read_failed"
        return [no_yolo_row, yolo_row]

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)
    if not result.multi_face_landmarks:
        no_yolo_row["error"] = "face_not_detected"
        yolo_row["error"] = "face_not_detected"
        return [no_yolo_row, yolo_row]

    face_landmarks = result.multi_face_landmarks[0]
    image_shape = bgr_image.shape
    height, width = image_shape[:2]

    angle_pose = estimate_with_mode(face_landmarks, image_shape, 3, point_data, args.focal_length)
    initial_yaw = angle_pose["yaw"] if angle_pose is not None else None

    if image_type == "maskless":
        no_yolo_mode = 0 if args.maskless_no_yolo_mode == "all" else 3
    else:
        no_yolo_mode = 3
    no_yolo_pose = estimate_with_mode(face_landmarks, image_shape, no_yolo_mode, point_data, args.focal_length)
    fill_pose_row(
        no_yolo_row,
        no_yolo_pose,
        no_yolo_mode,
        point_data,
        initial_yaw="" if initial_yaw is None else initial_yaw,
    )
    no_yolo_row["render_path"] = render_pose_result(
        bgr_image,
        no_yolo_pose,
        image_id,
        image_type,
        "no_yolo_eye_points",
        model_points,
        model_mesh,
        args,
        pose_point_mode_name=point_data[no_yolo_mode]["name"],
        angle_yaw=initial_yaw,
        angle_point_mode_name=point_data[3]["name"],
    )
    rows.append(no_yolo_row)

    if args.switch_yaw_source == "tracked_yaw":
        switch_yaw = get_tracked_switch_yaw(image_id, image_type, tracked_yaws)
    else:
        switch_yaw = get_switch_yaw(image_id, initial_yaw, args)
    point_mode = get_point_mode_from_yaw(switch_yaw, args.yaw_threshold)
    rinkaku_mode = get_rinkaku_mode_from_point_mode(point_mode)
    yolo_landmarks = copy.deepcopy(face_landmarks)
    yolo_success = False
    override_count = 0
    yolo_debug_path = ""
    overrides = {}

    if rinkaku_mode is not None:
        corrector = get_corrector(correctors, width, height, args)
        corrector.landmark_overrides_px = {}
        corrector.landmark_overrides_loaded = False
        yolo_success, overrides = corrector.update_landmark_overrides_from_yolo(
            bgr_image,
            rinkaku_mode=rinkaku_mode,
            face_landmarks=yolo_landmarks,
        )
        override_count = len(overrides)
        yolo_debug_path = save_yolo_debug_image(
            bgr_image,
            corrector,
            overrides,
            image_id,
            image_type,
            rinkaku_mode,
            yolo_success,
            args,
        )
        if yolo_success:
            corrector.apply_landmark_overrides(yolo_landmarks)

    yolo_pose = estimate_with_mode(yolo_landmarks, image_shape, point_mode, point_data, args.focal_length)
    fill_pose_row(
        yolo_row,
        yolo_pose,
        point_mode,
        point_data,
        initial_yaw="" if initial_yaw is None else initial_yaw,
    )
    yolo_row["rinkaku_mode"] = "" if rinkaku_mode is None else rinkaku_mode
    yolo_row["switch_yaw_source"] = args.switch_yaw_source
    yolo_row["switch_yaw"] = "" if switch_yaw is None else switch_yaw
    yolo_row["yolo_used"] = rinkaku_mode is not None
    yolo_row["yolo_success"] = yolo_success
    yolo_row["override_count"] = override_count
    yolo_row["yolo_debug_path"] = yolo_debug_path
    yolo_row["render_path"] = render_pose_result(
        bgr_image,
        yolo_pose,
        image_id,
        image_type,
        "with_yolo_angle_points",
        model_points,
        model_mesh,
        args,
        pose_point_mode_name=point_data[point_mode]["name"],
        angle_yaw=initial_yaw,
        angle_point_mode_name=point_data[3]["name"],
    )
    tracked_update_yaw = yolo_pose["yaw"] if yolo_pose is not None else None
    if args.switch_yaw_source == "tracked_yaw":
        update_tracked_yaw(image_id, image_type, tracked_update_yaw, tracked_yaws)
    rows.append(yolo_row)
    return rows


def write_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "ground_truth_yaw",
        "image_type",
        "condition",
        "image_path",
        "texture_path",
        "render_path",
        "yolo_debug_path",
        "face_detected",
        "pose_success",
        "initial_yaw",
        "switch_yaw_source",
        "switch_yaw",
        "point_mode",
        "point_mode_name",
        "rinkaku_mode",
        "yolo_used",
        "yolo_success",
        "override_count",
        "yaw",
        "pitch",
        "roll",
        "t_x",
        "t_y",
        "t_z",
        "rvec_x",
        "rvec_y",
        "rvec_z",
        "R_00",
        "R_01",
        "R_02",
        "R_10",
        "R_11",
        "R_12",
        "R_20",
        "R_21",
        "R_22",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_yolo_summary(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "ground_truth_yaw",
        "image_type",
        "initial_yaw",
        "switch_yaw_source",
        "switch_yaw",
        "point_mode",
        "point_mode_name",
        "yolo_used",
        "rinkaku_mode",
        "yolo_success",
        "override_count",
        "yaw",
        "pitch",
        "roll",
        "render_path",
        "yolo_debug_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row.get("condition") != "with_yolo_angle_points":
                continue
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main():
    args = parse_args()
    point_data, model_points, model_mesh = load_point_data(args.texture)
    maskless_images = collect_images(args.maskless_dir)
    mask_images = collect_images(args.mask_dir)
    image_ids = sorted(set(maskless_images) | set(mask_images))
    rows = []
    correctors = {}
    tracked_yaws = {}

    if not image_ids:
        raise FileNotFoundError("評価対象画像が見つかりません。")

    print("Pose evaluation test")
    print(f"maskless images: {len(maskless_images)}")
    print(f"masked images: {len(mask_images)}")
    print(f"output: {args.output}")

    face_mesh_module = mp.solutions.face_mesh
    with face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for image_id in image_ids:
            if image_id in maskless_images:
                rows.extend(
                    evaluate_image(
                        image_id,
                        "maskless",
                        maskless_images[image_id],
                        face_mesh,
                        point_data,
                        model_points,
                        model_mesh,
                        correctors,
                        tracked_yaws,
                        args,
                    )
                )
            if image_id in mask_images:
                rows.extend(
                    evaluate_image(
                        image_id,
                        args.mask_image_type,
                        mask_images[image_id],
                        face_mesh,
                        point_data,
                        model_points,
                        model_mesh,
                        correctors,
                        tracked_yaws,
                        args,
                    )
                )

    write_csv(rows, args.output)
    write_yolo_summary(rows, args.yolo_summary_output)
    print(f"saved: {args.output}")
    print(f"saved: {args.yolo_summary_output}")


if __name__ == "__main__":
    main()
