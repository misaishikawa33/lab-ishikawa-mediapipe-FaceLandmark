#!/usr/bin/env python3
"""Capture no-mask/masked images, align them, and build paired face OBJ files."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from create_MQO import CreateMQO  # noqa: E402
from convert_mqo_to_obj import parse_mqo, write_obj  # noqa: E402


# Eyes, eyebrows, forehead and nose root: normally visible with a face mask.
STABLE_IDS = sorted({
    6, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    33, 46, 52, 53, 55, 65, 66, 70, 105, 107, 133, 151, 158,
    159, 160, 161, 168, 243, 263, 276, 282, 283, 285, 293,
    295, 296, 300, 334, 336, 362, 383, 384, 385, 386, 417,
})


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="CG01")
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "captures")
    parser.add_argument("--camera", type=int, default=-1, help="-1 tries camera IDs 0..4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--guide-alpha", type=float, default=0.35)
    parser.add_argument(
        "--reuse-captured", action="store_true",
        help="Skip camera capture and reuse source/nomask.jpg and source/mask_raw.jpg",
    )
    parser.add_argument(
        "--alignment", choices=("manual", "auto", "none"), default="manual",
        help="How to align the masked image to the no-mask image",
    )
    parser.add_argument(
        "--no-mirror-preview",
        action="store_true",
        help="Disable the default mirror-like preview (saved images are never mirrored)",
    )
    return parser.parse_args()


def open_camera(camera_id: int, width: int, height: int):
    ids = range(5) if camera_id < 0 else [camera_id]
    for candidate_id in ids:
        capture = cv2.VideoCapture(candidate_id)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return capture, candidate_id
        capture.release()
    raise RuntimeError("Camera IDs 0..4 could not be opened")


def capture_frame(
        capture, title: str, instruction: str, guide=None, alpha=0.35,
        mirror_preview=True):
    while True:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Failed to read a camera frame")
        display = frame.copy()
        if guide is not None:
            resized = cv2.resize(guide, (frame.shape[1], frame.shape[0]))
            display = cv2.addWeighted(display, 1.0 - alpha, resized, alpha, 0.0)
        # Mirror only what the user sees. Keep the captured frame in the
        # camera's original orientation for landmark detection and modeling.
        if mirror_preview:
            display = cv2.flip(display, 1)
        cv2.rectangle(display, (0, 0), (display.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(display, instruction, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(title, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space
            return frame.copy()
        if key in (27, ord("q")):
            raise KeyboardInterrupt


def detect_landmarks(face_mesh, image: np.ndarray) -> np.ndarray:
    result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        raise RuntimeError("MediaPipe could not detect a face")
    height, width = image.shape[:2]
    points = np.array([
        (landmark.x * width, landmark.y * height, landmark.z * width)
        for landmark in result.multi_face_landmarks[0].landmark[:468]
    ], dtype=np.float64)
    if len(points) != 468:
        raise RuntimeError(f"Expected 468 landmarks, detected {len(points)}")
    return points


def save_landmarks(path: Path, points: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["landmark_id", "x_px", "y_px", "z_relative"])
        writer.writerows((index, *point) for index, point in enumerate(points))


def align_masked(reference, masked, reference_points, masked_points):
    source = masked_points[STABLE_IDS, :2].astype(np.float32)
    target = reference_points[STABLE_IDS, :2].astype(np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=5000, confidence=0.995, refineIters=20)
    if matrix is None:
        raise RuntimeError("Face alignment failed")
    height, width = reference.shape[:2]
    aligned = cv2.warpAffine(masked, matrix, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return aligned, matrix, float(inliers.mean()) if inliers is not None else float("nan")


def manual_align(reference: np.ndarray, masked: np.ndarray):
    """Interactively align masked to reference without landmark-based correction."""
    height, width = reference.shape[:2]
    dx = dy = angle = 0.0
    scale = 1.0
    center = (width / 2.0, height / 2.0)
    window = "Manual alignment"
    print("Manual alignment controls:")
    print("  A/D: left/right  W/S: up/down  Q/E: rotate")
    print("  Z/X: shrink/enlarge  R: reset  SPACE/ENTER: accept  ESC: cancel")
    while True:
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        aligned = cv2.warpAffine(
            masked, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT)
        display = cv2.addWeighted(reference, 0.5, aligned, 0.5, 0.0)
        cv2.rectangle(display, (0, 0), (width, 62), (0, 0, 0), -1)
        cv2.putText(display, "WASD:move Q/E:rotate Z/X:scale SPACE:accept",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"dx={dx:+.0f} dy={dy:+.0f} angle={angle:+.2f} scale={scale:.4f}",
                    (8, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("a"):
            dx -= 1
        elif key == ord("d"):
            dx += 1
        elif key == ord("w"):
            dy -= 1
        elif key == ord("s"):
            dy += 1
        elif key == ord("q"):
            angle += 0.25
        elif key == ord("e"):
            angle -= 0.25
        elif key == ord("z"):
            scale = max(0.5, scale - 0.0025)
        elif key == ord("x"):
            scale += 0.0025
        elif key == ord("r"):
            dx = dy = angle = 0.0
            scale = 1.0
        elif key in (13, 32):
            cv2.destroyWindow(window)
            return aligned, matrix, {"dx": dx, "dy": dy, "angle_deg": angle, "scale": scale}
        elif key == 27:
            cv2.destroyWindow(window)
            raise KeyboardInterrupt


def make_model_pair(nomask_path: Path, mask_path: Path, models_dir: Path) -> None:
    model = CreateMQO(
        str(nomask_path), use_edge_texture_extension=False,
        use_cut=False, masked_face=False)
    generated_mqo = (REPO_ROOT / model.model_filename).resolve()
    nomask_mqo = models_dir / "model_nomask.mqo"
    masked_mqo = models_dir / "model_masked.mqo"
    shutil.copy2(generated_mqo, nomask_mqo)
    text = nomask_mqo.read_text(encoding="utf-8")
    masked_text, count = re.subn(r'tex\("[^"]+"\)', f'tex("{mask_path.resolve()}")', text, count=1)
    if count != 1:
        raise RuntimeError("Texture entry was not found in generated MQO")
    masked_mqo.write_text(masked_text, encoding="utf-8")

    for mqo_path, obj_name in ((nomask_mqo, "model_nomask.obj"), (masked_mqo, "model_masked.obj")):
        name, vertices, faces, texture = parse_mqo(mqo_path)
        write_obj(models_dir / obj_name, name, vertices, faces, texture, True, True)


def main() -> None:
    args = arguments()
    root = (args.output_root / args.name).resolve()
    source_dir = root / "source"
    aligned_dir = root / "aligned"
    landmark_dir = root / "landmarks"
    models_dir = root / "models"
    for directory in (source_dir, aligned_dir, landmark_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    nomask_path = source_dir / "nomask.jpg"
    mask_raw_path = source_dir / "mask_raw.jpg"
    if args.reuse_captured:
        nomask = cv2.imread(str(nomask_path))
        masked = cv2.imread(str(mask_raw_path))
        if nomask is None or masked is None:
            raise FileNotFoundError(
                f"Reuse requested but images are missing: {nomask_path}, {mask_raw_path}")
    else:
        capture, camera_id = open_camera(args.camera, args.width, args.height)
        print(f"Camera ID: {camera_id}")
        try:
            nomask = capture_frame(capture, "CG pair capture",
                                   "NO MASK: align face, then press SPACE (Q: quit)",
                                   mirror_preview=not args.no_mirror_preview)
            masked = capture_frame(capture, "CG pair capture",
                                   "WEAR MASK: align with ghost, then press SPACE",
                                   nomask, args.guide_alpha,
                                   mirror_preview=not args.no_mirror_preview)
        finally:
            capture.release()
            cv2.destroyAllWindows()
        cv2.imwrite(str(nomask_path), nomask)
        cv2.imwrite(str(mask_raw_path), masked)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        nomask_points = detect_landmarks(face_mesh, nomask)
        mask_points = detect_landmarks(face_mesh, masked)

    if args.alignment == "manual":
        aligned, matrix, manual_values = manual_align(nomask, masked)
        inlier_ratio = float("nan")
        quality_text = "alignment_method=manual\n" + "\n".join(
            f"{key}={value}" for key, value in manual_values.items()) + "\n"
    elif args.alignment == "auto":
        aligned, matrix, inlier_ratio = align_masked(nomask, masked, nomask_points, mask_points)
        quality_text = (
            "alignment_method=auto\n"
            f"stable_point_inlier_ratio={inlier_ratio:.6f}\n")
    else:
        aligned = masked.copy()
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        inlier_ratio = float("nan")
        quality_text = "alignment_method=none\n"
    aligned_path = aligned_dir / "mask_aligned.png"
    comparison_path = aligned_dir / "alignment_comparison.png"
    cv2.imwrite(str(aligned_path), aligned)
    cv2.imwrite(str(comparison_path), cv2.addWeighted(nomask, 0.5, aligned, 0.5, 0.0))
    save_landmarks(landmark_dir / "nomask_landmarks.csv", nomask_points)
    save_landmarks(landmark_dir / "mask_raw_landmarks.csv", mask_points)
    np.savetxt(aligned_dir / "alignment_affine.csv", matrix, delimiter=",")
    (aligned_dir / "alignment_quality.txt").write_text(quality_text, encoding="utf-8")

    make_model_pair(nomask_path, aligned_path, models_dir)
    print(f"Completed: {root}")
    print(f"Alignment method: {args.alignment}")
    if args.alignment == "auto":
        print(f"Alignment inlier ratio: {inlier_ratio:.3f}")
    print(f"No-mask OBJ: {models_dir / 'model_nomask.obj'}")
    print(f"Masked OBJ: {models_dir / 'model_masked.obj'}")


if __name__ == "__main__":
    main()
