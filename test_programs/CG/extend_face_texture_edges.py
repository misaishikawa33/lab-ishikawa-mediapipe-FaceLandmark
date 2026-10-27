#!/usr/bin/env python3
"""Extend face-edge colors in a texture to reduce dark mesh boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def extend_face_texture_edges(
    image: np.ndarray,
    face_landmarks,
    safe_margin: int | None = None,
    extend_margin: int | None = None,
    inpaint_radius: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the extended image and the uint8 mask used for inpainting."""
    height, width = image.shape[:2]
    oval_indices = sorted({
        landmark_id
        for connection in mp.solutions.face_mesh.FACEMESH_FACE_OVAL
        for landmark_id in connection
    })
    points = np.array([
        [
            int(np.clip(round(face_landmarks.landmark[index].x * width), 0, width - 1)),
            int(np.clip(round(face_landmarks.landmark[index].y * height), 0, height - 1)),
        ]
        for index in oval_indices
    ], dtype=np.int32)
    if len(points) < 3:
        raise ValueError("At least three face-oval points are required")

    face_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, cv2.convexHull(points), 255)

    safe_margin = safe_margin if safe_margin is not None else max(3, min(height, width) // 60)
    extend_margin = (
        extend_margin if extend_margin is not None else max(18, min(height, width) // 14)
    )
    if safe_margin < 0 or extend_margin < 0 or inpaint_radius <= 0:
        raise ValueError("Margins must be non-negative and inpaint radius must be positive")

    safe_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (safe_margin * 2 + 1, safe_margin * 2 + 1))
    extend_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (extend_margin * 2 + 1, extend_margin * 2 + 1))
    safe_face_mask = cv2.erode(face_mask, safe_kernel)
    extended_face_mask = cv2.dilate(face_mask, extend_kernel)
    extension_mask = cv2.subtract(extended_face_mask, safe_face_mask)
    extended = cv2.inpaint(image, extension_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return extended, extension_mask


def detect_face_landmarks(image: np.ndarray):
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        raise RuntimeError("No face was detected in the input image")
    return result.multi_face_landmarks[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mask-output", type=Path, default=None)
    parser.add_argument("--safe-margin", type=int, default=None)
    parser.add_argument("--extend-margin", type=int, default=None)
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_edge_extended.png")
    )
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    landmarks = detect_face_landmarks(image)
    extended, extension_mask = extend_face_texture_edges(
        image,
        landmarks,
        safe_margin=args.safe_margin,
        extend_margin=args.extend_margin,
        inpaint_radius=args.inpaint_radius,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), extended):
        raise RuntimeError(f"Failed to write: {output_path}")
    if args.mask_output:
        mask_path = args.mask_output.resolve()
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), extension_mask):
            raise RuntimeError(f"Failed to write: {mask_path}")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    if args.mask_output:
        print(f"Inpaint mask: {args.mask_output.resolve()}")


if __name__ == "__main__":
    main()
