"""Render FaceMesh at fixed Z-axis angle increments from an open Blend scene.

Example:
  blender --background generated/face_experiment.blend \
    --python render_z_angle_sequence.py -- --start -45 --end 45 --step 5
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "output/reference/testCG01")
    parser.add_argument("--start", type=float, default=-45.0)
    parser.add_argument("--end", type=float, default=45.0)
    parser.add_argument("--step", type=float, default=5.0)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument(
        "--blender-angle-sign", type=int, choices=(-1, 1), default=-1,
        help="Convert nominal angle to Blender Z angle. Default -1 matches the existing pose datasets.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest without rendering images")
    return parser.parse_args(argv)


def angle_sequence(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--step must be positive")
    count_float = (end - start) / step
    count = int(round(count_float))
    if count < 0 or not math.isclose(count_float, count, abs_tol=1e-9):
        raise ValueError("The range from --start to --end must be divisible by --step")
    return [start + index * step for index in range(count + 1)]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    head = bpy.data.objects.get("HeadRoot")
    if head is None:
        raise KeyError("HeadRoot was not found in the Blend file")
    if scene.camera is None:
        raise RuntimeError("No active camera is configured")

    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.quality = args.quality
    scene.render.use_file_extension = False

    original_rotation = head.rotation_euler.copy()
    rows = []
    try:
        for photo_number, angle in enumerate(angle_sequence(args.start, args.end, args.step), start=1):
            # The existing pose datasets use the opposite sign to Blender's
            # Z-axis convention for this imported model.
            blender_angle = args.blender_angle_sign * angle
            head.rotation_euler.z = math.radians(blender_angle)
            bpy.context.view_layer.update()

            filename = f"face{photo_number:02d}.jpg"
            output_path = output_dir / filename
            scene.render.filepath = str(output_path)
            world_matrix = head.matrix_world.to_3x3()
            rows.append({
                "photo_number": photo_number,
                "angle_deg": angle,
                "blender_z_angle_deg": blender_angle,
                "rotation_axis": "Z",
                "filename": filename,
                **{
                    f"r{row}{column}": world_matrix[row][column]
                    for row in range(3)
                    for column in range(3)
                },
            })
            print(
                f"[{photo_number:02d}] nominal={angle:+.1f} deg "
                f"Blender-Z={blender_angle:+.1f} deg -> {output_path}"
            )
            if not args.dry_run:
                bpy.ops.render.render(write_still=True)
    finally:
        head.rotation_euler = original_rotation
        bpy.context.view_layer.update()

    manifest = output_dir / "render_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rendered images: {len(rows)}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
