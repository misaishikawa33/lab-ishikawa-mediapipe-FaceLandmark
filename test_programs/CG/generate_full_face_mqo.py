#!/usr/bin/env python3
"""Restore a complete 468-point MQO from existing project assets.

This does not run MediaPipe. It combines the 468 vertices retained in
model_cut.mqo, all 903 triangles in mesh.dat, and saved normalized landmarks
for UV coordinates.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from convert_mqo_to_obj import parse_mqo


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def latest_landmark() -> Path:
    candidates = sorted(
        path
        for path in (REPO_ROOT / "mqodata/landmark").glob("landmark_*.dat")
        if re.fullmatch(r"landmark_\d{8}\.dat", path.name)
    )
    if not candidates:
        raise FileNotFoundError("No mqodata/landmark/landmark_*.dat file exists")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "mqodata/model/model_cut.mqo")
    parser.add_argument("--mesh", type=Path, default=REPO_ROOT / "mqodata/mesh.dat")
    parser.add_argument("--landmark", type=Path, default=None)
    parser.add_argument("--texture", type=Path, default=REPO_ROOT / "mqodata/model/nomask_edge_extended.png")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "mqodata/model/model.mqo")
    args = parser.parse_args()

    source = args.source.resolve()
    mesh_path = args.mesh.resolve()
    landmark_path = args.landmark.resolve() if args.landmark else latest_landmark().resolve()
    texture = args.texture.resolve()
    output = args.output.resolve()

    _, vertices, _, _ = parse_mqo(source)
    triangles = np.loadtxt(mesh_path, dtype=np.int64).reshape(-1, 3)
    landmarks = np.loadtxt(landmark_path, dtype=np.float64)
    if len(vertices) != 468 or landmarks.shape[0] < 468:
        raise ValueError(f"Expected 468 vertices and at least 468 landmarks, got {len(vertices)}/{landmarks.shape[0]}")
    # Some saved files contain 10 refined iris landmarks after the standard
    # 468 Face Mesh points. mesh.dat addresses only the first 468 points.
    landmarks = landmarks[:468]
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("mesh.dat contains an out-of-range vertex index")
    if not texture.is_file():
        raise FileNotFoundError(texture)

    try:
        texture_ref = texture.relative_to(output.parent).as_posix()
    except ValueError:
        texture_ref = texture.as_posix()

    lines = [
        "Metasequoia Document",
        "Format Text Ver 1.1",
        "",
        "Material 1 {",
        f'\t"mat1" shader(3) col(1.000 1.000 1.000 1.000) dif(1.000) amb(0.200) emi(0.000) spc(0.000) power(5.00) tex("{texture_ref}")',
        "}",
        'Object "obj" {',
        "\tdepth 0",
        "\tfolding 0",
        "\tscale 1 1 1",
        "\trotation 0 0 0",
        "\ttranslation 0 0 0",
        "\tvisible 15",
        "\tshading 1",
        "\tfacet 59.5",
        f"\tvertex {len(vertices)} {{",
    ]
    lines.extend(f"\t\t{x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices)
    lines.extend(["\t}", f"\tface {len(triangles)} {{"])
    for a, b, c in triangles:
        ua, va = landmarks[a, :2]
        ub, vb = landmarks[b, :2]
        uc, vc = landmarks[c, :2]
        lines.append(
            f"\t\t3 V({a} {b} {c}) M(0) "
            f"UV({ua:.9f} {va:.9f} {ub:.9f} {vb:.9f} {uc:.9f} {vc:.9f})"
        )
    lines.extend(["\t}", "}", "Eof", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    print(f"Full face MQO: {output}")
    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(triangles)}")
    print(f"UV source: {landmark_path}")
    print(f"Texture: {texture}")


if __name__ == "__main__":
    main()
