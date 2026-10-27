#!/usr/bin/env python3
"""Convert the project's text MQO face mesh to an OBJ/MTL pair.

The converter is intentionally independent of mqoloader because mqoloader
initializes OpenGL.  Vertex order is preserved: MQO vertex i becomes OBJ
vertex i + 1.  Per-corner UVs are written as OBJ texture vertices.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


VERTEX_BLOCK_RE = re.compile(r"^\s*vertex\s+(\d+)\s*\{")
FACE_BLOCK_RE = re.compile(r"^\s*face\s+(\d+)\s*\{")
FACE_RE = re.compile(r"^\s*(\d+)\s+V\(([^)]*)\)(?:\s+M\((\d+)\))?(?:\s+UV\(([^)]*)\))?")
TEXTURE_RE = re.compile(r'tex\("([^"]+)"\)')
OBJECT_RE = re.compile(r'^\s*Object\s+"([^"]+)"')


@dataclass
class Face:
    vertices: list[int]
    material: int
    uvs: list[tuple[float, float]]


def parse_mqo(path: Path) -> tuple[str, list[tuple[float, float, float]], list[Face], Path | None]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    object_name = path.stem
    vertices: list[tuple[float, float, float]] = []
    faces: list[Face] = []
    texture: Path | None = None

    for line in lines:
        match = TEXTURE_RE.search(line)
        if match and texture is None:
            texture = (path.parent / match.group(1)).resolve()

    i = 0
    while i < len(lines):
        object_match = OBJECT_RE.match(lines[i])
        if object_match:
            object_name = object_match.group(1)

        vertex_match = VERTEX_BLOCK_RE.match(lines[i])
        if vertex_match:
            count = int(vertex_match.group(1))
            block = lines[i + 1 : i + 1 + count]
            if len(block) != count:
                raise ValueError("MQO vertex block is shorter than declared")
            for row in block:
                values = row.split()
                if len(values) < 3:
                    raise ValueError(f"Invalid MQO vertex: {row}")
                vertices.append(tuple(map(float, values[:3])))
            i += count + 1
            continue

        face_match = FACE_BLOCK_RE.match(lines[i])
        if face_match:
            count = int(face_match.group(1))
            block = lines[i + 1 : i + 1 + count]
            if len(block) != count:
                raise ValueError("MQO face block is shorter than declared")
            for row in block:
                match = FACE_RE.match(row)
                if not match:
                    raise ValueError(f"Unsupported MQO face: {row}")
                polygon_size = int(match.group(1))
                indices = [int(value) for value in match.group(2).split()]
                if len(indices) != polygon_size:
                    raise ValueError(f"Face size mismatch: {row}")
                material = int(match.group(3) or 0)
                uv_values = [float(value) for value in (match.group(4) or "").split()]
                if uv_values and len(uv_values) != polygon_size * 2:
                    raise ValueError(f"UV size mismatch: {row}")
                uvs = list(zip(uv_values[0::2], uv_values[1::2])) if uv_values else []
                faces.append(Face(indices, material, uvs))
            i += count + 1
            continue
        i += 1

    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    if not faces:
        raise ValueError(f"No faces found in {path}")
    return object_name, vertices, faces, texture


def write_obj(
    output_obj: Path,
    object_name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[Face],
    texture_source: Path | None,
    flip_v: bool,
    reverse_winding: bool,
) -> None:
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    output_mtl = output_obj.with_suffix(".mtl")
    texture_name = None
    if texture_source is not None:
        if not texture_source.is_file():
            raise FileNotFoundError(f"Texture referenced by MQO was not found: {texture_source}")
        texture_name = texture_source.name
        destination = output_obj.parent / texture_name
        if texture_source.resolve() != destination.resolve():
            shutil.copy2(texture_source, destination)

    obj_lines = [
        "# Converted from MQO; vertex order is preserved.",
        f"mtllib {output_mtl.name}",
        f"o {object_name}",
    ]
    obj_lines.extend(f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices)

    # OBJ UV indices are global, whereas MQO stores UVs at each face corner.
    face_records: list[str] = []
    uv_index = 1
    obj_lines.append("usemtl face_material")
    for face in faces:
        order = list(range(len(face.vertices)))
        if reverse_winding:
            order.reverse()
        corner_records = []
        for corner in order:
            vertex_index = face.vertices[corner] + 1
            if face.uvs:
                u, v = face.uvs[corner]
                if flip_v:
                    v = 1.0 - v
                obj_lines.append(f"vt {u:.9f} {v:.9f}")
                corner_records.append(f"{vertex_index}/{uv_index}")
                uv_index += 1
            else:
                corner_records.append(str(vertex_index))
        face_records.append("f " + " ".join(corner_records))
    obj_lines.extend(face_records)
    output_obj.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")

    mtl_lines = [
        "newmtl face_material",
        "Ka 0.200000 0.200000 0.200000",
        "Kd 1.000000 1.000000 1.000000",
        "Ks 0.000000 0.000000 0.000000",
        "d 1.0",
        "illum 1",
    ]
    if texture_name:
        mtl_lines.append(f"map_Kd {texture_name}")
    output_mtl.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=repo_root / "mqodata/model/model.mqo")
    parser.add_argument("--output", type=Path, default=script_dir / "generated/model.obj")
    parser.add_argument("--no-flip-v", action="store_true", help="Do not convert MQO image-space V to OBJ V")
    parser.add_argument("--keep-winding", action="store_true", help="Do not reverse MQO face winding")
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    name, vertices, faces, texture = parse_mqo(source)
    write_obj(output, name, vertices, faces, texture, not args.no_flip_v, not args.keep_winding)
    print(f"Input:    {source}")
    print(f"OBJ:      {output}")
    print(f"MTL:      {output.with_suffix('.mtl')}")
    print(f"Vertices: {len(vertices)}")
    print(f"Faces:    {len(faces)}")
    print(f"Texture:  {texture if texture else 'none'}")


if __name__ == "__main__":
    main()
