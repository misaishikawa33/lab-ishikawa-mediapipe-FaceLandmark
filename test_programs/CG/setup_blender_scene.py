"""Import the converted face OBJ and create the fixed CG experiment scene.

Run with Blender, not ordinary Python:
  blender --background --python setup_blender_scene.py -- [options]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=Path, default=script_dir / "generated/model.obj")
    parser.add_argument("--output", type=Path, default=script_dir / "generated/face_experiment.blend")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fx", type=float, default=1500.0)
    parser.add_argument("--fy", type=float, default=1500.0)
    # Avoid --cx/--cy because older Blender versions treat --cy as an
    # ambiguous abbreviation of their own --cycles-* command-line options.
    parser.add_argument("--principal-x", dest="cx", type=float, default=640.0)
    parser.add_argument("--principal-y", dest="cy", type=float, default=360.0)
    parser.add_argument("--sensor-width", type=float, default=36.0)
    parser.add_argument("--camera-distance", type=float, default=700.0, help="Model-coordinate units (mm here)")
    parser.add_argument("--light-energy", type=float, default=3000.0)
    parser.add_argument("--world-strength", type=float, default=1.0)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument(
        "--subdivision-level",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Optional render subdivision. Keep 0 for landmark-geometry validation.",
    )
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    # A Blender camera looks along its local -Z axis; local Y is up.
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def import_face(path: Path) -> bpy.types.Object:
    if not path.is_file():
        raise FileNotFoundError(f"OBJ not found: {path}. Run convert_mqo_to_obj.py first.")
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.obj(filepath=str(path), use_image_search=True, axis_forward="-Z", axis_up="Y")
    imported = [obj for obj in bpy.context.scene.objects if obj not in before and obj.type == "MESH"]
    if not imported:
        raise RuntimeError("No mesh was imported from the OBJ")
    face = imported[0]
    face.name = "FaceMesh"
    return face


def smooth_face(face: bpy.types.Object, subdivision_level: int) -> None:
    # Smooth shading changes only interpolated normals, not the 468 vertex
    # positions. This is the safe default for pose/landmark experiments.
    for polygon in face.data.polygons:
        polygon.use_smooth = True

    # Subdivision makes the silhouette smoother but the rendered surface no
    # longer passes exactly through every original landmark. Therefore it is
    # opt-in and disabled for geometric validation by default.
    if subdivision_level:
        modifier = face.modifiers.new(name="RenderSubdivision", type="SUBSURF")
        modifier.subdivision_type = "CATMULL_CLARK"
        modifier.levels = subdivision_level
        modifier.render_levels = subdivision_level
        modifier.show_render = True


def create_head_root(face: bpy.types.Object) -> bpy.types.Object:
    root = bpy.data.objects.new("HeadRoot", None)
    bpy.context.collection.objects.link(root)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 40.0
    face.parent = root
    return root


def create_camera(args: argparse.Namespace) -> bpy.types.Object:
    if not math.isclose(args.fx, args.fy, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("This initial setup requires fx == fy. Pixel-aspect handling can be added later.")
    if not (math.isclose(args.cx, args.width / 2.0) and math.isclose(args.cy, args.height / 2.0)):
        raise ValueError("This initial setup requires the principal point at the image center.")

    data = bpy.data.cameras.new("ExperimentCamera")
    camera = bpy.data.objects.new("ExperimentCamera", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = args.sensor_width
    data.lens = args.fx * args.sensor_width / args.width
    data.shift_x = 0.0
    data.shift_y = 0.0
    data.clip_start = 1.0
    data.clip_end = 5000.0
    # The OBJ importer keeps the face with +90 degrees around X. In that
    # orientation the face is upright in the X-Z plane, its front is -Y, and
    # rotating HeadRoot around Z produces the requested left/right angle.
    # Therefore the camera must be placed on -Y, not above the face on +Z.
    camera.location = (0.0, -args.camera_distance, 0.0)
    look_at(camera, Vector((0.0, 0.0, 0.0)))
    return camera


def create_lighting(args: argparse.Namespace) -> None:
    light_data = bpy.data.lights.new("KeyLight", type="AREA")
    light_data.energy = args.light_energy
    light_data.size = 300.0
    light = bpy.data.objects.new("KeyLight", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (-250.0, -450.0, 300.0)
    look_at(light, Vector((0.0, 0.0, 20.0)))

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.18, 0.18, 0.18, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = args.world_strength


def configure_render(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.exposure = args.exposure
    scene.render.filepath = str(Path(args.output).with_name("camera_check.png"))
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"
    scene.unit_settings.scale_length = 0.001

    # Keep a machine-readable copy of the intended OpenCV intrinsics.
    scene["camera_fx"] = args.fx
    scene["camera_fy"] = args.fy
    scene["camera_cx"] = args.cx
    scene["camera_cy"] = args.cy
    scene["image_width"] = args.width
    scene["image_height"] = args.height
    scene["distortion_model"] = "none"


def main() -> None:
    args = arguments()
    args.obj = args.obj.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    clear_scene()
    face = import_face(args.obj)
    smooth_face(face, args.subdivision_level)
    create_head_root(face)
    camera = create_camera(args)
    create_lighting(args)
    configure_render(args)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))

    print("CG experiment scene created")
    print(f"Blend: {args.output}")
    print(f"Resolution: {args.width} x {args.height}")
    print(f"K: fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")
    print(f"Blender lens: {camera.data.lens:.6f} mm")
    print(f"Smooth shading: enabled; subdivision level: {args.subdivision_level}")
    print(
        f"Lighting: energy={args.light_energy}, "
        f"world={args.world_strength}, exposure={args.exposure}"
    )


if __name__ == "__main__":
    main()
