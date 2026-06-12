import argparse
import os

import cv2
import numpy as np
from PIL import Image


MARKER_IDS = {
    0: "left_top",
    1: "right_top",
    2: "left_bottom",
    3: "right_bottom",
}


def mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))


def generate_marker(dictionary, marker_id, marker_px, margin_px):
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px, borderBits=1)
    canvas = np.full(
        (marker_px + margin_px * 2, marker_px + margin_px * 2),
        255,
        dtype=np.uint8)
    canvas[margin_px:margin_px + marker_px, margin_px:margin_px + marker_px] = marker
    return canvas


def main():
    parser = argparse.ArgumentParser(description="DICT_4X4_50 ID 0-3 のArUcoマーカーを生成する。")
    parser.add_argument("--output-dir", default="output/aruco_markers", help="出力先ディレクトリ。")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="ArUco辞書名。")
    parser.add_argument("--marker-mm", type=float, default=33.0, help="黒いマーカー本体の1辺。単位はmm。")
    parser.add_argument("--margin-mm", type=float, default=8.0, help="周囲の白余白。単位はmm。")
    parser.add_argument("--dpi", type=int, default=300, help="印刷用DPI。")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dictionary_id = getattr(cv2.aruco, args.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    marker_px = mm_to_px(args.marker_mm, args.dpi)
    margin_px = mm_to_px(args.margin_mm, args.dpi)

    marker_images = []
    for marker_id, name in MARKER_IDS.items():
        marker = generate_marker(dictionary, marker_id, marker_px, margin_px)
        path = os.path.join(args.output_dir, f"aruco_{args.dictionary}_id{marker_id}_{name}.png")
        Image.fromarray(marker).save(path, dpi=(args.dpi, args.dpi))
        marker_images.append((marker_id, name, marker))
        print(path)

    a4_w = mm_to_px(210, args.dpi)
    a4_h = mm_to_px(297, args.dpi)
    sheet = np.full((a4_h, a4_w), 255, dtype=np.uint8)

    cell_w = a4_w // 2
    cell_h = a4_h // 2
    marker_total = marker_px + margin_px * 2

    for index, (marker_id, name, marker) in enumerate(marker_images):
        row = index // 2
        col = index % 2
        x = col * cell_w + (cell_w - marker_total) // 2
        y = row * cell_h + (cell_h - marker_total) // 2 - mm_to_px(8, args.dpi)
        sheet[y:y + marker_total, x:x + marker_total] = marker
        label = f"ID {marker_id} {name}  marker={args.marker_mm:.0f}mm"
        cv2.putText(
            sheet,
            label,
            (x, y + marker_total + mm_to_px(8, args.dpi)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            0,
            2,
            cv2.LINE_AA)

    sheet_path = os.path.join(args.output_dir, f"aruco_{args.dictionary}_id0-3_A4_300dpi.png")
    Image.fromarray(sheet).save(sheet_path, dpi=(args.dpi, args.dpi))
    print(sheet_path)


if __name__ == "__main__":
    main()
