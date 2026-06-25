import argparse
import csv
import math

import cv2
import numpy as np



MARKER_IDS = {
    0: "left_top",
    1: "right_top",
    2: "left_bottom",
    3: "right_bottom",
}


def create_camera_matrix(width, height, focal_length=None):
    if focal_length is None:
        focal_length = width
    return np.array([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1],
    ], dtype=np.float32)


def rotation_vector_to_euler_deg(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(
        rotation_matrix[0, 0] * rotation_matrix[0, 0]
        + rotation_matrix[1, 0] * rotation_matrix[1, 0])

    if sy >= 1e-6:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0.0

    return np.degrees([x, y, z])


def open_capture(args):
    if args.video:
        capture = cv2.VideoCapture(args.video)
    else:
        capture = None
        camera_ids = [args.camera] if args.camera >= 0 else [0, 1, 2, 3, 4]
        for camera_id in camera_ids:
            candidate = cv2.VideoCapture(camera_id)
            if candidate.isOpened():
                capture = candidate
                print(f"camera {camera_id} opened")
                break
            candidate.release()
        if capture is None:
            raise RuntimeError("入力を開けませんでした。")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, 30)

    if not capture.isOpened():
        raise RuntimeError("入力を開けませんでした。")
    return capture


def create_detector(dictionary_name):
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 35
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.04
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def estimate_single_marker_pose(corners, marker_length, camera_matrix, dist_coeffs):
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        return cv2.aruco.estimatePoseSingleMarkers(
            corners,
            marker_length,
            camera_matrix,
            dist_coeffs)[:2]

    half = marker_length / 2.0
    object_points = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)

    rvecs = []
    tvecs = []
    for corner in corners:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            corner[0].astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if success:
            rvecs.append(rvec.reshape(1, 3))
            tvecs.append(tvec.reshape(1, 3))
        else:
            rvecs.append(np.zeros((1, 3), dtype=np.float32))
            tvecs.append(np.zeros((1, 3), dtype=np.float32))

    return np.asarray(rvecs, dtype=np.float32), np.asarray(tvecs, dtype=np.float32)


def estimate_markers(frame, detector, marker_length, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    detections = []

    if ids is None or len(ids) == 0:
        return corners, ids, rejected, detections

    rvecs, tvecs = estimate_single_marker_pose(
        corners,
        marker_length,
        camera_matrix,
        dist_coeffs)

    for marker_id, corner, rvec, tvec in zip(ids.flatten(), corners, rvecs, tvecs):
        if int(marker_id) not in MARKER_IDS:
            continue
        euler = rotation_vector_to_euler_deg(rvec[0])
        detections.append({
            "id": int(marker_id),
            "name": MARKER_IDS[int(marker_id)],
            "corner": corner,
            "rvec": rvec[0],
            "tvec": tvec[0],
            "roll": euler[0],
            "pitch": euler[1],
            "yaw": euler[2],
        })

    return corners, ids, rejected, detections


def draw_detections(frame, corners, ids, detections, camera_matrix, dist_coeffs, marker_length):
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        detected_ids = ",".join(str(int(marker_id)) for marker_id in ids.flatten())
    else:
        detected_ids = "none"

    for detection in detections:
        cv2.drawFrameAxes(
            frame,
            camera_matrix,
            dist_coeffs,
            detection["rvec"],
            detection["tvec"],
            marker_length * 0.5)

        center = detection["corner"][0].mean(axis=0).astype(int)
        text = (
            f'ID {detection["id"]} {detection["name"]} '
            f'yaw={detection["yaw"]:.1f} '
            f'pitch={detection["pitch"]:.1f} '
            f'roll={detection["roll"]:.1f}')
        cv2.putText(
            frame,
            text,
            (center[0] - 80, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA)

    cv2.putText(
        frame,
        f"detected target markers: {len(detections)}/4",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA)
    cv2.putText(
        frame,
        f"all detected IDs: {detected_ids}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA)


def create_csv_writer(path):
    if not path:
        return None, None
    file_obj = open(path, "w", newline="")
    writer = csv.DictWriter(file_obj, fieldnames=[
        "frame",
        "id",
        "name",
        "rvec_x",
        "rvec_y",
        "rvec_z",
        "tvec_x",
        "tvec_y",
        "tvec_z",
        "roll",
        "pitch",
        "yaw",
    ])
    writer.writeheader()
    return file_obj, writer


def create_summary_csv_writer(path):
    if not path:
        return None, None
    file_obj = open(path, "w", newline="")
    writer = csv.DictWriter(file_obj, fieldnames=[
        "frame",
        "marker_count",
        "marker_ids",
        "rvec_x",
        "rvec_y",
        "rvec_z",
        "tvec_x",
        "tvec_y",
        "tvec_z",
        "roll",
        "pitch",
        "yaw",
    ])
    writer.writeheader()
    return file_obj, writer


def summarize_detections(frame_index, detections):
    if not detections:
        return None

    rvecs = np.asarray([detection["rvec"] for detection in detections], dtype=np.float32)
    tvecs = np.asarray([detection["tvec"] for detection in detections], dtype=np.float32)
    rolls = np.asarray([detection["roll"] for detection in detections], dtype=np.float32)
    pitches = np.asarray([detection["pitch"] for detection in detections], dtype=np.float32)
    yaws = np.asarray([detection["yaw"] for detection in detections], dtype=np.float32)
    marker_ids = [detection["id"] for detection in detections]

    return {
        "frame": frame_index,
        "marker_count": len(detections),
        "marker_ids": " ".join(str(marker_id) for marker_id in marker_ids),
        "rvec_x": float(np.mean(rvecs[:, 0])),
        "rvec_y": float(np.mean(rvecs[:, 1])),
        "rvec_z": float(np.mean(rvecs[:, 2])),
        "tvec_x": float(np.mean(tvecs[:, 0])),
        "tvec_y": float(np.mean(tvecs[:, 1])),
        "tvec_z": float(np.mean(tvecs[:, 2])),
        "roll": float(np.mean(rolls)),
        "pitch": float(np.mean(pitches)),
        "yaw": float(np.mean(yaws)),
    }


def main():
    parser = argparse.ArgumentParser(description="4枚のArUcoマーカー検出テスト。")
    parser.add_argument("--video", default=None, help="入力動画ファイル。未指定ならカメラを使用する。")
    parser.add_argument("--camera", type=int, default=-1, help="カメラID。-1なら0から4を順に試す。")
    parser.add_argument("--width", type=int, default=640, help="カメラ入力幅。")
    parser.add_argument("--height", type=int, default=480, help="カメラ入力高さ。")
    parser.add_argument("--marker-length", type=float, default=0.036, help="マーカー1辺の長さ。単位はm。")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="ArUco辞書名。")
    parser.add_argument("--csv", default=None, help="検出結果CSVの保存先。")
    parser.add_argument("--summary-csv", default=None, help="フレームごとの平均姿勢CSVの保存先。")
    parser.add_argument("--no-display", action="store_true", help="画面表示せずに処理する。")
    args = parser.parse_args()

    capture = open_capture(args)
    detector = create_detector(args.dictionary)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    camera_matrix = create_camera_matrix(width, height)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    csv_file, csv_writer = create_csv_writer(args.csv)
    summary_csv_file, summary_csv_writer = create_summary_csv_writer(args.summary_csv)
    frame_index = 0

    print("ArUco marker test")
    print(f"target IDs: {MARKER_IDS}")
    print(f"marker length: {args.marker_length} m")
    print(f"dictionary: {args.dictionary}")
    print("press q to quit")

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            corners, ids, _, detections = estimate_markers(
                frame,
                detector,
                args.marker_length,
                camera_matrix,
                dist_coeffs)

            if frame_index % 30 == 0:
                if ids is None:
                    print(f"frame {frame_index}: no marker detected")
                else:
                    detected_ids = [int(marker_id) for marker_id in ids.flatten()]
                    target_ids = [detection["id"] for detection in detections]
                    print(
                        f"frame {frame_index}: all IDs={detected_ids}, "
                        f"target IDs={target_ids}")

            if csv_writer is not None:
                for detection in detections:
                    csv_writer.writerow({
                        "frame": frame_index,
                        "id": detection["id"],
                        "name": detection["name"],
                        "rvec_x": detection["rvec"][0],
                        "rvec_y": detection["rvec"][1],
                        "rvec_z": detection["rvec"][2],
                        "tvec_x": detection["tvec"][0],
                        "tvec_y": detection["tvec"][1],
                        "tvec_z": detection["tvec"][2],
                        "roll": detection["roll"],
                        "pitch": detection["pitch"],
                        "yaw": detection["yaw"],
                    })

            if summary_csv_writer is not None:
                summary = summarize_detections(frame_index, detections)
                if summary is not None:
                    summary_csv_writer.writerow(summary)

            if not args.no_display:
                draw_detections(
                    frame,
                    corners,
                    ids,
                    detections,
                    camera_matrix,
                    dist_coeffs,
                    args.marker_length)
                cv2.imshow("Aruco marker test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        if csv_file is not None:
            csv_file.close()
        if summary_csv_file is not None:
            summary_csv_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
