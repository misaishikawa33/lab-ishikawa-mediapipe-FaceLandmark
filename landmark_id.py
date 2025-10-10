#指定した画像に対して、FaceMeshで検出したランドマークを表示し、クリックでランドマークIDを取得するプログラム

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
image_path = 'mqodata/nomask.jpg'
image = cv2.imread(image_path)
if image is None:
    print(f"Could not read {image_path}")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(image_rgb)

if not results.multi_face_landmarks:
    print("No face landmarks detected.")
    exit()

face_landmarks = results.multi_face_landmarks[0]
ih, iw, _ = image.shape
landmark_points = [(lm.x * iw, lm.y * ih) for lm in face_landmarks.landmark]  # keep float coords

# --- Parameters ---
selected_id = None
click_threshold = 20  # in original image coords
radius = 3
window_size = (800, 600)  # fixed display window size (width, height)
zoom = 1.0
zoom_step = 0.1
max_zoom = 5.0
min_zoom = 1.0
zoom_center = (iw / 2, ih / 2)


def zoom_at(img, zoom_factor, center, display_size):
    h, w = img.shape[:2]
    disp_w, disp_h = display_size
    display_aspect = disp_w / disp_h
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)
    crop_aspect = crop_w / crop_h
    if crop_aspect > display_aspect:
        # crop_w too wide, reduce width
        crop_w = int(crop_h * display_aspect)
    else:
        # crop_h too tall, reduce height
        crop_h = int(crop_w / display_aspect)

    # Clamp crop rectangle inside image bounds
    x1 = int(center[0] - crop_w // 2)
    y1 = int(center[1] - crop_h // 2)
    x1 = max(0, min(x1, w - crop_w))
    y1 = max(0, min(y1, h - crop_h))

    crop = img[y1:y1 + crop_h, x1:x1 + crop_w]
    zoomed = cv2.resize(crop, display_size, interpolation=cv2.INTER_LINEAR)
    return zoomed, x1, y1, crop_w, crop_h

def draw_landmarks(img, landmarks, crop_rect, display_size, selected_idx=None):
    x_offset, y_offset, crop_w, crop_h = crop_rect
    w, h = display_size

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    img_copy = img.copy()

    for idx, (lx, ly) in enumerate(landmarks):
        # Check if landmark inside crop region
        if not (x_offset <= lx <= x_offset + crop_w and y_offset <= ly <= y_offset + crop_h):
            continue

        # Map landmark from original image coords to zoomed display coords
        zx = int((lx - x_offset) * w / crop_w)
        zy = int((ly - y_offset) * h / crop_h)

        color = (0, 255, 255)  # yellow
        if idx == selected_idx:
            color = (0, 0, 255)  # red
            cv2.circle(img_copy, (zx, zy), radius + 3, color, 2)
        cv2.circle(img_copy, (zx, zy), radius, color, -1)

        text = str(idx)
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = zx - text_w // 2
        text_y = (zy + text_h // 2) - 4
        cv2.putText(img_copy, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return img_copy

def mouse_callback(event, x, y, flags, param):
    global selected_id, zoom_center, landmark_points, zoom, click_threshold

    if event == cv2.EVENT_LBUTTONDOWN:
        display_w, display_h = window_size
        # Convert click pos in display window back to original image coords
        crop_w = int(iw / zoom)
        crop_h = int(ih / zoom)
        x_offset = int(zoom_center[0] - crop_w // 2)
        y_offset = int(zoom_center[1] - crop_h // 2)
        x_offset = max(0, min(x_offset, iw - crop_w))
        y_offset = max(0, min(y_offset, ih - crop_h))

        orig_x = x_offset + x * crop_w / display_w
        orig_y = y_offset + y * crop_h / display_h

        # Find closest landmark within threshold
        distances = [np.hypot(lx - orig_x, ly - orig_y) for (lx, ly) in landmark_points]
        min_dist = min(distances)
        if min_dist < click_threshold:
            selected_id = distances.index(min_dist)
            print(f"Selected landmark ID: {selected_id}")
            # Center zoom on selected landmark
            zoom_center = (landmark_points[selected_id][0], landmark_points[selected_id][1])

cv2.namedWindow('Zoom Crop Viewer')
cv2.setMouseCallback('Zoom Crop Viewer', mouse_callback)

while True:
    zoomed_img, x_off, y_off, c_w, c_h = zoom_at(image, zoom, zoom_center, window_size)
    display_img = draw_landmarks(zoomed_img, landmark_points, (x_off, y_off, c_w, c_h), window_size, selected_id)
    cv2.imshow('Zoom Crop Viewer', display_img)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('+') or key == ord('='):
        zoom = min(zoom + zoom_step, max_zoom)
    elif key == ord('-') or key == ord('_'):
        zoom = max(zoom - zoom_step, min_zoom)

cv2.destroyAllWindows()
