import cv2
import mediapipe as mp
import warnings
import time
import os
import pandas as pd
import sys
import numpy as np

warnings.simplefilter("ignore", UserWarning)

f1 = sys.argv[1]
f2 = sys.argv[2]
f3 = sys.argv[3]

filename_input1 = os.path.join("results/input_image2",f1)
filename_input2 = os.path.join("results/input_image2",f2)


img1 = cv2.imread(filename_input1)
h1, w1, _ = img1.shape

img2 = cv2.imread(filename_input2)
h2, w2, _ = img2.shape

face_mesh_solution = mp.solutions.face_mesh.FaceMesh(
        static_image_mode = True,
        min_detection_confidence = 0,
        min_tracking_confidence = 0)

face_mesh1 = face_mesh_solution.process(img1)
face_mesh2 = face_mesh_solution.process(img2)

nomask = []
error_list = []
tmp = []
point_list = [6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 46, 52, 53, 
            54, 55, 56, 63, 65, 66, 67, 68, 69, 70, 71, 103, 104, 105, 107, 108, 109, 
            110, 112, 113, 122, 124, 127, 130, 133, 139, 143, 144, 145, 151, 153, 154, 155, 
            156, 157, 158, 159, 160, 161, 162, 163, 168, 173, 189, 190, 193, 221, 222, 
            223, 224, 225, 226, 243, 244, 245, 246, 247, 249, 251, 252, 253, 254, 255, 256, 257, 258, 
            259, 260, 263, 264, 265, 276, 282, 283, 284, 285, 286, 293, 295, 296, 297, 
            298, 299, 300, 301, 332, 333, 334, 336, 337, 338, 339, 341, 342, 351, 353, 356, 
            359, 362, 368, 372, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 
            389, 390, 398, 413, 414, 417, 441, 442, 443, 444, 445, 446, 463, 464, 465, 466, 467]
point_list = list(range(468))

if face_mesh1.multi_face_landmarks:
    for landmarks1 in face_mesh1.multi_face_landmarks:
        for idx, p1 in enumerate(landmarks1.landmark):
            # 座標のリストを指定
            if idx in point_list:
                nomask.append([p1.x * w1, p1.y * h1,  p1.z * w1])

cnt = 0     
error_cnt = 0
if face_mesh2.multi_face_landmarks:
    for landmarks2 in face_mesh2.multi_face_landmarks:
        for idx, p2 in enumerate(landmarks2.landmark):
            # 座標のリストを指定
            if idx in point_list:
                error = np.linalg.norm(np.array([p2.x * w2, p2.y * h2, p2.z * w2]) - np.array(nomask[cnt]))
                error_list.append([idx, error])
                if error < 5:
                    error_cnt += 1
                    tmp.append(idx)
                cnt += 1
        mp.solutions.drawing_utils.draw_landmarks(
            img2,
            landmarks2,
            # 描画モード
            mp.solutions.face_mesh.FACEMESH_TESSELATION,
            mp.solutions.drawing_utils.DrawingSpec(1,1),
            mp.solutions.drawing_utils.DrawingSpec(1,1))


norm = sorted(error_list, key=lambda x: x[1])
a1 = [x[0] for x in norm]
a2 = [x[1] for x in norm]
filename_norm = 'results/error3/norm_{}.dat'.format(f3)   
np.savetxt(filename_norm, norm, fmt=["%.0f", "%.4e"])

img3 = cv2.imread("nomask0.jpg")
#img3 = cv2.flip (img3, 1)
h3, w3, _ = img3.shape
face_mesh3 = face_mesh_solution.process(img3)

scale = 255/len(point_list)
if face_mesh3.multi_face_landmarks:
    for landmarks in face_mesh3.multi_face_landmarks:
        for idx, p in enumerate(landmarks.landmark):
            if idx in a1:
                number = a1.index(idx)
                coord = (int(p.x * w3), int(p.y * h3))
                cv2.circle(img3, coord, 2, (int(number*scale), int(number*scale), 255), 2)
                cnt += 1

filename_img = 'results/error3/rank_{}.png'.format(f3)     
filename_lnd = 'results/error3/landmark_{}.png'.format(f3)          
cv2.imwrite(filename_img, img3)
#cv2.imwrite(filename_lnd, img2)
print(error_cnt)
print(tmp)