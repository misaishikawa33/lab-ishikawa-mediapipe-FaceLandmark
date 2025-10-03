import cv2
import mediapipe as mp
import warnings
import time
import os
import pandas as pd
import sys

warnings.simplefilter("ignore", UserWarning)

image_file = "nomask.jpg"
img = cv2.imread(image_file)
h, w, _ = img.shape

face_mesh_solution = mp.solutions.face_mesh.FaceMesh(
        static_image_mode = True,
        min_detection_confidence = 0.1,
        min_tracking_confidence = 0.1)

# 実行時間計測
start = time.time()
face_mesh = face_mesh_solution.process(img)
end = time.time()

idlist = [ 234, 454, 127, 356, 132, 361]

up = [6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 46, 52, 53, 
          54, 55, 56, 63, 65, 66, 67, 68, 69, 70, 71, 103, 104, 105, 107, 108, 109, 
          110, 112, 113, 122, 124, 127, 130, 133, 139, 143, 144, 145, 151, 153, 154, 155, 
          156, 157, 158, 159, 160, 161, 162, 163, 168, 173, 189, 190, 193, 221, 222, 
          223, 224, 225, 226, 243, 244, 245, 246, 247, 249, 251, 252, 253, 254, 255, 256, 257, 258, 
          259, 260, 263, 264, 265, 276, 282, 283, 284, 285, 286, 293, 295, 296, 297, 
          298, 299, 300, 301, 332, 333, 334, 336, 337, 338, 339, 341, 342, 351, 353, 356, 
          359, 362, 368, 372, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 
          389, 390, 398, 413, 414, 417, 441, 442, 443, 444, 445, 446, 463, 464, 465, 466, 467]
right = [6, 8, 9, 10, 151, 168, 249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 263, 264, 265, 276, 282, 283, 284, 285, 286, 293, 295, 296, 297, 298, 299, 300, 301, 332, 333, 334, 336, 337, 338, 339, 341, 342, 351, 353, 356, 359, 362, 368, 372, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 398, 413, 414, 417, 441, 442, 443, 444, 445, 446, 463, 465, 464, 466, 467]
left = [6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 46, 52, 53, 54, 55, 56, 63, 65, 66, 67, 68, 69, 70, 71, 103, 104, 105, 107, 108, 109, 110, 112, 113, 122, 124, 127, 130, 133, 139, 143, 144, 145, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 168, 173, 189, 190, 193, 221, 222, 223, 224,225, 226, 243, 244, 245, 246, 247]
print(len(idlist))
cnt = 0
if face_mesh.multi_face_landmarks:
    for landmarks in face_mesh.multi_face_landmarks:
        for idx, p in enumerate(landmarks.landmark):
            if idx in idlist:
                coord = (int(p.x * w), int(p.y * h))
                cv2.circle(img, coord, 2, (0,0,225), 2)
                cnt += 1
            
                
    print("faces:" + str(len(face_mesh.multi_face_landmarks)))
else:
    print("no detect")

print("実行時間："+str(end-start))
cv2.imwrite(os.path.join("results","landmark_mask.jpg"), img)