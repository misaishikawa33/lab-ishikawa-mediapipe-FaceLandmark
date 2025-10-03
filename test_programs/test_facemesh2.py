import cv2
import mediapipe as mp
import warnings
import time
import os
import pandas as pd
import sys

warnings.simplefilter("ignore", UserWarning)

f1 = sys.argv[1]
f2 = sys.argv[2]

image_file = "nomask0.jpg"
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

filename1 = os.path.join("results/error1",f1)
df1 = pd.read_csv(filename1, sep = ',', header=None, names=['id', 'rank'])
df1_s = df1.sort_values('rank')
idlist = df1_s['id']
a1 = list(idlist.values)

cnt = 0
if face_mesh.multi_face_landmarks:
    for landmarks in face_mesh.multi_face_landmarks:
        for idx, p in enumerate(landmarks.landmark):
            if idx in a1:
                number = a1.index(idx)
                coord = (int(p.x * w), int(p.y * h))
                cv2.circle(img, coord, 2, (number*3, number*3, 255), 2)
                cnt += 1
            
                
    print("faces:" + str(len(face_mesh.multi_face_landmarks)))
else:
    print("no detect")

print("実行時間："+str(end-start))
cv2.imwrite(os.path.join("output/error1",f2), img)
