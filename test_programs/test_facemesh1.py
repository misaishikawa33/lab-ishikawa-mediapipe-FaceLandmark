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
f3 = sys.argv[3]
f4 = sys.argv[4]
f5 = sys.argv[5]
f6 = sys.argv[6]
f7 = sys.argv[7]

image_file = "nomask0.jpg"
img = cv2.imread(image_file)
h, w, _ = img.shape

face_mesh_solution = mp.solutions.face_mesh.FaceMesh(
        static_image_mode = True,
        min_detection_confidence = 0.1,
        min_tracking_confidence = 0.1)

face_mesh = face_mesh_solution.process(img)

filename1 = os.path.join("results/error3",f1)
filename2 = os.path.join("results/error3",f2)
filename3 = os.path.join("results/error3",f3)
filename4 = os.path.join("results/error3",f4)
filename5 = os.path.join("results/error3",f5)
filename6 = os.path.join("results/error3",f6)
filename7 = os.path.join("results/error3",f7)

df1 = pd.read_csv(filename1, sep = ' ', header=None, names=['id', 'norm'])
df2 = pd.read_csv(filename2, sep = ' ', header=None, names=['id', 'norm'])
df3 = pd.read_csv(filename3, sep = ' ', header=None, names=['id', 'norm'])
df4 = pd.read_csv(filename4, sep = ' ', header=None, names=['id', 'norm'])
df5 = pd.read_csv(filename4, sep = ' ', header=None, names=['id', 'norm'])
df6 = pd.read_csv(filename4, sep = ' ', header=None, names=['id', 'norm'])
df7 = pd.read_csv(filename4, sep = ' ', header=None, names=['id', 'norm'])
idlist1 = df1['id']
a1 = list(idlist1.values)
idlist2 = df2['id']
a2 = list(idlist2.values)
idlist3 = df3['id']
a3 = list(idlist3.values)
idlist4 = df4['id']
a4 = list(idlist4.values)
idlist5 = df5['id']
a5 = list(idlist5.values)
idlist6 = df6['id']
a6 = list(idlist6.values)
idlist7 = df7['id']
a7 = list(idlist7.values)


# 順位をファイルに保存したい
filename5 = os.path.join("results/error3","rank_plus.txt") # 二回目以降
filename6 = os.path.join("results/error3","rank_minus.txt") # 二回目以降
filename7 = os.path.join("results/error3","rank_all.txt") # 二回目以降
cols = ['rank']
df5 = pd.DataFrame(columns=cols)

cnt = 0
scale = 255/len(idlist1)
if face_mesh.multi_face_landmarks:
    for landmarks in face_mesh.multi_face_landmarks:
        for idx, p in enumerate(landmarks.landmark):
            if idx in idlist1:
                a1number = a1.index(idx)
                a2number = a2.index(idx)
                a3number = a3.index(idx)
                a4number = a4.index(idx)
                a5number = a5.index(idx)
                a6number = a6.index(idx)
                a7number = a7.index(idx)
                number = (a1number + a2number + a3number + a4number + a5number + a6number + a7number)/7
                # idxのナンバーをrankに加算してファイルに保存
                df = pd.DataFrame({'rank': number}, index=[idx])
                df5 = pd.concat([df5, df])
                coord = (int(p.x * w), int(p.y * h))
                cv2.circle(img, coord, 2, (int(number*scale), int(number*scale), 255), 2)
                cnt += 1

else:
    print("no detect")

df5.to_csv(filename7, header=None)
cv2.imwrite(os.path.join("results/error3","rank_all.png"), img)
