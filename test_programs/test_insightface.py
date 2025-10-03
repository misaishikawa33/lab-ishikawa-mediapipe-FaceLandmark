import numpy as np
import cv2
from insightface.app import FaceAnalysis
import warnings
import time
import os


warnings.simplefilter("ignore", UserWarning)
 
image_file = "input.png"
img = cv2.imread(image_file)
h, w, _ = img.shape

# モデルをダウンロード(初回実行時には時間がかかる)
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 実行時間計測
# 画像サイズ縮小
start = time.time()
resizeimg = cv2.resize(img,(w, h))
detect = app.get(resizeimg, max_num=1)
end = time.time()
if detect:
    # for face in detect:
        # for xy in face.kps:
        #     print(xy)
        # for xy in face.landmark_2d_106:
        #     print()
    print("faces:" + str(len(detect)))
else:
    print("no detect")
print("実行時間："+str(end-start))
rimg = app.draw_on(img, detect)
cv2.imwrite(os.path.join("others","insightface4.jpg"), rimg)


