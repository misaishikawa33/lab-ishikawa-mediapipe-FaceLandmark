import cv2
import numpy as np
import insightface
import time
import os

# load detection model
detector = insightface.model_zoo.get_model("models/det_10g.onnx") # 任意のパスに変更してください
detector.prepare(ctx_id=-1, input_size=(640, 640))

# 入力画像を準備
rgb_img = cv2.imread("nomask0.jpg") # 任意のパスに変更してください

start = time.time()
# 検出
bboxes, kpss = detector.detect(rgb_img, max_num=1)
# 実行時間計測
end = time.time()
for kps in kpss:
    kps_int = [list(map(int,x))for x in kps]
    cv2.circle(rgb_img, kps_int[0], 2, (0,0,255), 4)
    cv2.circle(rgb_img, kps_int[1], 2, (255,0,0), 4)
#    for kp in kps_int[0]:
#        cv2.circle(rgb_img, kp, 4, (0,0,255), 4)
#for bbox in bboxes:
#    bbox_int = [int(x) for x in bbox]
#    cv2.rectangle(rgb_img, (bbox_int[0],bbox_int[1]), (bbox_int[2],bbox_int[3]), (255,255,255),2)
        
print("実行時間："+str(end-start))
cv2.imwrite(os.path.join("others","retinaface_landmark.jpg"), rgb_img)