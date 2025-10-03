import cv2
from mtcnn import MTCNN
import warnings
import time
import os

warnings.simplefilter("ignore", UserWarning)
 
image_file = "mask1.jpg"
img = cv2.imread(image_file)

# モデルをダウンロード(初回実行時には時間がかかる)
app = MTCNN()

# 実行時間計測
start = time.time()
detect = app.detect_faces(img)
end = time.time()
if detect:
    bounding_box = detect[0]['box']
    keypoints = detect[0]['keypoints']

    cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)

    print("faces:" + str(len(detect)))
else:
    print("no detect")
    
print("実行時間："+str(end-start))
cv2.imwrite(os.path.join("others","mtcnn1.jpg"),img)


