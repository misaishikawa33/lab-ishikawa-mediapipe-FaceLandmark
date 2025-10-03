import cv2
import mediapipe as mp
import warnings
import time
import os

warnings.simplefilter("ignore", UserWarning)
 
image_file = "others/mtcnn3.jpg"
img = cv2.imread(image_file)
h, w, _ = img.shape

face_detection_solution = mp.solutions.face_detection.FaceDetection(
        min_detection_confidence = 0.1)

# 実行時間計測
start = time.time()
face_detection = face_detection_solution.process(img)
end = time.time()

if face_detection.detections:
    for landmarks in face_detection.detections:
        mp.solutions.drawing_utils.draw_detection(img, landmarks)
        # nose_tip = mp.solutions.face_detection.get_key_point(
        #     landmarks, mp.solutions.face_detection.FaceKeyPoint.NOSE_TIP)
        # cv2.circle(img, nose_tip, 2, (0,155,255), 2)
                
    print("faces:" + str(len(face_detection.detections)))
else:
    print("no detect")
    
print("実行時間："+str(end-start))
cv2.imwrite(os.path.join("others","facedetection1.jpg"), img)


