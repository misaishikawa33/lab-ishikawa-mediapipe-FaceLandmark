# Application.py
# editor : tagawa kota, sugano yasuyuki
# last updated : 2023/6/9
# overview : 
# Display camera footage and 3D model and face landmark.
# Describe the processing of most of the app


import numpy as np
import datetime
import cv2
from OpenGL.GL import *
import glfw
import mediapipe as mp
import GLWindow
import PoseEstimation as ps
import USBCamera as cam
from mqoloader.loadmqo import LoadMQO

# 未使用
# from ultralytics import YOLO
# import insightface

#
# MRアプリケーションクラス
#
class Application:

    #
    # コンストラクタ
    #
    # @param width    : 画像の横サイズ
    # @param height   : 画像の縦サイズ
    #
    def __init__(self, title, width, height, use_api, draw_landmark, use_facelandmark=False):
        self.width   = width
        self.height  = height
        self.channel = 3

        # カウント用変数
        self.count_img = 0
        self.count_rec = 0
        self.count_func = 0

        # 顔検出に用いる対応点に関する変数(顔全体の場合0)
        self.detect_stable = 0
        # 顔のランドマークを記述するかどうか
        self.draw_landmark = draw_landmark
        # FaceLandmark機能を使用するかどうか
        self.use_facelandmark = use_facelandmark

        # モデル自動スケーリング機能
        self.auto_scale_model = False
        self.model_scale_factor = 1.0
        
        # ランドマーク位置調整機能（右端・左端のみ）
        self.adjust_landmarks = False
        self.alignment_info = None
        
        # 録画用変数
        self.use_record = False # 初期値はFalse
        self.video = None

        #
        # USBカメラの設定
        # USBCameraクラスのインスタンス生成
        #
        self.camera = cam.USBCamera(width, height, use_api)

        #
        # GLウィンドウの設定
        # GLウィンドウクラスのインスタンス生成
        #
        self.glwindow = GLWindow.GLWindow(
            title, 
            width, height, 
            self.display_func, 
            self.keyboard_func)

        #
        # カメラの内部パラメータ(usbカメラ)
        #
        self.focus = 700.0
        self.u0    = width / 2.0
        self.v0    = height / 2.0

        #
        # OpenGLの表示パラメータ
        #
        scale = 0.01
        self.viewport_horizontal = self.u0 * scale
        self.viewport_vertical   = self.v0 * scale
        self.viewport_near       = self.focus * scale
        self.viewport_far        = self.viewport_near * 1.0e+6
        self.modelview           = (GLfloat * 16)()
        self.draw_axis           = False
        self.use_normal          = False
        
        #
        # カメラ姿勢を推定の設定
        # PoseEstimationクラスのインスタンス生成
        #
        self.estimator = ps.PoseEstimation(self.focus, self.u0, self.v0)
        self.point_3D = np.array([])
        self.point_list = np.array([])

        

        #
        # mediapipeを使った顔検出モデル
        # Mediapipe FaceMeshのインスタンス生成
        #
        self.face_mesh = None
        self.face_mesh_solution = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence = 0.25,
            min_tracking_confidence = 0.25)

        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness = 1, 
            circle_radius = 1)
        
        #
        # FaceLandmark追加機能
        #
        if self.use_facelandmark:
            # より高精度なFaceLandmark設定,
            # 近距離model_selection=0, 遠距離model_selection=1
            #model_selectionは信頼度x以上を顔とする

            self.face_landmark_solution = mp.solutions.face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=0.5)
            
            # 追加のランドマーク描画設定
            self.landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
                thickness = 2, 
                circle_radius = 2, 
                color = (0, 255, 0))
            print("FaceLandmark機能が有効化されました")
        
        #
        # マスク着用有無の推論モデルYOLOv8(未使用)
        # train : Yolov8, datasets : face mask dataset(Yolo format)
        # initial_weight : yolov8n.pt , epoch : 200 , image_size : 640
        #
        self.use_mask = False
        # if self.use_mask:
        #     self.mask_model = YOLO("./yolov8n/detect/train/weights/best.pt")
        #     self.mask = False # mask未着用
        # else:
        #     self.mask = True # mask着用
        
        #
        # 高精度顔検出モデルinsightface(未使用)
        #
        self.use_faceanalysis = False
        # if self.use_faceanalysis:
        #     # load detection model
        #     self.detector = insightface.model_zoo.get_model("models/det_10g.onnx")
        #     self.detector.prepare(ctx_id=-1, input_size=(640, 640))
        # else:
        #     self.detect = False
        
    
    #
    # カメラの内部パラメータの設定関数
    # 
    def SetCameraParam(self, focus, u0, v0):
        self.focus = focus
        self.u0    = u0
        self.v0    = v0

    #
    # マスクの着用判別(実行に時間がかかるため、リアルタイムでの使用が難しく未使用)
    #
    # def Yolov8(self):
    #     if self.count_func % 100 == 0:
    #         # 画像に対して顔の占める割合が大きすぎると誤判別するため、リサイズ
    #         image = cv2.cvtColor (self.image, cv2.COLOR_BGR2RGB)
    #         img_resized = cv2.resize(image, dsize=(self.width // 2, self.height //2))
    #         back = np.zeros((self.height, self.width, 3))
    #         back[0:self.height // 2, 0:self.width // 2] = img_resized
    #         # save=Trueで結果を保存
    #         results = self.mask_model(back, max_det=1) 
    #         if(len(results[0]) == 1):
    #             names = results[0].names
    #             # 画像サイズを半分にしているため、座標を2倍してもとのスケールに戻す
    #             cls = results[0].boxes.cls
    #             # conf = results[0].boxes.conf
    #             name = names[abs(int(cls) - 1)]
    #             if name == "no-mask":
    #                 self.mask = False
    #             else:
    #                 self.mask = True
    #         else:
    #             # 検出できなかった場合、self.maskはそのまま
    #             pass
    #     else:
    #         pass
        
    #
    # 顔認識(マスクを着用している場合でも構成度で顔検出を行えるが、実行に時間がかかるため未使用)
    #
    # def Retinaface(self):
    #     if self.use_faceanalysis:
    #         bboxes, kpss = self.detector.detect(self.image, max_num=1)
    #         if len(bboxes) == 1:
    #             self.bbox = bboxes[0]
    #             self.kps = kpss[0]
    #             return True
    #         else:
    #             return False
        
    #
    # カメラ映像を表示するための関数
    # ここに作成するアプリケーションの大部分の処理を書く
    #
    def display_func(self, window):

        # 初回実行
        if self.count_func == 0:
            self.count_func = 1
            glClear(GL_COLOR_BUFFER_BIT)
            return

        # バッファを初期化
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # 画像の読み込み
        success, self.image = self.camera.CaptureImage()
        if not success:
            print("error : video error")
            return
    
        # 描画設定
        self.image.flags.writeable = False
       
        # マスク検出のメソッドを実行
        # if self.use_mask:
        #     self.Yolov8()
        
        #
        # 顔特徴点検出(FaceMesh)を実行
        #
        self.face_mesh = self.face_mesh_solution.process(self.image)
        
        #
        # FaceLandmark追加処理
        #
        if self.use_facelandmark:
            self.face_detection = self.face_landmark_solution.process(self.image)
        
        #
        # 画像の描画を実行
        #
        self.image.flags.writeable = True
        
        # 
        # カメラ姿勢推定前にランドマークを調整
        # (姿勢推定にも調整を反映させる)
        #
        if self.adjust_landmarks and self.use_facelandmark and self.face_mesh.multi_face_landmarks:
            # 調整情報を先に計算
            self.alignment_info = self.calculate_landmark_alignment()
            if self.alignment_info:
                # ランドマークを調整（元のデータを上書き）
                import copy
                for face_landmarks in self.face_mesh.multi_face_landmarks:
                    face_landmarks.landmark[234].x = self.alignment_info['right_ear_target'][0] / self.width
                    face_landmarks.landmark[234].y = self.alignment_info['right_ear_target'][1] / self.height
                    face_landmarks.landmark[454].x = self.alignment_info['left_ear_target'][0] / self.width
                    face_landmarks.landmark[454].y = self.alignment_info['left_ear_target'][1] / self.height

        # ランドマークの描画
        if self.draw_landmark:
            # ランドマークを描画するメソッドを実行
            self.draw_landmarks(self.image)
        
        # FaceLandmark追加描画
        if self.use_facelandmark:
            self.draw_face_detection(self.image)
        
        # ステータス表示を追加
        self.draw_status_info(self.image)

        # 画像を描画するメソッドを実行
        self.glwindow.draw_image(self.image)
        
        # 
        # カメラ姿勢推定
        # 顔のランドマーク検出
        #
        if self.face_mesh.multi_face_landmarks:
            #
            # 座標の正規化用リスト
            #
            point_2D = []
            point_3D = []
            cnt = 0
            #
            # 対応点を指定(顔全体を用いる場合は0)
            #
            if self.detect_stable == 0:
                # print("all")
                point_list = self.point_list
                point_3D = self.point_3D
            elif self.detect_stable == 1:
                # print("upper")
                point_list = self.point_list1
                point_3D = self.point_3D1
            elif self.detect_stable == 2:
                # print("selected")
                point_list = self.point_list2
                point_3D = self.point_3D2
            else:
                point_list = self.point_list
                point_3D = self.point_3D
            
            #
            # 顔の特徴点を取得
            #
            for landmarks in self.face_mesh.multi_face_landmarks:
                for idx, p in enumerate(landmarks.landmark):
                    cnt += 1
                    if idx in point_list:
                        # 画像サイズに合わせて正規化  
                        point_2D.append([p.x * self.width, p.y * self.height])

            #
            # カメラ位置、姿勢計算
            #
            success, vector, angle = self.compute_camera_pose(point_2D, point_3D)
            self.angle = angle
            
            #
            # モデル自動スケーリングが有効な場合、スケールを計算
            #
            if self.auto_scale_model and self.use_facelandmark:
                self.model_scale_factor = self.calculate_model_scale_from_ears()
            else:
                self.model_scale_factor = 1.0
            
            #
            # マスク着用時、モデルを描画
            #
            if success:
                self.draw_model()
    
        else:
            #
            # 検出が安定しない
            #
            print("not detection")    

            
        # 関数実行回数を更新
        self.count_func += 1
        
        # バッファを入れ替えて画面を更新
        glfw.swap_buffers(window)
            
        # 録画している場合画面を保存
        if self.use_record:
            frame = self.save_image()
            self.video.write(frame)

    #
    # モデル描画に関する処理を行う関数
    #
    def draw_model(self, scale_x = 1.0, scale_y = 1.0):
        #
        # モデル表示に関するOpenGLの値の設定
        #
        # 射影行列を選択
        glMatrixMode(GL_PROJECTION)
        # 単位行列
        glLoadIdentity()
        # 透視変換行列を作成            
        glFrustum(-self.viewport_horizontal, self.viewport_horizontal, -self.viewport_vertical, self.viewport_vertical, self.viewport_near, self.viewport_far)
        # モデルビュー行列を選択
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # モデルビュー行列を作成(よくわかってない)
        glLoadMatrixf(self.modelview)

        # 照明をオン
        if self.use_normal:
            # 光のパラメータの設定(光源0,照明位置,照明位置パラメータ)
            glLightfv(GL_LIGHT0, GL_POSITION, self.camera_pos)
            # GL_LIGHTNING(光源0)の機能を有効にする
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)

        model_shift_X = 0.0
        model_shift_Y = 0.0
        model_shift_Z = 0.0
        
        # ランドマーク位置調整が有効な場合、モデルに平行移動を適用
        if self.adjust_landmarks and self.alignment_info:
            # 234(右端)と454(左端)の平均オフセットを計算
            offset_x_234 = self.alignment_info['right_ear_target'][0] - self.alignment_info['right_face_current'][0]
            offset_y_234 = self.alignment_info['right_ear_target'][1] - self.alignment_info['right_face_current'][1]
            offset_x_454 = self.alignment_info['left_ear_target'][0] - self.alignment_info['left_face_current'][0]
            offset_y_454 = self.alignment_info['left_ear_target'][1] - self.alignment_info['left_face_current'][1]
            
            # 平均オフセット(ピクセル単位)
            avg_offset_x = (offset_x_234 + offset_x_454) / 2.0
            avg_offset_y = (offset_y_234 + offset_y_454) / 2.0
            
            # OpenGLの座標系に変換(画像座標→正規化座標→OpenGL座標)
            # X軸: 画像の横方向のオフセット
            model_shift_X = avg_offset_x
            # Y軸: 画像の縦方向のオフセット(OpenGLはY軸が反転)
            model_shift_Y = -avg_offset_y
            # Z軸: 変更なし
            model_shift_Z = 0.0
            
            # print(f"モデル平行移動: X={model_shift_X:.1f}, Y={model_shift_Y:.1f}")
        
        model_scale_X = 1.0 * scale_x * self.model_scale_factor
        model_scale_Y = 1.0 * scale_y * self.model_scale_factor
        model_scale_Z = 1.0 * self.model_scale_factor 
    
        # 世界座標系の描画
        if self.draw_axis:
            mesh_size = 200.0
            mesh_grid = 10.0
            # カメラを平行移動
            glTranslatef(model_shift_X, model_shift_Y, model_shift_Z)
            # 回転(x方向に90度)
            glRotatef(90.0, 1.0, 0.0, 0.0)
            # 世界座標系の軸を描画する関数
            
            # xz平面のグリッドを記述するメソッド
            #self.glwindow.draw_XZ_plane(mesh_size, mesh_grid)
            # カメラをもとに戻す
            glRotatef(90.0, -1.0, 0.0, 0.0)
            glTranslatef(-model_shift_X, -model_shift_Y, -model_shift_Z)


        # 3次元モデルを描画
        glTranslatef(model_shift_X, model_shift_Y, model_shift_Z)
        # 3次元モデルのスケールに変更
        glScalef(model_scale_X, model_scale_Y, model_scale_Z)
        glRotatef(0.0, 1.0, 0.0, 0.0)
        # 3次元モデルを記述(mqoloderクラスのdrawメソッド)
        self.model.draw()

        # 照明をオフ
        if self.use_normal:
            # GL_LIGHTNING(光源0)の機能を無効にする            
            glDisable(GL_LIGHTING)
            glDisable(GL_LIGHT0)
    
        
    #
    # 検出したランドマークを画像上に描画する関数
    #
    def draw_landmarks(self, image):
        if self.face_mesh.multi_face_landmarks:
            for face_landmarks in self.face_mesh.multi_face_landmarks:
                # 通常の描画（既にdisplay_func内で調整済み）
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    face_landmarks,
                    # 描画モード
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    self.drawing_spec,
                    self.drawing_spec)
    
    #
    # FaceLandmark検出結果を画像上に描画する関数（バウンディングボックスなし）
    #
    def draw_face_detection(self, image):
        if self.face_detection.detections:
            for detection in self.face_detection.detections:
                # バウンディングボックスを描画せず、キーポイントのみを描画
                if detection.location_data.relative_keypoints:
                    for idx, keypoint in enumerate(detection.location_data.relative_keypoints):
                        # キーポイントの座標を画像サイズに変換
                        x = int(keypoint.x * self.width)
                        y = int(keypoint.y * self.height)
                        # キーポイントを円で描画
                        cv2.circle(image, (x, y), 
                                   self.landmark_drawing_spec.circle_radius, 
                                   self.landmark_drawing_spec.color, 
                                   self.landmark_drawing_spec.thickness)
                        # キーポイント番号を表示
                        cv2.putText(image, str(idx), (x + 5, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # ランドマーク位置調整が有効な場合、デバッグ表示を追加
        if self.adjust_landmarks and self.alignment_info and self.face_mesh.multi_face_landmarks:
            # FaceMeshのランドマーク取得
            face_landmarks = self.face_mesh.multi_face_landmarks[0]
            
            # 元のランドマーク234と454の位置を描画（赤色の円）
            original_234_x = int(face_landmarks.landmark[234].x * self.width)
            original_234_y = int(face_landmarks.landmark[234].y * self.height)
            original_454_x = int(face_landmarks.landmark[454].x * self.width)
            original_454_y = int(face_landmarks.landmark[454].y * self.height)
            
            cv2.circle(image, (original_234_x, original_234_y), 10, (0, 0, 255), -1)  # 赤色: 元の234
            cv2.circle(image, (original_454_x, original_454_y), 10, (0, 0, 255), -1)  # 赤色: 元の454
            # cv2.putText(image, "234(org)", (original_234_x + 12, original_234_y), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # cv2.putText(image, "454(org)", (original_454_x + 12, original_454_y), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 目標位置（FaceDetectionの耳）を描画（青色の円）
            target_234_x = int(self.alignment_info['right_ear_target'][0])
            target_234_y = int(self.alignment_info['right_ear_target'][1])
            target_454_x = int(self.alignment_info['left_ear_target'][0])
            target_454_y = int(self.alignment_info['left_ear_target'][1])
            
            cv2.circle(image, (target_234_x, target_234_y), 10, (255, 0, 0), -1)  # 青色: 目標234
            cv2.circle(image, (target_454_x, target_454_y), 10, (255, 0, 0), -1)  # 青色: 目標454
            # cv2.putText(image, "Ear4(target)", (target_234_x + 12, target_234_y - 12), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # cv2.putText(image, "Ear5(target)", (target_454_x + 12, target_454_y - 12), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 線で結ぶ
            cv2.line(image, (original_234_x, original_234_y), (target_234_x, target_234_y), 
                    (0, 255, 255), 3)  # 黄色の線
            cv2.line(image, (original_454_x, original_454_y), (target_454_x, target_454_y), 
                    (0, 255, 255), 3)  # 黄色の線
    
    #
    # 耳の位置からモデルのスケールを計算する関数
    #
    def calculate_model_scale_from_ears(self):
        """
        FaceDetectionの耳の位置(キーポイント4,5)に
        FaceMeshのランドマーク234,454が完全に重なるようにスケールを計算
        """
        if not self.use_facelandmark or not hasattr(self, 'face_detection'):
            return 1.0
        
        if not self.face_detection.detections:
            return 1.0
        
        try:
            # FaceDetectionのキーポイントから耳の位置を取得
            detection = self.face_detection.detections[0]
            keypoints = detection.location_data.relative_keypoints
            
            # MediaPipe FaceDetectionのキーポイント:
            # 0: 右目, 1: 左目, 2: 鼻先, 3: 口, 4: 右耳, 5: 左耳
            if len(keypoints) >= 6 and self.face_mesh.multi_face_landmarks:
                right_ear_fd = keypoints[4]  # FaceDetectionの右耳
                left_ear_fd = keypoints[5]   # FaceDetectionの左耳
                
                # FaceMeshのランドマーク取得
                landmarks = self.face_mesh.multi_face_landmarks[0]
                # ランドマーク234: 顔の右端（右耳付近）
                # ランドマーク454: 顔の左端（左耳付近）
                right_face_fm = landmarks.landmark[234]
                left_face_fm = landmarks.landmark[454]

                # FaceDetectionの耳の間の距離（正規化座標）
                ear_distance_fd = abs(left_ear_fd.x - right_ear_fd.x)
                
                # FaceMeshのランドマーク234-454の間の距離（正規化座標）
                face_width_fm = abs(left_face_fm.x - right_face_fm.x)
                
                if face_width_fm > 0:
                    # 水平方向のスケール係数
                    # FaceMeshの234-454がFaceDetectionの耳の位置に重なるように
                    scale_factor = ear_distance_fd / face_width_fm
                    
                    # スケール係数が極端にならないように制限
                    scale_factor = max(0.3, min(scale_factor, 3.0))
                    
                    # print(f"自動スケール: {scale_factor:.2f}")
                    
                    return scale_factor
        except Exception as e:
            print(f"スケール計算エラー: {e}")
        
        return 1.0
    
    #
    # ランドマーク234と454を耳の位置に合わせる調整情報を計算する関数
    #
    def calculate_landmark_alignment(self):
        """
        FaceDetectionの耳の位置(キーポイント4,5)に
        FaceMeshのランドマーク234,454が重なるように調整情報を計算
        """
        if not self.use_facelandmark or not hasattr(self, 'face_detection'):
            return None
        
        if not self.face_detection.detections:
            return None
        
        try:
            # FaceDetectionのキーポイントから耳の位置を取得
            detection = self.face_detection.detections[0]
            keypoints = detection.location_data.relative_keypoints
            
            # MediaPipe FaceDetectionのキーポイント:
            # 0: 右目, 1: 左目, 2: 鼻先, 3: 口, 4: 右耳, 5: 左耳
            if len(keypoints) >= 6 and self.face_mesh.multi_face_landmarks:
                right_ear_fd = keypoints[4]  # FaceDetectionの右耳
                left_ear_fd = keypoints[5]   # FaceDetectionの左耳
                
                # FaceMeshのランドマーク取得
                landmarks = self.face_mesh.multi_face_landmarks[0]
                # ランドマーク234: 顔の右端（右耳付近）
                # ランドマーク454: 顔の左端（左耳付近）
                right_face_fm = landmarks.landmark[234]
                left_face_fm = landmarks.landmark[454]
                
                # ピクセル座標に変換
                right_ear_fd_px = (right_ear_fd.x * self.width, right_ear_fd.y * self.height)
                left_ear_fd_px = (left_ear_fd.x * self.width, left_ear_fd.y * self.height)
                right_face_fm_px = (right_face_fm.x * self.width, right_face_fm.y * self.height)
                left_face_fm_px = (left_face_fm.x * self.width, left_face_fm.y * self.height)
                
                # 距離を計算
                # offset_234_x = right_ear_fd_px[0] - right_face_fm_px[0]
                # offset_234_y = right_ear_fd_px[1] - right_face_fm_px[1]
                # offset_454_x = left_ear_fd_px[0] - left_face_fm_px[0]
                # offset_454_y = left_ear_fd_px[1] - left_face_fm_px[1]
                
                alignment_info = {
                    'right_ear_target': right_ear_fd_px,
                    'left_ear_target': left_ear_fd_px,
                    'right_face_current': right_face_fm_px,
                    'left_face_current': left_face_fm_px
                }
                
                # print(f"========== ランドマーク位置調整デバッグ情報 ==========")
                # print(f"FaceMesh 234 (元): ({right_face_fm_px[0]:.1f}, {right_face_fm_px[1]:.1f})")
                # print(f"FaceDetection 耳4 (目標): ({right_ear_fd_px[0]:.1f}, {right_ear_fd_px[1]:.1f})")
                # print(f"オフセット234: X={offset_234_x:.1f}px, Y={offset_234_y:.1f}px")
                # print(f"")
                # print(f"FaceMesh 454 (元): ({left_face_fm_px[0]:.1f}, {left_face_fm_px[1]:.1f})")
                # print(f"FaceDetection 耳5 (目標): ({left_ear_fd_px[0]:.1f}, {left_ear_fd_px[1]:.1f})")
                # print(f"オフセット454: X={offset_454_x:.1f}px, Y={offset_454_y:.1f}px")
                # print(f"===================================================")
                
                return alignment_info
        except Exception as e:
            print(f"ランドマーク位置調整計算エラー: {e}")
        
        return None
        
    #
    # キー関数
    #
    def keyboard_func(self, window, key, scancode, action, mods):
        # Qで終了
        if key == glfw.KEY_Q:
            if self.use_record:
                print("録画を終了します")
                self.use_record = False
            # window_should_closeフラグをセットする。
            glfw.set_window_should_close(self.glwindow.window, GL_TRUE)

        # Sで画像の保存
        if action == glfw.PRESS and key == glfw.KEY_S:
            if self.use_record:
                print("録画実行中です...録画を終了してから画像の保存を実行できます")
            else:
                # 画像を保存する関数を実行
                self.save_image()
                # ランドマークを保存する関数を実行
                # self.save_landmarks()
                # 回転行列、並進ベクトルを保存するフラッグを立てる
                #self.flag_save_matrix = 0
                # 画像カウントを+1する
                self.count_img += 1

        
        # Rで画面録画開始
        if action == glfw.PRESS and key == glfw.KEY_R:
            if self.use_record == False:
                # 録画用変数をTrueに
                self.use_record = True
                #　録画を保存する関数を実行
                self.video = self.save_record()
                self.count_rec += 1
            else:
                print("録画を終了します")
                self.use_record = False
        
        # Pで対応点を変更        
        if action == glfw.PRESS and key == glfw.KEY_P:
            if self.detect_stable == 0:
                self.detect_stable = 1
                print("対応点をモード1(顔上部)に変更")
            elif self.detect_stable == 1:
                self.detect_stable = 2
                print("対応点をモード2(ずれが小さいランドマーク選択)に変更")
            elif self.detect_stable == 2:
                self.detect_stable = 0
                print("対応点をモード0(顔全体)に変更")
            else:
                pass
        
        # FでFaceLandmark機能のON/OFF切り替え
        if action == glfw.PRESS and key == glfw.KEY_F:
            self.use_facelandmark = not self.use_facelandmark
            if self.use_facelandmark:
                # FaceLandmark機能を有効化
                if not hasattr(self, 'face_landmark_solution'):
                    self.face_landmark_solution = mp.solutions.face_detection.FaceDetection(
                        model_selection=0, 
                        min_detection_confidence=0.5)
                    self.landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
                        thickness = 2, 
                        circle_radius = 2, 
                        color = (0, 255, 0))
                print("FaceLandmark機能を有効化しました")
            else:
                print("FaceLandmark機能を無効化しました")
        
        # Aでモデル自動スケーリングのON/OFF切り替え
        if action == glfw.PRESS and key == glfw.KEY_A:
            if not self.use_facelandmark:
                print("FaceLandmark機能を有効化してください（Fキー）")
            else:
                self.auto_scale_model = not self.auto_scale_model
                if self.auto_scale_model:
                    print("モデル自動スケーリングを有効化しました")
                else:
                    self.model_scale_factor = 1.0
                    print("モデル自動スケーリングを無効化しました")
        
        # Dでランドマーク位置調整のON/OFF切り替え
        if action == glfw.PRESS and key == glfw.KEY_D:
            if not self.use_facelandmark:
                print("FaceLandmark機能を有効化してください（Fキー）")
            else:
                self.adjust_landmarks = not self.adjust_landmarks
                if self.adjust_landmarks:
                    print("ランドマーク位置調整を有効化しました（ランドマーク234,454を耳の位置に合わせます）")
                else:
                    self.alignment_info = None
                    print("ランドマーク位置調整を無効化しました")
        
        # MでFaceMesh描画のON/OFF切り替え
        if action == glfw.PRESS and key == glfw.KEY_M:
            self.draw_landmark = not self.draw_landmark
            if self.draw_landmark:
                print("FaceMesh描画を有効化しました")
            else:
                print("FaceMesh描画を無効化しました")

    #
    # モデル設定
    #
    def display(self, model_filename):
        #
        # 3次元モデルの読み込み
        #   (OpenGLのウィンドウを作成してからでないとテクスチャが反映されない)
        #
        msg = 'Loading %s ...' % model_filename
        print(msg)
        #
        # 第3引数をTrueにすると面の法線計算を行い、陰影がリアルに描画されます
        # その代わりに計算にかなり時間がかかります
        #
        self.use_normal = False
        model_scale = 10.0
        model = LoadMQO(model_filename, model_scale, self.use_normal)
        print('Done.')
        self.set_mqo_model(model)
        
    #
    # 画像を保存する関数
    #
    def save_image(self):
        today = str(datetime.date.today()).replace('-','')
        filename = 'output/images/image_{}-{}.png'.format(today, self.count_img)
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # バッファを読み込む(画面を読み込む)
        glReadBuffer(GL_FRONT)
        # ピクセルを読み込む
        glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, image.data)
        image = cv2.cvtColor (image, cv2.COLOR_BGR2RGB)
        image = cv2.flip (image, 0)
        if self.use_record:
            return image
        else:
            # 画像を保存
            print("画像を保存します..." + filename)
            cv2.imwrite(filename, image)
        
    #
    # 画面録画を保存する関数
    #
    def save_record(self):
        today = str(datetime.date.today()).replace('-','')
        filename = 'output/videos/video_{}-{}.mp4'.format(today, self.count_rec)
        video = self.camera.SaveRecord(filename)
        print("録画を開始します..." + filename)
        return video
    
    #
    # mediapipeで検出した顔のランドマーク座標を出力する関数
    #
    def save_landmarks(self, add = False, landmark = 0, txt = None):
        today = str(datetime.date.today()).replace('-','')
        filename = 'output/landmarks/landmarks_{}_{}.dat'.format(today, self.count_img)
        output = open(filename, mode='w')
        if self.face_mesh.multi_face_landmarks:
            for landmarks in self.face_mesh.multi_face_landmarks:
                # enumerate()...オブジェクトの要素とインデックス番号を取得
                for idx, p in enumerate(landmarks.landmark):
                    # 座標のリストを指定
                    if idx in self.point_list:
                        text = str(idx) + ',' + str(p.x * self.width) + ',' + str(p.y * self.height) + ',' + str(p.z * self.width) + '\n'
                        # text = str(p.x * self.width) + ',' + str(p.y * self.height) + '\n'
                        output.write(text)
                        
        output.close()

    #
    # カメラ姿勢を計算する関数
    #
    def compute_camera_pose(self, point_2D, point_3D):
        point_2D = np.array(point_2D)
        point_3D = np.array(point_3D)
        # カメラ姿勢を計算
        # PoseEstimationクラスのcompute_camera_poseメソッドを実行
        success, R, t, r = self.estimator.compute_camera_pose(
            point_3D, point_2D, use_objpoint = True)
    
        if success:
            # 世界座標系に対するカメラ位置を計算
            # この位置を照明位置として使用
            if self.use_normal:
                # カメラ位置姿勢計算
                pos = -R.transpose().dot(t)
                self.camera_pos = np.array([pos[0], pos[1], pos[2], 1.0], dtype = "double")

            self.generate_modelview(R,t)
            
            # 顔の方向ベクトルを計算
            # PoseEstimationクラスのcompute_head_vectorメソッドを実行
            vector = self.estimator.compute_head_vector()
            # 顔のオイラー角を計算
            # PoseEstimationクラスのcompute_head_angleメソッドを実行
            angle = self.estimator.compute_head_angle(R, t)
            # # 行列の値をファイルに保存
            # if self.flag_save_matrix == 1:
            #     today = str(datetime.date.today()).replace('-','')
            #     filename = 'output/images/matrix_{}_{}.dat'.format(today, self.count_img)
            #     output = open(filename, mode='a')
            #     output.write(str(np.linalg.norm(r)))
            #     output.write(",")
            #     output.write(str(np.linalg.norm(t)))
            #     output.write(",")
            #     output.write(str(vector))
            #     output.write(",")
            #     output.write(str(angle))
            #     output.write("\n")
            #     output.close()
            return success, vector, angle
            
        else:
            vector = None
            angle = None
            return success, vector, angle
    
    #
    # モデルビュー行列を生成
    #
    def generate_modelview(self, R, t):
        # OpenGLで使用するモデルビュー行列を生成
            self.modelview[0] = R[0][0]
            self.modelview[1] = R[1][0]
            self.modelview[2] = R[2][0]
            self.modelview[3] = 0.0
            self.modelview[4] = R[0][1]
            self.modelview[5] = R[1][1]
            self.modelview[6] = R[2][1]
            self.modelview[7] = 0.0
            self.modelview[8] = R[0][2]
            self.modelview[9] = R[1][2]
            self.modelview[10] = R[2][2]
            self.modelview[11] = 0.0
            self.modelview[12] = t[0]
            self.modelview[13] = t[1]
            self.modelview[14] = t[2]
            self.modelview[15] = 1.0
      
      
    #
    # セッター
    #  
    # 三次元データをセット(対応点全て)
    def set_3D_point(self, point_3D, point_list):
        self.point_3D = point_3D
        self.point_list = point_list
        self.estimator.ready = True
    
    # 三次元データをセット(一部の対応点)
    def set_3D_point_1(self, point_3D, point_list):
        self.point_3D1 = point_3D
        self.point_list1 = point_list       
    def set_3D_point_2(self, point_3D, point_list):
        self.point_3D2 = point_3D
        self.point_list2 = point_list 

    # ３次元モデルをセット
    def set_mqo_model(self, model):
        self.model = model
    
    # 入力画像をセット
    def set_image(self, image):
        image = cv2.cvtColor (image, cv2.COLOR_BGR2RGB)
        self.image = image

    #
    # 現在の状態を画面に表示する関数
    #
    def draw_status_info(self, image):
        """
        画面右上に現在の機能ON/OFF状態を表示
        """
        # 背景の半透明ボックスを描画（見やすくするため）
        # 右上に配置
        box_x1 = self.width - 410  # 右端から410ピクセル左
        box_y1 = 10
        box_x2 = self.width - 10   # 右端から10ピクセル左
        box_y2 = 180  # 行を追加したので高さを増やす
        
        overlay = image.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # テキストのX座標（ボックスの左端から少し右）
        text_x = box_x1 + 10
        y_offset = 30
        line_height = 30
        
        # タイトル
        cv2.putText(image, "=== Status ===", (text_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # FaceMesh描画の状態 (Mキー)
        mesh_status = "ON" if self.draw_landmark else "OFF"
        mesh_color = (0, 255, 0) if self.draw_landmark else (128, 128, 128)
        cv2.putText(image, f"[M] FaceMesh Draw: {mesh_status}", (text_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mesh_color, 1)
        y_offset += line_height
        
        # FaceLandmark機能の状態 (Fキー)
        face_landmark_status = "ON" if self.use_facelandmark else "OFF"
        face_landmark_color = (0, 255, 0) if self.use_facelandmark else (128, 128, 128)
        cv2.putText(image, f"[F] FaceLandmark: {face_landmark_status}", (text_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_landmark_color, 1)
        y_offset += line_height
        
        # 位置調整機能の状態 (Dキー)
        adjust_status = "ON" if self.adjust_landmarks else "OFF"
        adjust_color = (0, 255, 0) if self.adjust_landmarks else (128, 128, 128)
        status_text = f"[D] Position Adjust (Fix Edges): {adjust_status}"
        if self.adjust_landmarks and not self.use_facelandmark:
            status_text += " (Need F key ON)"
            adjust_color = (0, 165, 255)  # オレンジ色
        cv2.putText(image, status_text, (text_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, adjust_color, 1)
        y_offset += line_height
        
        # 自動スケール機能の状態 (Aキー)
        scale_status = "ON" if self.auto_scale_model else "OFF"
        scale_color = (0, 255, 0) if self.auto_scale_model else (128, 128, 128)
        status_text = f"[A] Auto Scale: {scale_status}"
        if self.auto_scale_model and not self.use_facelandmark:
            status_text += " (Need F key ON)"
            scale_color = (0, 165, 255)  # オレンジ色
        cv2.putText(image, status_text, (text_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, scale_color, 1)
        y_offset += line_height
        
        # 対応点モード (Pキー)
        point_mode_names = {0: "All Points", 1: "Upper Points", 2: "Selected Points"}
        point_mode = point_mode_names.get(self.detect_stable, "Unknown")
        cv2.putText(image, f"[P] Point Mode: {point_mode}", (text_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
