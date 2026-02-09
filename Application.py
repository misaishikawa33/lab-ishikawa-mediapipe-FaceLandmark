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
    def __init__(self, title, width, height, use_api, draw_landmark, use_facedetector=False):
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

        # 顔検出モデル使用フラグ
        self.use_face_landmarker = False 
        self.face_landmarker_results = None  
        
        # モデル描画制御
        self.draw_model_flag = True  # モデル描画のON/OFF
        
        # ランドマーク位置調整機能
        self.adjust_landmarks = False
        
        # ステータス表示モード (0:コンパクト, 1:詳細, 2:コンソール)
        self.status_display_mode = 0
        self.console_printed = False
        
        # Face Landmarkerスケール係数
        self.face_landmarker_scale = 1.0
        self.manual_scale_set = False  # 手動スケール調整フラグ
        
        # 録画用変数
        self.use_record = False # 初期値はFalse
        self.video = None

        # ランドマーク座標の手動上書き（ピクセル指定）
        # 形式: {ランドマーク番号: (x_px, y_px)}
        # 例: 152番のみ初期登録。その他はCSVで指定可能。
        self.landmark_overrides_px = {152: (234, 437)}
        self.landmark_overrides_loaded = False

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
        self.focus = 1500.0 #20251118に変更
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

    def display_func(self, window):

        # 初回実行
        if self.count_func == 0:
            self.count_func = 1
            glClear(GL_COLOR_BUFFER_BIT)
            return

        # バッファを初期化
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # 画像の読み込み
        
        # # === カメラ映像処理（リアルタイム） ===
        # success, self.image = self.camera.CaptureImage()
        # if not success:
        #     print("error : video error")
        #     return
        # # USBCameraが既にRGB変換済みのため、追加変換は不要
        # self.rgb_image_for_display = self.image.copy()
        
        # # === 静的画像処理（単一画像） ===
        # 画像の形式の変換なリアルタイムの場合は、USBCameraクラス内で自動的にBGR→RGB変換
        static_image_path = "/home/misa/lab/mediapipe/FaceLandmark/mqodata/input/masked4_face_up.jpg"
        bgr_image = cv2.imread(static_image_path)
        success = bgr_image is not None

        if not success:
            print(f"error : could not load image from {static_image_path}")
            return
        
        # 画像サイズを640x480にリサイズ
        height, width = bgr_image.shape[:2]
        if width != 640 or height != 480:
            bgr_image = cv2.resize(bgr_image, (640, 480))
        
        # MediaPipe用にRGBに変換
        self.image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # 描画用にもRGB画像を作成（GLWindowはRGBを期待）
        self.rgb_image_for_display = self.image.copy()

        # ### ここまでMediaPipe処理部分 ###
    
        # 描画設定
        self.image.flags.writeable = False
       
        # 顔特徴点検出(FaceMesh)を実行
        #
        self.face_mesh = self.face_mesh_solution.process(self.image)

        ##リアルタイル時コメントアウト開始##
        # # 上書き座標の読込(ishikawa0119)
        if not self.landmark_overrides_loaded:
            self.load_landmark_overrides('mqodata/input/rinkaku.csv')
            self.landmark_overrides_loaded = True

        # 指定ランドマークの座標を手動上書き
        if self.face_mesh.multi_face_landmarks:
            for face_landmarks in self.face_mesh.multi_face_landmarks:
                self.apply_manual_landmark_overrides(face_landmarks)
                
        # # ##リアルタイル時コメントアウト終了##



                    

        # 変更後のランドマーク152番を描画（MediaPipeの座標から取得）
        # x = int(face_landmarks.landmark[152].x * self.width)
        # y = int(face_landmarks.landmark[152].y * self.height)
        # cv2.circle(self.rgb_image_for_display, (x, y), 5, (0, 0, 255), -1)  
        x = int(face_landmarks.landmark[234].x * self.width)
        y = int(face_landmarks.landmark[234].y * self.height)
        cv2.circle(self.rgb_image_for_display, (x, y), 5, (0, 0, 255), -1)  


        #
        # 画像の描画を実行
        #
        self.image.flags.writeable = True

        # ステータス表示を追加（RGB画像に描画）
        self.draw_status_info(self.rgb_image_for_display)

        # RGB画像を描画するメソッドを実行
        self.glwindow.draw_image(self.rgb_image_for_display)    

        # ランドマークの描画（RGB画像に描画）
        if self.draw_landmark:
            # ランドマークを描画するメソッドを実行
            self.draw_landmarks(self.rgb_image_for_display)

            


        # ランドマークの描画（RGB画像に描画）
        if self.draw_landmark:
            # ランドマークを描画するメソッドを実行
            self.draw_landmarks(self.rgb_image_for_display)


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
            # 常に通常のPnP方式を使用
            success, vector, angle = self.compute_camera_pose(point_2D, point_3D)
            self.angle = angle
            self.alignment_info = None  # 固定値に設定
            
            #
            # モデル描画フラグが有効な場合のみモデルを描画
            #
            if success and self.draw_model_flag:
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
        
        # Face Landmarker直接姿勢推定モードの場合は、face_landmarker_scaleを使用
        if hasattr(self, 'use_direct_pose') and self.use_direct_pose:
            model_scale_X = 1.0 * scale_x * self.face_landmarker_scale
            model_scale_Y = 1.0 * scale_y * self.face_landmarker_scale
            model_scale_Z = 1.0 * self.face_landmarker_scale
        else:
            model_scale_X = 1.0 * scale_x
            model_scale_Y = 1.0 * scale_y
            model_scale_Z = 1.0 
    
        # 世界座標系の描画
        if self.draw_axis:
            mesh_size = 200.0
            mesh_grid = 10.0
            # カメラを平行移動
            glTranslatef(model_shift_X, model_shift_Y, model_shift_Z)
            # 回転(x方向に90度)
            glRotatef(90.0, 1.0, 0.0, 0.0)


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
        
        # Tでステータス表示モードの切り替え
        if action == glfw.PRESS and key == glfw.KEY_T:
            if not hasattr(self, 'status_display_mode'):
                self.status_display_mode = 0  # 0:コンパクト, 1:詳細, 2:コンソール
            
            self.status_display_mode = (self.status_display_mode + 1) % 3
            mode_names = ["コンパクト", "詳細", "コンソール"]
            print(f"ステータス表示モードを{mode_names[self.status_display_mode]}に変更しました")
            
            # コンソールモードの場合、現在の状態を出力
            if self.status_display_mode == 2:
                self.print_status_to_console()
        
        # Nでモデル描画のON/OFF切り替え
        if action == glfw.PRESS and key == glfw.KEY_N:
            self.draw_model_flag = not self.draw_model_flag
            if self.draw_model_flag:
                print("モデル描画を有効化しました")
            else:
                print("モデル描画を無効化しました")
        

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
    # ランドマーク座標の手動上書き関連
    #
    def load_landmark_overrides(self, csv_path):
        """
        CSVからランドマーク上書き座標を読み込む。
        形式: idx,x,y（ピクセル値）。#で始まる行はコメントとして無視。
        """
        import os
        if not os.path.exists(csv_path):
            return
        try:
            overrides = {}
            with open(csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) < 3:
                        continue
                    try:
                        idx = int(parts[0])
                        x_px = float(parts[1])
                        y_px = float(parts[2])
                        overrides[idx] = (x_px, y_px)
                    except ValueError:
                        # 数値に変換できない行はスキップ
                        continue
            # 既存の辞書に上書き（CSV優先）
            self.landmark_overrides_px.update(overrides)
            if overrides:
                print(f"landmark_overrides.csv を読み込み: {len(overrides)}件")
        except Exception as e:
            print(f"ランドマーク上書きCSV読込エラー: {e}")

    def apply_manual_landmark_overrides(self, face_landmarks):
        """
        self.landmark_overrides_px に基づき、指定ランドマークの x,y を
        画像サイズで正規化した値に置換する。
        """
        try:
            total = len(face_landmarks.landmark)
            for idx, (x_px, y_px) in self.landmark_overrides_px.items():
                if idx < 0 or idx >= total:
                    continue
                face_landmarks.landmark[idx].x = x_px / self.width
                face_landmarks.landmark[idx].y = y_px / self.height
        except Exception as e:
            print(f"ランドマーク上書き適用エラー: {e}")
      
      
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
    # 両方の姿勢推定を実行して結果を比較する関数（デバッグ用）
    #
    def compute_pose_comparison(self, point_2D, point_3D):
        """
        PnP方式とFace Landmarker方式の両方を実行して結果を比較
        """
        # 1. 従来のPnP方式を実行
        pnp_success, pnp_vector, pnp_angle = self.compute_camera_pose(point_2D, point_3D)
        
        # 2. Face Landmarker方式を実行（利用可能な場合）
        fl_success, fl_vector, fl_angle = False, None, None
        if (self.use_face_landmarker and self.face_landmarker_results and 
            hasattr(self.face_landmarker_results, 'facial_transformation_matrixes') and
            self.face_landmarker_results.facial_transformation_matrixes):
            
            fl_success, fl_vector, fl_angle = self.compute_pose_from_face_landmarker()
        
        # 3. 結果を比較・保存
        comparison_result = {
            'timestamp': datetime.datetime.now(),
            'pnp_success': pnp_success,
            'pnp_vector': pnp_vector,
            'pnp_angle': pnp_angle,
            'fl_success': fl_success,
            'fl_vector': fl_vector,
            'fl_angle': fl_angle
        }
        
        self.comparison_results.append(comparison_result)
        
        # 結果をファイルに保存（一度だけ）
        if not hasattr(self, 'comparison_saved') or not self.comparison_saved:
            self.save_pose_comparison_results(comparison_result)
            self.comparison_saved = True
        
        # 結果をファイルに保存（最新の10件のみ保持）
        if len(self.comparison_results) > 10:
            self.comparison_results = self.comparison_results[-10:]
        
        return pnp_success, pnp_vector, pnp_angle
    

    def print_status_to_console(self):
        """
        現在の状態をコンソールに詳細出力
        """
        if self.console_printed and self.status_display_mode == 2:
            return  # 一度だけ出力
        
        print("=" * 50)
        print("          MediaPipe AR システム状態")
        print("=" * 50)
        print(f"モデル描画 [N]:           {'ON' if self.draw_model_flag else 'OFF'}")
        
        
        point_mode_names = {0: "全点", 1: "上部", 2: "選択"}
        point_mode = point_mode_names.get(self.detect_stable, "不明")
        print(f"対応点モード [P]:         {point_mode}")
        
        print("-" * 50)
        print("キー操作:")
        print("  [Q] 終了    [S] 画像保存    [R] 録画")
        print("  [N] モデル描画    [P] 対応点モード")
        print("  [T] 表示モード切替")
        print("=" * 50)
        
        if self.status_display_mode == 2:
            self.console_printed = True
    
    #
    # 姿勢比較結果をファイルに保存する関数
    #
    def save_pose_comparison_results(self, comparison_result):
        """
        姿勢比較結果を詳細なテキストファイルとして保存
        """
        try:
            import os
            # outputディレクトリが存在しない場合は作成
            os.makedirs('output', exist_ok=True)
            
            today = str(datetime.date.today()).replace('-','')
            filename = f'output/pose_comparison_{today}_{self.count_img}.txt'
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=== 姿勢推定比較結果 ===\n")
                f.write(f"画像番号: {self.count_img}\n")
                f.write(f"日時: {comparison_result['timestamp']}\n\n")
                
                # PnP方式の結果
                f.write("--- PnP方式 ---\n")
                f.write(f"成功: {comparison_result['pnp_success']}\n")
                if comparison_result['pnp_success'] and comparison_result['pnp_angle'] is not None:
                    angle = comparison_result['pnp_angle']
                    f.write(f"オイラー角: X={angle[0]:.3f}, Y={angle[1]:.3f}, Z={angle[2]:.3f}\n")
                    if comparison_result['pnp_vector'] is not None:
                        vector = comparison_result['pnp_vector']
                        f.write(f"方向ベクトル: ({vector[0]}, {vector[1]})\n")
                else:
                    f.write("姿勢推定失敗\n")
                f.write("\n")
                
                # Face Landmarker方式の結果
                f.write("--- Face Landmarker方式 ---\n")
                f.write(f"成功: {comparison_result['fl_success']}\n")
                if comparison_result['fl_success'] and comparison_result['fl_angle'] is not None:
                    angle = comparison_result['fl_angle']
                    f.write(f"オイラー角: X={angle[0]:.3f}, Y={angle[1]:.3f}, Z={angle[2]:.3f}\n")
                    # Face Landmarkerのスケール係数を追加
                    if hasattr(self, 'face_landmarker_scale'):
                        f.write(f"スケール係数: {self.face_landmarker_scale}\n")
                    if comparison_result['fl_vector'] is not None:
                        vector = comparison_result['fl_vector']
                        f.write(f"方向ベクトル: {vector}\n")
                else:
                    f.write("姿勢推定失敗またはFace Landmarker無効\n")
                f.write("\n")
                
                # 角度差分の計算と保存
                if (comparison_result['pnp_success'] and comparison_result['fl_success'] and 
                    comparison_result['pnp_angle'] is not None and comparison_result['fl_angle'] is not None):
                    
                    pnp_angle = comparison_result['pnp_angle']
                    fl_angle = comparison_result['fl_angle']
                    
                    diff_x = fl_angle[0] - pnp_angle[0]
                    diff_y = fl_angle[1] - pnp_angle[1]
                    diff_z = fl_angle[2] - pnp_angle[2]
                    diff_norm = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
                    
                    f.write("--- 角度差分 (Face Landmarker - PnP) ---\n")
                    f.write(f"X軸: {diff_x:.3f}度\n")
                    f.write(f"Y軸: {diff_y:.3f}度\n")
                    f.write(f"Z軸: {diff_z:.3f}度\n")
                    f.write(f"差分ノルム: {diff_norm:.3f}度\n")
                
            print(f"姿勢比較結果を保存しました: {filename}")
            
        except Exception as e:
            print(f"姿勢比較結果の保存に失敗しました: {e}")

    #
    # 現在の状態を画面に表示する関数
    #
    def draw_status_info(self, image):
        """
        ステータス表示モードに応じて情報を表示
        """
        if self.status_display_mode == 0:
            # コンパクトモード
            self.draw_compact_status_info(image)
        elif self.status_display_mode == 1:
            # 詳細モード
            self.draw_detailed_status_info(image)
        elif self.status_display_mode == 2:
            # コンソールモード（画面には何も表示しない）
            pass
    
    def draw_compact_status_info(self, image):
        """
        コンパクトなステータス表示（ONの機能のみ表示）
        """
        # ONになっている機能のみを表示
        active_features = []
        if self.draw_model_flag:
            active_features.append("Model Draw")
        
        # ONの機能がない場合
        if not active_features:
            status_text = "All features OFF"
        else:
            status_text = "ON: " + ", ".join(active_features)
        
        cv2.putText(image, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def draw_detailed_status_info(self, image):
        """
        詳細なステータス表示（従来の表示）
        """
        # 背景の半透明ボックスを描画（見やすくするため）
        # 右上に配置
        box_x1 = self.width - 450  # 幅を少し広げて比較モードも表示
        box_y1 = 10
        box_x2 = self.width - 10
        box_y2 = 200  # Face DetectorとFace Landmarkerを削除したので高さを減らす
        
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
        
        # モデル描画の状態 (Nキー)
        model_draw_status = "ON" if self.draw_model_flag else "OFF"
        model_draw_color = (0, 255, 0) if self.draw_model_flag else (128, 128, 128)
        cv2.putText(image, f"[N] Model Draw: {model_draw_status}", (text_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, model_draw_color, 1)
        y_offset += line_height
        
        # 対応点モード (Pキー)
        point_mode_names = {0: "All Points", 1: "Upper Points", 2: "Selected Points"}
        point_mode = point_mode_names.get(self.detect_stable, "Unknown")
        cv2.putText(image, f"[P] Point Mode: {point_mode}", (text_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        # Face Landmarkerスケール情報（Face Landmarker有効時のみ）
        if self.use_face_landmarker:
            scale_mode = "Manual" if hasattr(self, 'manual_scale_set') and self.manual_scale_set else "Auto"
            scale_text = f"FL Scale: {self.face_landmarker_scale:.2f} ({scale_mode})"
            cv2.putText(image, scale_text, (text_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # 黄色
