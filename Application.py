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
    def __init__(self, title, width, height, use_api, draw_landmark):
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
        # 画像の描画を実行
        #
        self.image.flags.writeable = True

        # ランドマークの描画
        if self.draw_landmark:
            # ランドマークを描画するメソッドを実行
            self.draw_landmarks(self.image)

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
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    face_landmarks,
                    # 描画モード
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    self.drawing_spec,
                    self.drawing_spec)
        
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
