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
    def __init__(self, title, width, height, use_api, draw_landmark, use_facedetector=False, movie_path=None):
        self.width   = width
        self.height  = height
        self.channel = 3

        # カウント用変数
        self.count_img = 0
        self.count_rec = 0
        self.count_func = 0
        
        # 画像保存フラグ
        self.save_image_flag = False

        # 顔検出に用いる対応点に関する変数(顔全体の場合0)
        self.detect_stable = 0
        # 顔のランドマークを記述するかどうか
        self.draw_landmark = draw_landmark

        # モデル描画制御
        self.draw_model_flag = True  # モデル描画のON/OFF
        # アルファ処理で見た目が小さくなる分の描画補正（描画時のみ拡大）
        self.use_alpha_size_compensation = True
        self.alpha_compensation_scale_x = 1.05
        self.alpha_compensation_scale_y = 1.05
        
        # ランドマーク位置調整機能
        self.adjust_landmarks = False

        # 顔角度（yaw, pitch, roll）
        self.angle = None
        
        # ステータス表示モード (0:コンパクト, 1:詳細, 2:コンソール)
        self.status_display_mode = 0
        self.console_printed = False
        
        # 録画用変数
        self.use_record = False # 初期値はFalse
        self.video = None

        # ランドマーク座標の手動上書き（ピクセル指定）
        # 形式: {ランドマーク番号: (x_px, y_px)}
        self.landmark_overrides_px = {}
        self.landmark_overrides_loaded = False
        
        # YOLO輪郭によるリアルタイム補正（メモリ上で直接処理）
        # 設定項目：
        self.use_realtime_rinkaku_override = True         # YOLO補正の有効/無効
        self.landmark_update_interval = 5                  # Nフレームに1回だけYOLO推論を実行
        self.realtime_frame_count = 0
        self.use_yolo_outlier_filter = True               # 端寄りYOLO結果を無視する
        self.yolo_border_margin_ratio = 0.05              # 画面端判定の余白(画像短辺に対する比率)
        
        # デバッグオプション
        self.export_rinkaku_csv = False                    # True=デバッグ用にCSV出力, False=メモリのみで処理（推奨）
        self.rinkaku_yolo_csv_path = 'mqodata/input/yolooutput.csv'
        self.rinkaku_target_landmarks_right = [116, 123, 187, 207, 192, 214, 170, 176, 148, 152]
        self.rinkaku_target_landmarks_left = [345, 352, 376, 433, 367, 364, 378, 400, 377, 152]
        self.rinkaku_mode_config = {
            'right': {
                'start_key': 'right_edge',
                'avoid_key': None,
                'target_landmarks': self.rinkaku_target_landmarks_right,
            },
            'left': {
                'start_key': 'left_edge',
                'avoid_key': 'right_edge',
                'target_landmarks': self.rinkaku_target_landmarks_left,
            },
        }
        self.rinkaku_yaw_threshold_neg = -20            
        self.rinkaku_yaw_threshold_pos = 20              

        # 対応点選択モード
        # False: Pキーで手動切替（従来）
        # True: 顔角度で自動切替（基本=datalist3, yaw>=20:datalist2, yaw<=-20:datalist1）
        self.use_angle_based_point_selection = True
        self.point_mode_yaw_threshold = 20
        
        # YOLOモデル
        self.yolo_model_path = 'yolofolder/best.pt'
        self.yolo_model = None
        self.yolo_available = False
        
        # 表示用
        self.draw_yolo_debug_overlay = True
        self.latest_yolo_keypoints = None
        self.latest_rinkaku_points = []
        
        self.initialize_realtime_rinkaku_model()

        #
        # USBカメラの設定
        # USBCameraクラスのインスタンス生成
        #
        self.camera = cam.USBCamera(width, height, use_api)
        
        # デフォルトはUSBカメラだが、movie_path が指定されていれば動画ファイルを開く
        if movie_path is not None:
            # 既に開かれているカメラを閉じ、ビデオモードで再オープンする
            try:
                self.camera.Close()
            except Exception:
                pass
            # 切り替えフラグを設定して動画ファイルを開く
            self.camera.inputMode = cam.USBCamera.INPUT_VIDEO
            opened = self.camera.Open(width, height, movie_path, use_api)
            if opened:
                print(f"movie mode enabled: {movie_path}")
            else:
                print(f"failed to open movie: {movie_path}, falling back to camera")

        #
        # GLウィンドウの設定
        # GLウィンドウクラスのインスタンス生成
        #
        #
        # GLウィンドウの設定
        # GLウィンドウクラスのインスタンス生成
        #
        self.glwindow = None
        try:
            self.glwindow = GLWindow.GLWindow(
                title, 
                width, height, 
                self.display_func, 
                self.keyboard_func)
        except RuntimeError as e:
            print(f"Warning: GLWindow initialization failed: {e}")
            print("Continuing in headless mode...")
            if movie_path is not None:
                print("Processing video in headless mode without display")

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
        
        self.use_mask = False
        self.use_faceanalysis = False

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
        
        # === カメラ映像処理（リアルタイム） ===
        success, self.image = self.camera.CaptureImage()
        if not success:
            if self.camera.inputMode == cam.USBCamera.INPUT_VIDEO:
                if self.glwindow is not None:
                    glfw.set_window_should_close(self.glwindow.window, True)
                return
            print("error : video error")
            return
        # USBCameraが既にRGB変換済みのため、追加変換は不要
        self.rgb_image_for_display = self.image.copy()

        
        # # # === 静的画像処理（単一画像） ===
        # # 画像の形式の変換なリアルタイムの場合は、USBCameraクラス内で自動的にBGR→RGB変換
        # static_image_path = "/home/misa/lab/mediapipe/FaceLandmark/mqodata/input/maskpic/face17.jpg"
        # bgr_image = cv2.imread(static_image_path)
        # success = bgr_image is not None

        # if not success:
        #     print(f"error : could not load image from {static_image_path}")
        #     return
        
        # # 画像サイズを640x480にリサイズ
        # height, width = bgr_image.shape[:2]
        # if width != 640 or height != 480:
        #     bgr_image = cv2.resize(bgr_image, (640, 480))
        
        # # MediaPipe用にRGBに変換
        # self.image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # # 描画用にもRGB画像を作成（GLWindowはRGBを期待）
        # self.rgb_image_for_display = self.image.copy()

        # ###  静的画像処理（単一画像）終了 ###
    
        # 描画設定
        self.image.flags.writeable = False
       
        # 顔特徴点検出(FaceMesh)を実行
        #
        self.face_mesh = self.face_mesh_solution.process(self.image)

        # フレームカウンタ更新（リアルタイム補正の間引きに使用）
        self.realtime_frame_count += 1

        # 角度制御（draw_compact_status_info と同じ self.angle[0] の yaw を使用）
        yaw = None if self.angle is None else self.angle[0]
        rinkaku_mode = self.get_rinkaku_mode_from_yaw(yaw)
        run_rinkaku_override = rinkaku_mode is not None

        # NフレームごとにYOLOを実行して、輪郭から補正座標を更新する
        # (CSV経由ではなくメモリ上で直接処理)
        if self.use_realtime_rinkaku_override:
            should_update = (
                not self.landmark_overrides_loaded
                or (self.realtime_frame_count % max(1, self.landmark_update_interval) == 0)
            )

            if should_update and self.yolo_available and run_rinkaku_override:
                yolo_input_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                self.update_landmark_overrides_from_yolo(yolo_input_bgr, rinkaku_mode)

        # YOLO補正が有効なときのみ、補正座標をMediaPipeランドマークへ反映する
        if self.use_realtime_rinkaku_override and run_rinkaku_override and self.face_mesh.multi_face_landmarks and self.landmark_overrides_px:
            for face_landmarks in self.face_mesh.multi_face_landmarks:
                self.apply_manual_landmark_overrides(face_landmarks)


        # # === 静的画像処理（単一画像） ===


        # ##リアルタイル時コメントアウト開始##
        # # # 上書き座標の生成(ishikawa0119)
        # if not self.landmark_overrides_loaded:
        #     self.build_landmark_overrides_from_yolo_csv(
        #         self.rinkaku_yolo_csv_path,
        #         self.rinkaku_target_landmarks_right
        #     )
        #     self.landmark_overrides_loaded = True

        # # 指定ランドマークの座標を手動上書き
        # if self.face_mesh.multi_face_landmarks:
        #     for face_landmarks in self.face_mesh.multi_face_landmarks:
        #         self.apply_manual_landmark_overrides(face_landmarks)
                
        # # # ##リアルタイル時コメントアウト終了##           

        # 変更後のランドマーク152番を描画（MediaPipeの座標から取得）
        # x = int(face_landmarks.landmark[152].x * self.width)
        # y = int(face_landmarks.landmark[152].y * self.height)
        # cv2.circle(self.rgb_image_for_display, (x, y), 5, (0, 0, 255), -1)  


        #
        # 画像の描画を実行
        #
        self.image.flags.writeable = True

        # YOLO特徴点デバッグ表示（chin/right と輪郭線）
        if self.draw_yolo_debug_overlay:
            self.draw_yolo_analysis_overlay(self.rgb_image_for_display)

        # ステータス表示を追加（RGB画像に描画）
        self.draw_status_info(self.rgb_image_for_display)

        # RGB画像を描画するメソッドを実行
        self.glwindow.draw_image(self.rgb_image_for_display)    

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

            # 対応点モード決定
            # 角度連動が有効なら自動選択、無効ならPキーの手動選択を使う
            active_detect_mode = self.detect_stable
            if self.use_angle_based_point_selection:
                current_yaw = None if self.angle is None else self.angle[0]
                active_detect_mode = self.get_point_mode_from_yaw(current_yaw)

            #
            # 対応点を指定(顔全体を用いる場合は0)
            #
            if active_detect_mode == 0:
                # print("all")
                point_list = self.point_list
                point_3D = self.point_3D
            elif active_detect_mode == 1:
                # print("upper")
                point_list = self.point_list1
                point_3D = self.point_3D1
            elif active_detect_mode == 2:
                # print("selected")
                point_list = self.point_list2
                point_3D = self.point_3D2
            elif active_detect_mode == 3:
                # print("custom")
                point_list = self.point_list3
                point_3D = self.point_3D3
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
    
        #else:
            #
            # 検出が安定しない
            #
         #    print("not detection")    


        # 関数実行回数を更新
        self.count_func += 1
        
        # 画像保存フラグがTrueの場合、バッファスワップ前に保存
        if self.save_image_flag:
            self.save_image()
            self.save_image_flag = False
        
        # 録画している場合画面を保存
        if self.use_record:
            frame = self.save_image_for_recording()
            self.video.write(frame)
        
        # バッファを入れ替えて画面を更新
        glfw.swap_buffers(window)

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
        
        model_scale_X = 1.0 * scale_x
        model_scale_Y = 1.0 * scale_y
        model_scale_Z = 1.0 

        # PnPには影響させず、描画モデルのみをわずかに拡大してアルファ透過分を補正する
        if self.use_alpha_size_compensation:
            model_scale_X *= self.alpha_compensation_scale_x
            model_scale_Y *= self.alpha_compensation_scale_y
    
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
                # 画像保存フラグを立てる（次のdisplay_funcで保存される）
                self.save_image_flag = True

        
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
                print("対応点をモード1(右顔)に変更")
            elif self.detect_stable == 1:
                self.detect_stable = 2
                print("対応点をモード2(左顔)に変更")
            elif self.detect_stable == 2:
                self.detect_stable = 3
                print("対応点をモード3(目元)に変更")
            elif self.detect_stable == 3:
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

        # FでYOLO補正の更新間隔を切り替え
        if action == glfw.PRESS and key == glfw.KEY_F:
            interval_list = [1, 5, 10, 15]
            current_idx = 0
            if self.landmark_update_interval in interval_list:
                current_idx = interval_list.index(self.landmark_update_interval)
            self.landmark_update_interval = interval_list[(current_idx + 1) % len(interval_list)]
            print(f"YOLO補正の更新間隔を {self.landmark_update_interval}フレーム/回 に変更")

        # YでYOLOデバッグ描画のON/OFF切り替え
        if action == glfw.PRESS and key == glfw.KEY_Y:
            self.draw_yolo_debug_overlay = not self.draw_yolo_debug_overlay
            if self.draw_yolo_debug_overlay:
                print("YOLOデバッグ描画を有効化しました")
            else:
                print("YOLOデバッグ描画を無効化しました")
        

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
        filename = 'output/images/maskpic/image_{}-{}.png'.format(today, self.count_img)
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # バッファを読み込む(画面を読み込む)
        glReadBuffer(GL_BACK)  # ダブルバッファリングの場合はGL_BACKを使用
        # ピクセルを読み込む（RGBフォーマットで読み取り）
        glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, image.data)
        # OpenGLはRGB形式で読み取るので、BGR形式に変換（cv2.imwriteはBGRを期待）
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # OpenGLは下から上に読み取るので上下反転
        image = cv2.flip(image, 0)
        
        # 画像を保存
        print("画像を保存します..." + filename)
        cv2.imwrite(filename, image)
        self.count_img += 1  # カウンタを増やす
    
    #
    # 録画用に画像を返す関数
    #
    def save_image_for_recording(self):
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        glReadBuffer(GL_BACK)
        glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, image.data)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.flip(image, 0)
        return image
        
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

    def initialize_realtime_rinkaku_model(self):
        """
        リアルタイム輪郭補正に使うYOLOモデルを初期化する。
        """
        import os

        if not self.use_realtime_rinkaku_override:
            return

        if not os.path.exists(self.yolo_model_path):
            print(f"YOLOモデルが見つかりません: {self.yolo_model_path}")
            return

        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.yolo_model_path)
            self.yolo_available = True
            print(f"YOLO輪郭補正を有効化: interval={self.landmark_update_interval}")
        except Exception as e:
            self.yolo_model = None
            self.yolo_available = False
            print(f"YOLO初期化に失敗しました: {e}")

    def get_rinkaku_mode_from_yaw(self, yaw):
        """
        yaw値から輪郭補正モードを返す。
        yaw >= 20 なら左向き、yaw <= -20 なら右向き、それ以外は None。
        """
        if yaw is None:
            return None
        if yaw >= self.rinkaku_yaw_threshold_pos:
            return 'left'
        if yaw <= self.rinkaku_yaw_threshold_neg:
            return 'right'
        return None

    def get_point_mode_from_yaw(self, yaw):
        """
        対応点モードを顔角度から決定する。
        0: 全点, 1: 右顔, 2: 左顔, 3: 目元
        """
        if yaw is None:
            return 3
        if yaw >= self.point_mode_yaw_threshold:
            return 2
        if yaw <= -self.point_mode_yaw_threshold:
            return 1
        return 3

    def is_yolo_keypoints_reliable(self, keypoints, rinkaku_mode):
        """
        YOLOの輪郭特徴点が画面端に寄りすぎていないかを判定する。
        端点や顎が画面端に接している場合は、上書きに使わない。
        """
        if not keypoints:
            return False

        mode_config = self.rinkaku_mode_config.get(rinkaku_mode, self.rinkaku_mode_config['right'])
        border_margin = max(1, int(min(self.width, self.height) * self.yolo_border_margin_ratio))

        required_keys = ['chin', mode_config['start_key']]
        if mode_config.get('avoid_key'):
            required_keys.append(mode_config['avoid_key'])

        for key in required_keys:
            point = keypoints.get(key)
            if point is None:
                return False

            x, y = point
            if x <= border_margin or x >= (self.width - border_margin):
                print(f"YOLO結果を破棄: {key} が画面端に近すぎます ({x:.1f}, {y:.1f})")
                return False
            if y <= border_margin or y >= (self.height - border_margin):
                print(f"YOLO結果を破棄: {key} が画面端に近すぎます ({x:.1f}, {y:.1f})")
                return False

        return True

    def find_mask_keypoints(self, contour):
        """
        マスク輪郭から顎/鼻/左右端を抽出する。
        """
        points = []
        for point in contour:
            x, y = point[0][0], point[0][1]
            points.append((x, y))

        if not points:
            return None

        points_by_y = sorted(points, key=lambda p: p[1])
        chin_point = points_by_y[-1]
        nose_point = points_by_y[0]
        upper_points = points_by_y[:max(1, len(points_by_y) // 5)]
        left_point = max(upper_points, key=lambda p: p[0])
        right_point = min(upper_points, key=lambda p: p[0])

        return {
            'chin': chin_point,
            'nose': nose_point,
            'left_edge': left_point,
            'right_edge': right_point,
            'all_points': points,
        }

    def extract_contour_between_points(self, points, chin_point, right_point):
        """
        右端から顎までの輪郭を抽出する。
        """
        chin_idx = None
        right_idx = None

        for i, p in enumerate(points):
            if p == chin_point:
                chin_idx = i
            if p == right_point:
                right_idx = i

        if chin_idx is None or right_idx is None:
            return []

        if right_idx < chin_idx:
            return points[right_idx:chin_idx + 1]
        return points[right_idx:] + points[:chin_idx + 1]

    def extract_contour_between_points_avoiding(self, points, start_point, end_point, avoid_point):
        """
        start_point から end_point までの2経路のうち、avoid_point を通らない方を返す。
        """
        if not points:
            return []

        start_idx = None
        end_idx = None
        for i, p in enumerate(points):
            if p == start_point:
                start_idx = i
            if p == end_point:
                end_idx = i

        if start_idx is None or end_idx is None:
            return []

        n = len(points)

        forward_path = []
        i = start_idx
        while True:
            forward_path.append(points[i])
            if i == end_idx:
                break
            i = (i + 1) % n

        backward_path = []
        i = start_idx
        while True:
            backward_path.append(points[i])
            if i == end_idx:
                break
            i = (i - 1 + n) % n

        if avoid_point is None or avoid_point == start_point or avoid_point == end_point:
            return forward_path

        forward_has_avoid = avoid_point in forward_path
        backward_has_avoid = avoid_point in backward_path

        if forward_has_avoid and not backward_has_avoid:
            return backward_path
        if backward_has_avoid and not forward_has_avoid:
            return forward_path

        return forward_path

    def save_rinkaku_points_to_csv(self, points, csv_path):
        """
        輪郭点をCSVに保存する（毎回上書き）。
        """
        import csv
        import os

        try:
            dir_path = os.path.dirname(csv_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(['番号', 'x座標', 'y座標'])
                for idx, point in enumerate(points):
                    writer.writerow([idx, point[0], point[1]])
        except Exception as e:
            print(f"輪郭CSV保存エラー: {e}")

    def draw_yolo_analysis_overlay(self, image):
        """
        YOLOの解析用に、chin/left/right座標と輪郭線を画像へ重畳する。
        """
        if image is None:
            return

        # 抽出した輪郭線
        if self.latest_rinkaku_points:
            rinkaku_array = np.array(self.latest_rinkaku_points, dtype=np.int32)
            cv2.polylines(image, [rinkaku_array], False, (255, 255, 0), 2)

        if not self.latest_yolo_keypoints:
            return

        chin = self.latest_yolo_keypoints.get('chin')
        left = self.latest_yolo_keypoints.get('left_edge')
        right = self.latest_yolo_keypoints.get('right_edge')

        if chin is not None:
            cx, cy = int(chin[0]), int(chin[1])
            cv2.circle(image, (cx, cy), 7, (255, 0, 0), -1)
            cv2.putText(image, f"chin({cx},{cy})", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if left is not None:
            lx, ly = int(left[0]), int(left[1])
            cv2.circle(image, (lx, ly), 7, (0, 255, 0), -1)
            cv2.putText(image, f"left({lx},{ly})", (lx + 8, ly - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if right is not None:
            rx, ry = int(right[0]), int(right[1])
            cv2.circle(image, (rx, ry), 7, (0, 255, 255), -1)
            cv2.putText(image, f"right({rx},{ry})", (rx + 8, ry - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


    def update_landmark_overrides_from_yolo(self, bgr_image, rinkaku_mode='right'):
        """
        現在フレームからYOLO輪郭を抽出し、上書き座標を更新する。
        """
        if not self.yolo_available or self.yolo_model is None or bgr_image is None:
            self.latest_yolo_keypoints = None
            self.latest_rinkaku_points = []
            return False

        # YOLO側の入力サイズをアプリ表示サイズに合わせる
        if bgr_image.shape[1] != self.width or bgr_image.shape[0] != self.height:
            target_image = cv2.resize(bgr_image, (self.width, self.height))
        else:
            target_image = bgr_image

        try:
            results = self.yolo_model(target_image, max_det=1, verbose=False)[0]
        except Exception as e:
            print(f"YOLO推論エラー: {e}")
            self.latest_yolo_keypoints = None
            self.latest_rinkaku_points = []
            return False

        if results.masks is None:
            self.latest_yolo_keypoints = None
            self.latest_rinkaku_points = []
            return False

        best_contour = None
        best_area = 0.0

        for mask in results.masks.data:
            mask_resized = cv2.resize(mask.cpu().numpy(), (self.width, self.height))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            if area > best_area:
                best_area = area
                best_contour = main_contour

        if best_contour is None:
            self.latest_yolo_keypoints = None
            self.latest_rinkaku_points = []
            return False

        keypoints = self.find_mask_keypoints(best_contour)
        if not keypoints:
            self.latest_yolo_keypoints = None
            self.latest_rinkaku_points = []
            return False

        mode_config = self.rinkaku_mode_config.get(rinkaku_mode, self.rinkaku_mode_config['right'])
        contour_start_point = keypoints.get(mode_config['start_key'])
        avoid_key = mode_config['avoid_key']
        avoid_point = keypoints.get(avoid_key) if avoid_key else None
        target_landmarks = mode_config['target_landmarks']

        if contour_start_point is None:
            self.latest_yolo_keypoints = keypoints
            self.latest_rinkaku_points = []
            return False

        if self.use_yolo_outlier_filter and not self.is_yolo_keypoints_reliable(keypoints, rinkaku_mode):
            self.latest_yolo_keypoints = keypoints
            self.latest_rinkaku_points = []
            return False

        if avoid_point is not None:
            rinkaku_points = self.extract_contour_between_points_avoiding(
                keypoints['all_points'],
                contour_start_point,
                keypoints['chin'],
                avoid_point
            )
        else:
            rinkaku_points = self.extract_contour_between_points(
                keypoints['all_points'],
                keypoints['chin'],
                contour_start_point
            )
        if len(rinkaku_points) < 2:
            self.latest_yolo_keypoints = keypoints
            self.latest_rinkaku_points = []
            return False

        self.latest_yolo_keypoints = keypoints
        self.latest_rinkaku_points = rinkaku_points

        # (デバッグ用) CSV出力が有効な場合のみ保存
        if self.export_rinkaku_csv:
            self.save_rinkaku_points_to_csv(rinkaku_points, self.rinkaku_yolo_csv_path)

        # メモリ上で直接上書き座標を生成・反映
        success = self.build_landmark_overrides_from_points(
            rinkaku_points,
            target_landmarks
        )
        self.landmark_overrides_loaded = success
        
        return success

    def build_landmark_overrides_from_points(self, points, target_landmarks):
        """
        輪郭点列から等間隔点を計算して上書き座標を生成する。
        """
        if len(points) < 2 or not target_landmarks:
            return False

        distances = [0.0]
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            distances.append(distances[-1] + np.sqrt(dx * dx + dy * dy))

        total_distance = distances[-1]
        if total_distance <= 0:
            return False

        target_count = len(target_landmarks)
        overrides = {}
        for idx, landmark_id in enumerate(target_landmarks):
            if target_count > 1:
                target_distance = (idx / (target_count - 1)) * total_distance
            else:
                target_distance = 0.0

            for i in range(len(distances) - 1):
                if distances[i] <= target_distance <= distances[i + 1]:
                    segment = distances[i + 1] - distances[i]
                    ratio = 0.0 if segment <= 0 else (target_distance - distances[i]) / segment
                    x_interp = points[i][0] + ratio * (points[i + 1][0] - points[i][0])
                    y_interp = points[i][1] + ratio * (points[i + 1][1] - points[i][1])
                    overrides[landmark_id] = (x_interp, y_interp)
                    break

        if not overrides:
            return False

        self.landmark_overrides_px.update(overrides)
        return True

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
    def set_3D_point_3(self, point_3D, point_list):
        self.point_3D3 = point_3D
        self.point_list3 = point_list

    # ３次元モデルをセット
    def set_mqo_model(self, model):
        self.model = model
    
    # 入力画像をセット
    def set_image(self, image):
        image = cv2.cvtColor (image, cv2.COLOR_BGR2RGB)
        self.image = image

    

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
        
        
        point_mode_names = {0: "全点", 1: "上部", 2: "選択", 3: "追加"}
        point_mode = point_mode_names.get(self.detect_stable, "不明")
        print(f"対応点モード [P]:         {point_mode}")
        
        print("-" * 50)
        print("キー操作:")
        print("  [Q] 終了    [S] 画像保存    [R] 録画")
        print("  [N] モデル描画    [P] 対応点モード")
        print("  [T] 表示モード切替")
        print("  [F] YOLOYOLO補正の更新間隔切替")
        print("  [Y] YOLOデバッグ描画切替")
        print("=" * 50)
        
        if self.status_display_mode == 2:
            self.console_printed = True
    

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
        model_status = "ON" if self.draw_model_flag else "OFF"
        status_text = f"Model Draw: {model_status}"
        cv2.putText(image, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.angle is None:
            angle_text = "Face Yaw: N/A"
        else:
            yaw = self.angle[0]
            rounded_yaw = int(round(yaw / 5.0) * 5)
            angle_text = f"Face Yaw: {rounded_yaw} deg"

        cv2.putText(image, angle_text, (10, 58),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def draw_detailed_status_info(self, image):
        """
        詳細なステータス表示
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
        point_mode_names = {0: "All Points", 1: "Right Points", 2: "Left Points", 3: "Eye Points"}
        point_mode = point_mode_names.get(self.detect_stable, "Unknown")
        cv2.putText(image, f"[P] Point Mode: {point_mode}", (text_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
