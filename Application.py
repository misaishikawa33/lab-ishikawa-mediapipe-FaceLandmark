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
from ModelBoundaryBlender import ModelBoundaryBlender
from YoloRinkakuCorrector import YoloRinkakuCorrector
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
    def __init__(self, title, width, height, use_api, draw_landmark, use_facedetector=False, movie_path=None, record=False, record_format='mp4'):
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
        # 顔端の黒化対策として、モデル生成時にテクスチャ外挿を行うか
        self.use_edge_texture_extension = True
        # 横向き参照画像の色でモデル生成時の顔端テクスチャを補完するか
        self.use_reference_edge_color_completion = False
        self.reference_edge_color_source_mode = 'inner' # 'outer'：外側から外側, 'inner'：内側から外側
        self.draw_reference_edge_color_debug_landmarks = True
        self.reference_edge_left_image_path = 'mqodata/face7.jpg'
        self.reference_edge_right_image_path = 'mqodata/face22.jpg'
        # モデルと入力画像の境界を自然になじませる後処理
        self.use_model_boundary_blend = False
        self.model_boundary_blender = ModelBoundaryBlender(
            enabled=self.use_model_boundary_blend)

        # 入力画像の肌色基準ランドマーク群を基準にモデルテクスチャの色味・コントラストを合わせる
        self.use_model_color_match = True
        self.model_color_match_mode = 'lab_luminance' # 'rgb', 'ycrcb_luminance', 'lab_luminance'
        self.skin_landmarks = [10, 151, 9, 8, 168]
        self.color_match_patch_radius = 3
        self.model_reference_rgb = None
        self.model_edge_reference_rgb = {}
        self.model_edge_landmark_base_colors = {}
        self.model_edge_landmark_colors = {}
        self.smoothed_target_rgb = None
        self.color_match_smoothing = 0.85 
        self.color_match_update_yaw_limit = 15.0
        self.draw_skin_landmarks = False

        # 顔角度（yaw, pitch, roll）
        self.angle = None
        
        # ステータス表示モード (0:コンパクト, 1:詳細, 2:コンソール)
        self.status_display_mode = 0
        self.console_printed = False
        
        # 録画用変数
        self.use_record = False # 初期値はFalse
        self.video = None
        self.record_requested = record
        self.record_format = record_format
        self.record_output_dir = 'output/videos'
        self.record_output_path = None

        # YOLO輪郭補正の主な設定
        # 補正処理自体を使うかどうか
        self.use_yolo_rinkaku_correction = True
        # YOLOの検出結果・補正点を画面に描画するかどうか
        self.draw_yolo_debug_overlay = True
        self.realtime_frame_count = 0
        self.yolo_rinkaku_corrector = YoloRinkakuCorrector(
            width=width,
            height=height,
            enabled=self.use_yolo_rinkaku_correction,
            draw_debug_overlay=self.draw_yolo_debug_overlay)

        # 対応点選択モード
        # False: Pキーで手動切替（従来）
        # True: 顔角度で自動切替（基本=datalist3, yaw>=20:datalist2, yaw<=-20:datalist1）
        self.use_angle_based_point_selection = True
        self.point_mode_yaw_threshold = 20
        
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
    
        # 描画設定
        self.image.flags.writeable = False
       
        # 顔特徴点検出(FaceMesh)を実行
        #
        self.face_mesh = self.face_mesh_solution.process(self.image)

        # フレームカウンタ更新（リアルタイム補正の間引きに使用）
        self.realtime_frame_count += 1

        # 角度制御（draw_compact_status_info と同じ self.angle[0] の yaw を使用）
        yaw = None if self.angle is None else self.angle[0]
        rinkaku_mode = self.yolo_rinkaku_corrector.get_rinkaku_mode_from_yaw(yaw)
        run_rinkaku_override = rinkaku_mode is not None

        if self.yolo_rinkaku_corrector.enabled:
            should_update = self.yolo_rinkaku_corrector.should_update(self.realtime_frame_count)
            run_yolo_for_edge_inpaint = self.yolo_rinkaku_corrector.use_mask_edge_inpaint
            if (
                    should_update
                    and self.yolo_rinkaku_corrector.available
                    and (run_rinkaku_override or run_yolo_for_edge_inpaint)):
                yolo_input_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                yolo_face_landmarks = None
                if self.face_mesh.multi_face_landmarks:
                    yolo_face_landmarks = self.face_mesh.multi_face_landmarks[0]
                self.yolo_rinkaku_corrector.update_landmark_overrides_from_yolo(
                    yolo_input_bgr,
                    rinkaku_mode or 'right',
                    yolo_face_landmarks)

        if (
                self.yolo_rinkaku_corrector.enabled
                and run_rinkaku_override
                and self.face_mesh.multi_face_landmarks
                and self.yolo_rinkaku_corrector.landmark_overrides_px):
            for face_landmarks in self.face_mesh.multi_face_landmarks:
                self.yolo_rinkaku_corrector.apply_landmark_overrides(face_landmarks)

        self.image.flags.writeable = True

        if (
                self.yolo_rinkaku_corrector.enabled
                and self.yolo_rinkaku_corrector.use_mask_edge_inpaint):
            edge_rgb = self.get_yolo_edge_inpaint_rgb()
            edge_face_landmarks = None
            if self.face_mesh.multi_face_landmarks:
                edge_face_landmarks = self.face_mesh.multi_face_landmarks[0]
            self.rgb_image_for_display = self.yolo_rinkaku_corrector.apply_edge_inpaint(
                self.rgb_image_for_display,
                edge_rgb,
                edge_face_landmarks,
                self.model_edge_landmark_colors)

        # YOLO特徴点デバッグ表示（chin/right と輪郭線）
        if self.yolo_rinkaku_corrector.draw_debug_overlay:
            self.yolo_rinkaku_corrector.draw_overlay(self.rgb_image_for_display)

        # ステータス表示を追加（RGB画像に描画）
        self.draw_status_info(self.rgb_image_for_display)

        if self.draw_skin_landmarks and self.face_mesh.multi_face_landmarks:
            self.draw_skin_color_landmarks(
                self.rgb_image_for_display,
                self.face_mesh.multi_face_landmarks[0])

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
            if success:
                self.update_model_color_match(self.image, self.face_mesh.multi_face_landmarks[0])
            
            #
            # モデル描画フラグが有効な場合のみモデルを描画
            #
            if success and self.draw_model_flag:
                self.draw_model()
                self.apply_model_boundary_blend()
    
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

    def draw_model(self):
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
        glRotatef(0.0, 1.0, 0.0, 0.0)
        # 3次元モデルを記述(mqoloderクラスのdrawメソッド)
        self.model.draw()

        # 照明をオフ
        if self.use_normal:
            # GL_LIGHTNING(光源0)の機能を無効にする            
            glDisable(GL_LIGHTING)
            glDisable(GL_LIGHT0)

    def read_framebuffer_rgb(self):
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        glReadBuffer(GL_BACK)
        glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, image.data)
        return cv2.flip(image, 0)

    def apply_model_boundary_blend(self):
        if not self.use_model_boundary_blend:
            return

        rendered_rgb = self.read_framebuffer_rgb()
        blended_rgb = self.model_boundary_blender.apply(
            rendered_rgb,
            self.rgb_image_for_display)
        self.glwindow.draw_image(blended_rgb)

    


    

        
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
                self.stop_recording()
        
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

        # Bでモデル境界ブレンドのON/OFF切り替え
        if action == glfw.PRESS and key == glfw.KEY_B:
            self.use_model_boundary_blend = not self.use_model_boundary_blend
            self.model_boundary_blender.enabled = self.use_model_boundary_blend
            if self.use_model_boundary_blend:
                print("モデル境界ブレンドを有効化しました")
            else:
                print("モデル境界ブレンドを無効化しました")

        # YでYOLOデバッグ描画のON/OFF切り替え
        if action == glfw.PRESS and key == glfw.KEY_Y:
            self.draw_yolo_debug_overlay = not self.draw_yolo_debug_overlay
            self.yolo_rinkaku_corrector.draw_debug_overlay = self.draw_yolo_debug_overlay
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
        import os

        today = str(datetime.date.today()).replace('-','')
        os.makedirs(self.record_output_dir, exist_ok=True)

        ext = self.record_format.lower()
        if ext not in ('mp4', 'avi'):
            ext = 'mp4'

        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        else:
            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

        filename = os.path.join(self.record_output_dir, f'video_{today}-{self.count_rec}.{ext}')
        fps = int(self.camera.capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30
        video = cv2.VideoWriter(filename, fourcc, fps, (self.width, self.height))
        self.record_output_path = filename
        print("録画を開始します..." + filename)
        return video

    def stop_recording(self):
        if self.video is not None:
            self.video.release()
            self.video = None
        if self.use_record:
            print(f"録画を終了しました: {self.record_output_path}")
        self.use_record = False
    
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

    def sample_landmark_rgb(self, rgb_image, face_landmarks, landmark_id):
        if rgb_image is None or face_landmarks is None:
            return None

        if landmark_id < 0 or landmark_id >= len(face_landmarks.landmark):
            return None

        h, w = rgb_image.shape[:2]
        landmark = face_landmarks.landmark[landmark_id]
        x = int(round(landmark.x * w))
        y = int(round(landmark.y * h))
        radius = self.color_match_patch_radius

        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)
        if x1 >= x2 or y1 >= y2:
            return None

        patch = rgb_image[y1:y2, x1:x2, :3].astype('float32')
        return np.median(patch.reshape(-1, 3), axis=0)

    def sample_landmarks_rgb(self, rgb_image, face_landmarks, landmark_ids):
        if rgb_image is None or face_landmarks is None:
            return None

        patches = []
        for landmark_id in landmark_ids:
            if landmark_id < 0 or landmark_id >= len(face_landmarks.landmark):
                continue

            h, w = rgb_image.shape[:2]
            landmark = face_landmarks.landmark[landmark_id]
            x = int(round(landmark.x * w))
            y = int(round(landmark.y * h))
            radius = self.color_match_patch_radius

            x1 = max(0, x - radius)
            x2 = min(w, x + radius + 1)
            y1 = max(0, y - radius)
            y2 = min(h, y + radius + 1)
            if x1 >= x2 or y1 >= y2:
                continue

            patches.append(rgb_image[y1:y2, x1:x2, :3].astype('float32').reshape(-1, 3))

        if not patches:
            return None

        pixels = np.vstack(patches)
        return np.median(pixels, axis=0)

    def draw_skin_color_landmarks(self, image, face_landmarks):
        h, w = image.shape[:2]
        for landmark_id in self.skin_landmarks:
            if landmark_id < 0 or landmark_id >= len(face_landmarks.landmark):
                continue

            landmark = face_landmarks.landmark[landmark_id]
            x = int(round(landmark.x * w))
            y = int(round(landmark.y * h))
            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            cv2.circle(image, (x, y), 4, (255, 80, 0), -1)
            cv2.circle(image, (x, y), 6, (255, 255, 255), 1)
            cv2.putText(image, str(landmark_id), (x + 6, y - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    def sample_texture_reference_color_stats(self, texture_path, landmark_ids=None):
        img = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if img.shape[2] == 4:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.25) as face_mesh:
            results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            return self.sample_landmarks_rgb(
                rgb_image,
                results.multi_face_landmarks[0],
                landmark_ids or self.skin_landmarks)

        h, w = rgb_image.shape[:2]
        radius = max(self.color_match_patch_radius * 2, 10)
        x1 = max(0, w // 2 - radius)
        x2 = min(w, w // 2 + radius + 1)
        y1 = max(0, h // 2 - radius)
        y2 = min(h, h // 2 + radius + 1)
        patch = rgb_image[y1:y2, x1:x2, :3].astype('float32')
        return np.median(patch.reshape(-1, 3), axis=0)

    def sample_texture_landmark_colors(self, texture_path, landmark_ids):
        img = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {}

        if img.shape[2] == 4:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.25) as face_mesh:
            results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return {}

        h, w = rgb_image.shape[:2]
        face_landmarks = results.multi_face_landmarks[0]
        radius = max(1, int(self.yolo_rinkaku_corrector.edge_landmark_color_radius))
        colors = {}

        for landmark_id in landmark_ids:
            if landmark_id < 0 or landmark_id >= len(face_landmarks.landmark):
                continue

            landmark = face_landmarks.landmark[landmark_id]
            x = int(round(landmark.x * w))
            y = int(round(landmark.y * h))
            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            x1 = max(0, x - radius)
            x2 = min(w, x + radius + 1)
            y1 = max(0, y - radius)
            y2 = min(h, y + radius + 1)
            patch = rgb_image[y1:y2, x1:x2, :3]
            if patch.size == 0:
                continue

            colors[landmark_id] = np.median(
                patch.astype('float32').reshape(-1, 3),
                axis=0)

        return colors

    def initialize_model_color_reference(self):
        self.model_reference_rgb = None
        self.model_edge_reference_rgb = {}
        self.model_edge_landmark_base_colors = {}
        self.model_edge_landmark_colors = {}
        self.smoothed_target_rgb = None

        if not self.use_model_color_match or not hasattr(self, 'model'):
            return

        for material in self.model.materials:
            if material.tex is None:
                continue
            rgb = self.sample_texture_reference_color_stats(material.tex)
            if rgb is None:
                continue
            self.model_reference_rgb = rgb
            self.model_edge_reference_rgb['right'] = self.sample_texture_reference_color_stats(
                material.tex,
                self.yolo_rinkaku_corrector.target_landmarks_right)
            self.model_edge_reference_rgb['left'] = self.sample_texture_reference_color_stats(
                material.tex,
                self.yolo_rinkaku_corrector.target_landmarks_left)
            edge_landmark_ids = list(dict.fromkeys(
                self.yolo_rinkaku_corrector.target_landmarks_right
                + self.yolo_rinkaku_corrector.target_landmarks_left
                + self.yolo_rinkaku_corrector.target_landmarks_upper))
            self.model_edge_landmark_base_colors = self.sample_texture_landmark_colors(
                material.tex,
                edge_landmark_ids)
            self.model_edge_landmark_colors = dict(self.model_edge_landmark_base_colors)
            print(f"モデル色補正基準: mode={self.model_color_match_mode}, landmarks={self.skin_landmarks}, RGB={rgb.astype(int).tolist()}")
            for mode, edge_rgb in self.model_edge_reference_rgb.items():
                if edge_rgb is not None:
                    print(f"モデル輪郭色基準({mode}): RGB={edge_rgb.astype(int).tolist()}")
            print(f"モデル輪郭ランドマーク色数: {len(self.model_edge_landmark_colors)}")
            return

        print("モデル色補正基準を取得できませんでした")

    def update_model_edge_landmark_colors(self):
        if (
                self.model_reference_rgb is None
                or self.smoothed_target_rgb is None
                or not self.model_edge_landmark_base_colors):
            return

        updated_colors = {}
        for landmark_id, base_rgb in self.model_edge_landmark_base_colors.items():
            rgb = np.asarray(base_rgb, dtype=np.float32)
            if self.model_color_match_mode == 'rgb':
                rgb = (rgb - self.model_reference_rgb) + self.smoothed_target_rgb
            else:
                rgb = self.adjust_rgb_luminance(
                    rgb,
                    self.model_reference_rgb,
                    self.smoothed_target_rgb)
            updated_colors[landmark_id] = np.clip(rgb, 0, 255)

        self.model_edge_landmark_colors = updated_colors

    def adjust_rgb_luminance(self, rgb, source_rgb, target_rgb):
        mode = self.model_color_match_mode
        if mode == 'lab_luminance':
            convert_to = cv2.COLOR_RGB2LAB
            convert_from = cv2.COLOR_LAB2RGB
        else:
            convert_to = cv2.COLOR_RGB2YCrCb
            convert_from = cv2.COLOR_YCrCb2RGB

        color = np.asarray(rgb, dtype=np.uint8).reshape(1, 1, 3)
        source = np.asarray(source_rgb, dtype=np.uint8).reshape(1, 1, 3)
        target = np.asarray(target_rgb, dtype=np.uint8).reshape(1, 1, 3)

        converted = cv2.cvtColor(color, convert_to).astype(np.float32)
        source_luma = float(cv2.cvtColor(source, convert_to)[0, 0, 0])
        target_luma = float(cv2.cvtColor(target, convert_to)[0, 0, 0])
        converted[:, :, 0] = np.clip(converted[:, :, 0] + (target_luma - source_luma), 0, 255)
        return cv2.cvtColor(converted.astype(np.uint8), convert_from)[0, 0].astype(np.float32)

    def update_model_color_match(self, rgb_image, face_landmarks):
        if not self.use_model_color_match or self.model_reference_rgb is None:
            return

        if self.angle is not None and self.smoothed_target_rgb is not None:
            yaw = self.angle[0]
            if abs(yaw) > self.color_match_update_yaw_limit:
                return

        target_rgb = self.sample_landmarks_rgb(
            rgb_image,
            face_landmarks,
            self.skin_landmarks)
        if target_rgb is None:
            return

        if self.smoothed_target_rgb is None:
            self.smoothed_target_rgb = target_rgb
        else:
            alpha = self.color_match_smoothing
            self.smoothed_target_rgb = alpha * self.smoothed_target_rgb + (1.0 - alpha) * target_rgb

        for material in self.model.materials:
            material.update_color_adjustment(
                self.model_reference_rgb,
                self.smoothed_target_rgb,
                self.model_color_match_mode)

        self.update_model_edge_landmark_colors()

    def get_yolo_edge_inpaint_rgb(self):
        left_rgb = self.model_edge_reference_rgb.get('right')
        right_rgb = self.model_edge_reference_rgb.get('left')
        if left_rgb is not None or right_rgb is not None:
            return {
                'left': left_rgb,
                'right': right_rgb,
            }
        return self.model_reference_rgb


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
        self.initialize_model_color_reference()
    
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
        print("  [B] モデル境界ブレンド切替")
        print("  [T] 表示モード切替")
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
        boundary_status = "ON" if self.use_model_boundary_blend else "OFF"
        status_text = f"Model Draw: {model_status}  Boundary: {boundary_status}"
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

        boundary_status = "ON" if self.use_model_boundary_blend else "OFF"
        boundary_color = (0, 255, 0) if self.use_model_boundary_blend else (128, 128, 128)
        cv2.putText(image, f"[B] Boundary Blend: {boundary_status}", (text_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, boundary_color, 1)
        y_offset += line_height

        # 対応点モード (Pキー)
        point_mode_names = {0: "All Points", 1: "Right Points", 2: "Left Points", 3: "Eye Points"}
        point_mode = point_mode_names.get(self.detect_stable, "Unknown")
        cv2.putText(image, f"[P] Point Mode: {point_mode}", (text_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
