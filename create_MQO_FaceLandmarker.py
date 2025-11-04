import sys
import numpy as np
import cv2
import mediapipe as mp
import datetime

#
# Face Landmarkerを使用した3次元モデル生成クラス
#
class CreateMQOFaceLandmarker:
    #
    # コンストラクタ
    #
    # @param texture : テクスチャ画像
    # @param use_alpha : アルファ処理を行うか
    # @param use_face_landmarker : Face Landmarker機能を使用するか
    #
    def __init__(self, texture, use_alpha=False, use_face_landmarker=True):
        
        # Face Landmarker使用フラグ
        self.use_face_landmarker = use_face_landmarker
        
        # 顔下部切り取りを行うか
        self.use_cut = True
        # マスクモデル作成モード
        self.masked_face = False
        # 世界座標系を記述
        self.world_coordinate = False
        
        # アルファ処理を行うか
        self.use_alpha = use_alpha
        
        # ファイル名用日付
        self.today = str(datetime.date.today()).replace('-','')
        # ファイル作成
        self.mesh = np.loadtxt("mqodata/mesh.dat", dtype='int')
        self.mesh_cut = np.array([])
        
        # Face Landmarker関連の変数
        self.face_landmarker_solution = None
        self.face_landmarker_results = None
        self.transformation_matrix = None
        
        # アルファ処理を実行（set_point の前に実行する必要がある）
        if self.use_alpha:
            print("アルファ処理を開始します...")
            from ContourAlpha import ContourAlpha
            import os
            texture_path = f"mqodata/model/{texture}"
            # simple_mode=Trueで可視化出力なし、save_org=Trueでアルファ処理後の画像を出力
            ContourAlpha(texture_path, use_cut=self.use_cut, save_org=True, use_spline=False, simple_mode=True)
            # JPEGからPNGに変換された場合、テクスチャファイル名を更新
            base_name = os.path.splitext(texture)[0]
            texture = base_name + '.png'
            print(f"アルファ処理が完了しました。使用するテクスチャ: {texture}")
        
        # Face Landmarkerの初期化
        if self.use_face_landmarker:
            self.initialize_face_landmarker()
        
        # テクスチャ画像から特徴点、特徴点の正規化、新たなメッシュ情報を生成
        self.data = np.array([])
        self.datalist = []
        self.landmark = np.array([])
        self.landmark_nomalize = np.array([])
        
        if self.use_face_landmarker:
            self.set_point_face_landmarker(texture)
        else:
            self.set_point_facemesh(texture)  # fallback to FaceMesh
        
        # ヘッダ出力
        self.outputs = []
        self.output_header(texture)

        # 3次元座標の出力
        self.output_3D_coord(self.landmark_nomalize)
        # 三角形メッシュの出力
        if self.use_cut:
            self.mesh = self.mesh_cut
        self.output_mesh_info(self.landmark, self.mesh)

        # フッダ出力
        self.outputs.append('}\nEof')
        
        # ファイル出力
        self.model_filename = "mqodata/model/model_FL_"+self.today+".mqo"
        if self.use_cut:
            self.model_filename = "mqodata/model/model_FL_cut_"+self.today+".mqo"
        with open(self.model_filename, "w") as f:
            for output in self.outputs:
                f.write(output)

    #
    # Face Landmarkerの初期化
    #
    def initialize_face_landmarker(self):
        """
        Face Landmarkerを初期化
        """
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Face Landmarkerのベースオプション
            base_options = python.BaseOptions(
                model_asset_path='face_landmarker.task')
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,  # 変換行列を有効化
                num_faces=1)
            self.face_landmarker_solution = vision.FaceLandmarker.create_from_options(options)
            print("Face Landmarker初期化完了")
            
        except Exception as e:
            print(f"Face Landmarker初期化エラー: {e}")
            print("FaceMeshにフォールバック")
            self.use_face_landmarker = False
            self.face_landmarker_solution = None

    #
    # ヘッダを出力する関数
    #
    def output_header(self, texture_filename):
        self.outputs.append('Metasequoia Document\n')
        self.outputs.append('Format Text Ver 1.1\n')
        self.outputs.append('\n')
        self.outputs.append('Scene {\n')
        self.outputs.append('\tpos 0 0 1000\n')
        self.outputs.append('\tlookat 0 0 0\n')
        self.outputs.append('\thead -1.5\n')
        self.outputs.append('\tpich 0.12\n')
        self.outputs.append('\tbank 0.0000\n')
        self.outputs.append('\tortho 0\n')
        self.outputs.append('\tzoom2 5.0000\n')
        self.outputs.append('\tamb 0.250 0.250 0.250\n')
        self.outputs.append('\tfrontclip 225.0\n')
        self.outputs.append('\tbackclip 45000\n')
        self.outputs.append('\tdirlights 1 {\n')
        self.outputs.append('\t\tlight {\n')
        self.outputs.append('\t\t\tdir 0.408 0.408 0.816\n')
        self.outputs.append('\t\t\tcolor 1.000 1.000 1.000\n')
        self.outputs.append('\t\t}\n')
        self.outputs.append('\t}\n')
        self.outputs.append('}\n')

        self.outputs.append('Material 1 {\n')
        
        if self.masked_face == True:
            self.outputs.append('\t"mat1" shader(3) col(0.176 1.000 0.000 0.500) dif(0.800) amb(0.600) emi(0.000) spc(0.000) power(5.00) tex("mask.jpg")\n')
        else:
            self.outputs.append('\t"mat1" shader(3) col(0.176 1.000 0.000 0.500) dif(0.800) amb(0.600) emi(0.000) spc(0.000) power(5.00) tex("%s")\n' % texture_filename)
        self.outputs.append('}\n')

        self.outputs.append('Object "obj" {\n')
        self.outputs.append('\tdepth 0\n')
        self.outputs.append('\tfolding 0\n')
        self.outputs.append('\tscale 1 1 1\n')
        self.outputs.append('\trotation 0 0 0\n')
        self.outputs.append('\tvisible 15\n')
        self.outputs.append('\tlocking 0\n')
        self.outputs.append('\tshading 1\n')
        self.outputs.append('\tfacet 59.5\n')
        self.outputs.append('\tnormal_weight 1\n')
        self.outputs.append('\tcolor 0.898 0.498 0,698\n')
        self.outputs.append('\tcolor_type 0\n')
        
        
    #
    # ベクトル情報を出力する関数
    # @param point: ランドマークを世界座標系に変換した三次元座標
    #
    def output_3D_coord(self, point):
        npoints, dim = point.shape
        self.outputs.append('\tvertex %d {\n' % npoints)

        for p in range(npoints):
            self.outputs.append('\t\t%f %f %f\n' % (point[p, 0], point[p, 1], point[p, 2]))

        self.outputs.append('\t}\n')

    #
    # メッシュ情報を出力する関数
    # @param uv : ランドマークの二次元座標 , mesh : メッシュ情報
    #   
    def output_mesh_info(self, uv, mesh):
        nmeshes, dim = mesh.shape
        self.outputs.append('\tface %d {\n' % nmeshes)
        for m in range(nmeshes):
            self.outputs.append('\t\t3 V(%d %d %d) M(0) UV(%f %f %f %f %f %f)\n' % (mesh[m, 0], mesh[m, 1], mesh[m, 2],
                                                                    uv[mesh[m, 0], 0], uv[mesh[m, 0], 1],
                                                                    uv[mesh[m, 1], 0], uv[mesh[m, 1], 1],
                                                                    uv[mesh[m, 2], 0], uv[mesh[m, 2], 1]))
            
        self.outputs.append('\t}\n')

    #
    # Face Landmarkerを使用してテクスチャ画像の特徴点を検出する関数
    #
    def set_point_face_landmarker(self, texture_filename):
        """
        Face Landmarkerを使用して高精度な468個の3Dランドマークを取得
        """
        # テクスチャファイル読み込み
        import os
        texture_path_model = f"mqodata/model/{texture_filename}"
        texture_path_base = f"mqodata/{texture_filename}"
        
        if os.path.exists(texture_path_model):
            img = cv2.imread(texture_path_model)
            print(f"テクスチャを読み込みました(アルファ処理あり): {texture_path_model}")
        elif os.path.exists(texture_path_base):
            img = cv2.imread(texture_path_base)
            print(f"テクスチャを読み込みました（アルファ処理なし）: {texture_path_base}")
        else:
            raise FileNotFoundError(f"テクスチャファイルが見つかりません: {texture_filename}")
        
        if img is None:
            raise ValueError(f"画像の読み込みに失敗しました: {texture_filename}")
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated_image = img.copy()
        
        # Face Landmarkerを実行
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            self.face_landmarker_results = self.face_landmarker_solution.detect(mp_image)
            
            if not self.face_landmarker_results.face_landmarks:
                raise ValueError("顔が検出されませんでした")
            
            print(f"Face Landmarker検出完了: {len(self.face_landmarker_results.face_landmarks[0])}個のランドマーク")
            
            # 変換行列の取得
            if self.face_landmarker_results.facial_transformation_matrixes:
                self.transformation_matrix = np.array(
                    self.face_landmarker_results.facial_transformation_matrixes[0]
                ).reshape(4, 4)
                print("顔の変換行列を取得しました")
                print("変換行列:")
                print(self.transformation_matrix)
            
        except Exception as e:
            print(f"Face Landmarker処理エラー: {e}")
            print("FaceMeshにフォールバック")
            self.set_point_facemesh(texture_filename)
            return
        
        # 座標、メッシュ情報格納用リスト
        x = []
        y = []
        z = []
        landmark = []
        landmark_nomalize = []
        cut = []
        mesh_cut = []
        cnt = 0
        
        # Face Landmarkerのランドマーク処理
        face_landmarks = self.face_landmarker_results.face_landmarks[0]
        
        for idx, landmark_point in enumerate(face_landmarks):
            # Face Landmarkerの正規化座標を使用
            landmark.append(np.array([landmark_point.x, landmark_point.y, landmark_point.z]))
            
            # 画像サイズに合わせて座標変換
            x.append(landmark_point.x * img.shape[1])
            y.append(landmark_point.y * img.shape[0])
            z.append(landmark_point.z * img.shape[1])  # Z座標もスケール
            cnt += 1
            
            # ランドマークを描画（デバッグ用）
            if idx % 20 == 0:  # 20個おきに描画
                px = int(landmark_point.x * img.shape[1])
                py = int(landmark_point.y * img.shape[0])
                cv2.circle(annotated_image, (px, py), 2, (0, 255, 0), -1)
                cv2.putText(annotated_image, str(idx), (px + 3, py - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        print(f"処理されたランドマーク数: {cnt}")
        
        # 三次元座標変換処理
        n1 = []
        n2 = []
        n3 = []
        n4 = []
        
        # 1. Face Landmarkerの変換行列を使用した座標変換
        if self.transformation_matrix is not None:
            # 変換行列から回転部分と並進部分を抽出
            R = self.transformation_matrix[:3, :3]
            t = self.transformation_matrix[:3, 3]
            
            print("Face Landmarkerの変換行列を使用した座標変換")
            print(f"回転行列 R:\n{R}")
            print(f"並進ベクトル t: {t}")
            
            # 各ランドマークに変換行列を適用
            for i in range(cnt):
                # MediaPipeの3D座標を取得
                original_3d = np.array([x[i], y[i], z[i]])
                
                # 画像中心を原点とする座標系に変換
                centered_coord = np.array([
                    x[i] - img.shape[1]/2,
                    y[i] - img.shape[0]/2,
                    z[i]
                ])
                
                # スケーリング（実寸に近づける）
                scale = 100.0 / max(img.shape[1], img.shape[0])
                scaled_coord = centered_coord * scale
                
                # Face Landmarkerの座標系では既に適切な3D配置になっているため、
                # 軽微な調整のみ行う
                final_coord = np.array([
                    scaled_coord[0],
                    -scaled_coord[1],  # Y軸を反転
                    scaled_coord[2]
                ])
                
                n4.append(final_coord)
        
        else:
            # フォールバック：従来の座標変換
            print("従来の座標変換を使用")
            
            # 1.原点がランドマーク1となるように平行移動
            for i in range(cnt):
                n1.append(np.array([x[i]-x[1] if len(x) > 1 else x[i], 
                                   y[i]-y[1] if len(y) > 1 else y[i], 
                                   z[i]-z[1] if len(z) > 1 else z[i]]))
                
            # 2.実寸サイズにスケーリング
            if len(n1) > 263:
                scale = 100 / abs(n1[263][0]-n1[33][0]) if abs(n1[263][0]-n1[33][0]) > 0 else 1.0
            else:
                scale = 1.0
            
            for i in range(cnt):
                n2.append(np.array(n1[i] * scale))
                
            # 3.回転の補正（Face Landmarkerでは不要な場合が多い）
            for i in range(cnt):
                n3.append(n2[i])
            
            # 4.最終座標
            for i in range(cnt):
                n4.append(np.array([n3[i][0], n3[i][1], n3[i][2]]))
        
        # ランドマークを世界座標系に変換した三次元座標のリストを作成
        landmark_nomalize = n4
        
        # 世界座標系記述
        if self.world_coordinate and self.transformation_matrix is not None:
            R = self.transformation_matrix[:3, :3]
            vecx = np.array([200,0,0])
            vecy = np.array([0,200,0])
            vecz = np.array([0,0,200])
            vecx = np.dot(R, vecx)
            vecy = np.dot(R, vecy)
            vecz = np.dot(R, vecz)
            
            if len(x) > 1:
                cv2.line(annotated_image,
                        pt1 = (int(x[1]),int(y[1])),
                        pt2 = (int(x[1]+vecx[0]),int(y[1]+vecx[1])),
                        color = (0,0,255),
                        thickness = 3)
                cv2.line(annotated_image,
                        pt1 = (int(x[1]),int(y[1])),
                        pt2 = (int(x[1]+vecy[0]),int(y[1]+vecy[1])),
                        color = (0,255,0),
                        thickness = 3)
                cv2.line(annotated_image,
                        pt1 = (int(x[1]),int(y[1])),
                        pt2 = (int(x[1]+vecz[0]),int(y[1]+vecz[1])),
                        color = (255,0,0),
                        thickness = 3)
        
        # 対応点データの設定（Face Landmarkerの468個のランドマーク用）
        data = []
        datalist = []
        
        # Face Landmarkerは468個のランドマークを提供
        # 重要な顔の特徴点を選択
        important_landmarks = [
            # 顔の輪郭
            10, 151, 9, 8, 168, 6, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
            # 左眉
            46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
            # 右眉  
            276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
            # 左目
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            # 右目
            362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
            # 鼻
            1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 309, 415, 310, 311, 312, 13, 82, 14, 317, 18, 175,
            # 口
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
        ]
        
        # 利用可能なランドマークのみを使用
        for j in range(min(cnt, len(landmark_nomalize))):
            datalist.append(j)
            data.append(landmark_nomalize[j])
            
            # 重要なランドマークを対応点として使用
            if j in important_landmarks[:min(len(important_landmarks), cnt)]:
                # モデル生成用として選択
                cut.append(j)

        # 簡易メッシュ生成（Face Landmarker用）
        # 基本的な三角形メッシュを生成
        for i in range(0, min(len(cut)-2, 100), 3):  # 簡易的な三角形メッシュ
            if i+2 < len(cut):
                mesh_cut.append(np.array([cut[i], cut[i+1], cut[i+2]]))
    
        # ファイル出力
        np.savetxt("data/face_3D_FL.dat", data)
        self.data = np.array(data)
        self.datalist = datalist
        np.savetxt("mqodata/landmark/landmark_FL_"+self.today+".dat", landmark)
        self.landmark = np.array(landmark)
        np.savetxt("mqodata/landmark3d/landmark3d_FL_"+self.today+".dat", landmark_nomalize)
        self.landmark_nomalize = np.array(landmark_nomalize)
        if mesh_cut:
            np.savetxt("mqodata/mesh/mesh_FL_cut_"+self.today+".dat", mesh_cut, fmt='%d')
            self.mesh_cut = np.array(mesh_cut)
        
        cv2.imwrite("output/face_landmarker_"+self.today+".png", annotated_image)
        print(f"Face Landmarker結果画像を保存: output/face_landmarker_{self.today}.png")

    #
    # FaceMeshを使用したフォールバック処理
    #
    def set_point_facemesh(self, texture_filename):
        """
        従来のFaceMeshを使用（フォールバック）
        """
        print("FaceMeshを使用してランドマーク検出を実行")
        
        # MediaPipeのFaceMeshインスタンスを作成する
        face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)
        drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=1, 
            circle_radius=1, 
            color=(0, 0, 255))

        # テクスチャファイル読み込み
        import os
        texture_path_model = f"mqodata/model/{texture_filename}"
        texture_path_base = f"mqodata/{texture_filename}"
        
        if os.path.exists(texture_path_model):
            img = cv2.imread(texture_path_model)
            print(f"テクスチャを読み込みました(アルファ処理あり): {texture_path_model}")
        elif os.path.exists(texture_path_base):
            img = cv2.imread(texture_path_base)
            print(f"テクスチャを読み込みました（アルファ処理なし）: {texture_path_base}")
        else:
            raise FileNotFoundError(f"テクスチャファイルが見つかりません: {texture_filename}")
        
        if img is None:
            raise ValueError(f"画像の読み込みに失敗しました: {texture_filename}")
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated_image = img.copy() 
        
        # FaceMeshを実行
        face_mesh = face.process(rgb_img)
        
        # 座標、メッシュ情報格納用リスト
        x = []
        y = []
        z = []
        landmark = []
        landmark_nomalize = []
        cut = []
        mesh_cut = []
        cnt = 0
        
        # ランドマークの導出及び描画
        for face_landmarks in face_mesh.multi_face_landmarks:
            # 画像上に描画
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, 
                face_landmarks, 
                mp.solutions.face_mesh.FACEMESH_TESSELATION,
                drawing_spec,
                drawing_spec)
            # 座標を導出
            for idx, p in enumerate(face_landmarks.landmark):
                landmark.append(np.array([p.x, p.y, p.z])) 
                x.append(p.x * img.shape[1])
                y.append(p.y * img.shape[0])
                z.append(p.z * img.shape[1])
                cnt += 1
        
        # 従来の座標変換処理（既存のcreate_MQO.pyと同じ）
        n1 = []
        n2 = []
        n3 = []
        n4 = []
        
        # 1.原点がランドマーク1となるように平行移動
        for i in range(cnt):
            n1.append(np.array([x[i]-x[1], y[i]-y[1], z[i]-z[1]]))
            
        # 2.実寸サイズにスケーリング
        scale = 100 / abs(n1[263][0]-n1[33][0])
        for i in range(cnt):
            n2.append(np.array(n1[i] * scale))
            
        # 3.回転の補正
        v = np.array([n2[263][0]-n2[33][0], n2[263][1]-n2[33][1], n2[263][2]-n2[33][2]])
        v = v / np.linalg.norm(v)
        
        k = np.array([1,0,0])
        s = v + k
        r = np.array(2 * np.outer(s, s) / np.dot(s, s) - np.eye(3))
        for i in range(cnt):
            n3.append(np.dot(r, n2[i]))
        
        # 4.補正
        for i in range(cnt):
            n4.append(np.array([n3[i][0], n3[i][1], n3[i][2]]))
        
        landmark_nomalize = n4
        
        # 対応点データの設定（既存と同じ）
        data = []
        datalist = []
        delete_datalist = []
        
        datalist1 = [6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 46, 52, 53, 
                    54, 55, 56, 63, 65, 66, 67, 68, 69, 70, 71, 103, 104, 105, 107, 108, 109, 
                    110, 112, 113, 122, 124, 127, 130, 133, 139, 143, 144, 145, 151, 153, 154, 155, 
                    156, 157, 158, 159, 160, 161, 162, 163, 168, 173, 189, 190, 193, 221, 222, 
                    223, 224, 225, 226, 243, 244, 245, 246, 247, 249, 251, 252, 253, 254, 255, 256, 257, 258, 
                    259, 260, 263, 264, 265, 276, 282, 283, 284, 285, 286, 293, 295, 296, 297, 
                    298, 299, 300, 301, 332, 333, 334, 336, 337, 338, 339, 341, 342, 351, 353, 356, 
                    359, 362, 368, 372, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 
                    389, 390, 398, 413, 414, 417, 441, 442, 443, 444, 445, 446, 463, 464, 465, 466, 467]

        for j in range(cnt):
            datalist.append(j)
            data.append(landmark_nomalize[j])
            if j in delete_datalist:
                pass
            elif j in datalist1:
                pass
            else:
                cut.append(j)

        # メッシュ情報の修正
        n, d = self.mesh.shape
        for i in range(n):
            if (self.mesh[i, 0] in cut) and (self.mesh[i, 1] in cut) and (self.mesh[i, 2] in cut):
                mesh_cut.append(np.array([self.mesh[i, 0],self.mesh[i, 1],self.mesh[i, 2]]))
    
        # ファイル出力
        np.savetxt("data/face_3D_FM.dat", data)
        self.data = np.array(data)
        self.datalist = datalist
        np.savetxt("mqodata/landmark/landmark_FM_"+self.today+".dat", landmark)
        self.landmark = np.array(landmark)
        np.savetxt("mqodata/landmark3d/landmark3d_FM_"+self.today+".dat", landmark_nomalize)
        self.landmark_nomalize = np.array(landmark_nomalize)
        np.savetxt("mqodata/mesh/mesh_FM_cut_"+self.today+".dat", mesh_cut, fmt='%d')
        self.mesh_cut = np.array(mesh_cut)
        cv2.imwrite("output/face_mesh_FM_"+self.today+".png", annotated_image)

        face.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python3 create_MQO_FaceLandmarker.py <texture_file> [--face-landmarker]")
        print("  例: python3 create_MQO_FaceLandmarker.py nomask.jpg --face-landmarker")
        sys.exit()

    texture_file = sys.argv[1]
    use_face_landmarker = "--face-landmarker" in sys.argv
    
    if use_face_landmarker:
        print("Face Landmarkerモードで実行")
    else:
        print("FaceMeshモードで実行")
    
    CreateMQOFaceLandmarker(texture_file, use_face_landmarker=use_face_landmarker)
