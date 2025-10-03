import numpy as np
import cv2

# 
# 平面上の特徴点対応からカメラ姿勢を推定するクラス
#
class PoseEstimation:

    #
    # コンストラクタ
    #
    def __init__(self, f, u0, v0):

        # 投影行列
        self.A = np.array([[f, 0.0, u0], [0.0, f, v0], [0.0, 0.0, 1.0]], dtype = "double")

        # 歪み係数
        self.dist_coeff = np.zeros((4, 1))

        # 物体の3次元、2次元座標の初期化
        self.point_3D = np.array([])
        self.point_2D = np.array([])
        
        # 初期値
        self.rvec_init = np.zeros(3, dtype=np.float64)
        # 並進ベクトルはZ軸のプラスとマイナスが入れ替わってしまうことがあるため、初期値で固定する
        self.tvec_init = np.array([0, 0, 200], dtype=np.float64)
        
        
        # 座標軸
        self.R_ = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        
        # 検出方法
        self.use_objpoint = True
        #
        self.ready = False
    
    # 物体の3次元座標と2次元座標の対応から、カメラ姿勢を計算
    def compute_camera_pose(self, point_3D, point_2D, use_objpoint):
        if self.ready:
            self.use_objpoint = use_objpoint
            if self.use_objpoint:
            # 6点以上の空間座標を使用
            # solvepnp()...
            # 入力：位置、観測座標の配列、カメラの内部パラメータ行列、
            #       歪み係数、初期値、flag(ポーズ計算方法)
            # 出力：カメラの外部パラメータ(回転ベクトル、並進ベクトル)
                success, rvec, tvec = cv2.solvePnP(point_3D,
                                                point_2D,
                                                self.A,
                                                self.dist_coeff,
                                                rvec = self.rvec_init,
                                                tvec = self.tvec_init,
                                                useExtrinsicGuess = True,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        
            # 4点の平面座標を使用
            else:
                success, rvec, tvec = cv2.solvePnP(point_3D,
                                                point_2D,
                                                self.A,
                                                self.dist_coeff,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
            
            self.rvec_last = rvec
            self.tvec_last = tvec
            # 回転行列の計算
            R = cv2.Rodrigues(rvec)[0] 
            # 座標軸を変更
            R_ = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
            R = np.dot(R_, R)
            t = np.dot(R_, tvec)
            r = np.dot(R_, rvec)
            
            return True, R, t, r
        else:
            return False, None, None, None
    
    # 顔の方向ベクトルを計算
    # なぜかx、y座標ともにopencvの軸に反転してしまう。原因は？
    def compute_head_vector(self):
        nose_end_point2D, _ = cv2.projectPoints(
            np.array([0.0, 0.0, 200.0]),
            self.rvec_last,
            self.tvec_last,
            self.A,
            self.dist_coeff)
        
        vector = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        return vector

    # 顔のオイラー角度を計算
    def compute_head_angle(self, R, t):
        T = np.reshape(t,(3,1))
        mat = np.hstack((R,T))
        (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
        
        angle = (int(eulerAngles[1]),int(eulerAngles[0]),int(eulerAngles[2]))
        return angle
        
    # モデルの3次元座標をセット    
    def set_3D_points(self, point_3D):
        self.point_3D = point_3D
        self.ready = True