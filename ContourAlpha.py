import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # または 'Agg', 'Qt5Agg' など
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
from scipy.interpolate import UnivariateSpline
#
# 輪郭のアルファ変更クラス
#
class ContourAlpha:

    #
    # コンストラクタ
    #
    def __init__(self, texture, use_cut, save_org, use_spline, simple_mode=False):
        # アルファ値を変更する範囲（デフォルト:20）
        self.scope = 20
        # アルファ値の増分
        self.diff_alpha = 255 / self.scope
        # スプライン補間を行う
        self.use_spline = use_spline
        # シンプルモード（可視化出力なし）
        self.simple_mode = simple_mode
        # グラフの日本語フォント
        if not simple_mode:
            font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # 適切なフォントパスを指定
            self.font_prop = fm.FontProperties(fname=font_path)

        # 画像を読み込む
        img = cv2.imread(texture)
        # オリジナルコピー
        if save_org:
            copy_texture_name = texture.split('.')
            cv2.imwrite(copy_texture_name[0]+"_noalpha.png", img)
            print(copy_texture_name[0]+"_noalpha.png")
        self.h, self.w, channel = img.shape

        # アルファチャンネルがない場合、追加する
        if channel != 4:
            b, g, r = cv2.split(img)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255  # 完全な不透明度
            img = cv2.merge((b, g, r, alpha))

        # シンプルモードでは可視化用の画像コピーを作成しない
        if not self.simple_mode:
            # 輪郭のランドマークプロット画像
            self.contour_plot_img = img.copy()
            # 高次多項式のプロット画像
            self.contour_polynomial_img = img.copy()
            # 接線と垂直な直線のプロット画像
            self.tangent_normal_img = img.copy()
            # グラフ定義
            self.figure1, self.ax1 = plt.subplots()
            self.ax1.set_xlabel('x')
            self.ax1.set_ylabel('y')
            self.figure2, self.ax2 = plt.subplots()
            self.ax2.set_xlabel('x')
            self.ax2.set_ylabel('y')

        # RGBに変換
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # MediaPipeのFaceMeshソリューションを初期化
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        # 顔を検出
        self.results = face_mesh.process(rgb_img)
        # 顔の輪郭に対応するランドマークインデックス
        self.face_contour_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
        contour_indices_num = len(self.face_contour_indices)
        if use_cut:
            # 顔の下半分を切り取った場合の断面に対応するランドマークインデックス
            #self.cross_section_indices = [227, 116, 111, 31, 228, 229, 230, 231, 232, 233, 244, 245, 122, 6, 351, 465, 464, 453, 452, 451, 450, 449, 448, 261, 340, 345, 447]
            #self.cross_section_indices = [227, 116, 117, 118, 119, 120, 47, 217, 174, 196, 197, 419, 399, 437, 277, 349, 348, 347, 346, 345, 447]
            self.cross_section_indices = [227, 116, 111, 31, 228, 229, 230, 231, 232, 233, 244, 245, 188, 196, 197, 419, 412, 465, 464, 453, 452, 451, 450, 449, 448, 261, 340, 345, 447]
            #self.cross_section_indices = [137, 123, 50, 101, 100, 126, 198, 236, 3, 195, 248, 456, 420, 355, 329, 330, 280, 352, 366, ]
        else:
            # 顔の上半分の輪郭に対応するランドマークインデックス
            self.cross_section_indices = [234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454]
        cross_indices_num = len(self.cross_section_indices)
        # 検出結果を処理
        self.contour_x_list = []
        self.contour_y_list = []
        self.cross_x_list = []
        self.cross_y_list = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                # 輪郭
                for idx in self.face_contour_indices:
                    landmark = face_landmarks.landmark[idx]
                    # ランドマークの座標を画像の座標に変換
                    x, y = int(landmark.x * self.w), int(landmark.y * self.h)
                    self.contour_x_list.append(x)
                    self.contour_y_list.append(y)
                # 断面
                for idx in self.cross_section_indices:
                    landmark = face_landmarks.landmark[idx]
                    # ランドマークの座標を画像の座標に変換
                    x, y = int(landmark.x * self.w), int(landmark.y * self.h)
                    self.cross_x_list.append(x)
                    self.cross_y_list.append(y)
                    """
                    while True:
                        if not x in self.cross_x_list:
                            self.cross_x_list.append(x)
                            self.cross_y_list.append(y)
                            break
                        else:
                            x += 1
                    """
        # 高次多項式の係数を求める
        contour_x_list = np.array(self.contour_x_list)
        contour_y_list = np.array(self.contour_y_list)
        cross_x_list = np.array(self.cross_x_list)
        cross_y_list = np.array(self.cross_y_list)
        contour_coefficients = np.polyfit(contour_x_list, contour_y_list, contour_indices_num-1)
        cross_coefficients = np.polyfit(cross_x_list, cross_y_list, cross_indices_num-1)
        # 高次多項式のフィッティング
        self.contour_polynomial = np.poly1d(contour_coefficients)
        self.cross_polynomial = np.poly1d(cross_coefficients)
        # スプライン補間のフィッティング
        if self.use_spline:
            self.spline = UnivariateSpline(cross_x_list, cross_y_list, s=2)
        # x座標の走査範囲
        self.x_range = range(contour_x_list[0], contour_x_list[contour_indices_num-1]+1)

        # 高次多項式の微分を求める
        self.derivative = self.contour_polynomial.deriv()
        # アルファ値を変更するエリア
        self.alpha_area = []
        for _ in range(self.w):
            self.alpha_area.append([])
        # 輪郭のアルファを概ね変更
        for x in self.x_range:
            self.contour_alpha_changer(x, img)
        # 隙間を埋める
        for j in range(self.h):
            for i in range(self.w):
                if len(self.alpha_area[i]) != 0 and min(self.alpha_area[i]) < j and max(self.alpha_area[i]) > j and img[j, i, 3] == 255:
                    img[j, i, 3] = img[j, i-1, 3]
        # 断面のアルファを変更(コメントアウト: アルファ処理のみの画像を生成)
        # for x in self.x_range:
        #     self.cross_alpha_changer(x, img)

        # アルファ処理後のテクスチャ出力
        if save_org:
            # アルファ処理後の画像を別名で保存（確認用）
            output_filename = 'mqodata/model/alpha_processed_output.png'
            cv2.imwrite(output_filename, img)
            print(f"アルファ処理後の画像を保存しました: {output_filename}")
            # 元のファイルも上書き（.jpgの場合は.pngに変換してアルファチャンネルを保持）
            import os
            base_name = os.path.splitext(texture)[0]
            png_texture = base_name + '.png'
            cv2.imwrite(png_texture, img)
            print(f"アルファチャンネル付き画像を保存しました: {png_texture}")
        else:
            # .jpgの場合は.pngに変換してアルファチャンネルを保持
            import os
            if texture.lower().endswith('.jpg') or texture.lower().endswith('.jpeg'):
                base_name = os.path.splitext(texture)[0]
                texture = base_name + '.png'
            cv2.imwrite(texture, img)
            print(f"アルファ処理済みテクスチャを保存: {texture}")
        
        # シンプルモードではここで処理終了（可視化処理をスキップ）
        if self.simple_mode:
            print(f"アルファ処理完了: {texture}")
            return

    # 輪郭のアルファ値を変更する関数
    def contour_alpha_changer(self, x, img):
        # 接線の傾き
        slope_tangent = self.derivative(x)
        # 接線に垂直な傾き
        slope_normal = -1 / slope_tangent
        # 輪郭のy座標
        now_y = self.contour_polynomial(x)
        next_y = self.contour_polynomial(x+1)

        # 顎部分
        if np.abs(slope_normal) >= 1.0:
            for n, j in enumerate(range(int(now_y)-self.scope, int(now_y)+self.scope)):
                i = 1/slope_normal * (j - now_y) + x
                if j < int(now_y):
                    img[j, int(i), 3] = self.diff_alpha * (self.scope-n)
                else:
                    img[j, int(i), 3] = 0
                self.alpha_area[int(i)].append(j)
        # 右側面
        elif slope_normal < 0.0:
            for y in range(int(now_y), int(next_y)):
                for n, i in enumerate(range(int(x)-self.scope, int(x)+self.scope)):
                    j = slope_normal * (i - x) + y
                    if i > int(x):
                        img[int(j), i, 3] = self.diff_alpha * (n-self.scope)
                    else:
                        img[int(j), i, 3] = 0
                    self.alpha_area[i].append(int(j))
        # 左側面
        else:
            for y in range(int(now_y), int(next_y), -1):
                for n, i in enumerate(range(int(x)-self.scope, int(x)+self.scope)):
                    j = slope_normal * (i - x) + y
                    if i < int(x):
                        img[int(j), i, 3] = self.diff_alpha * (self.scope-n)
                    else:
                        img[int(j), i, 3] = 0
                    self.alpha_area[i].append(int(j))
    
    # 断面のアルファ値を変更する関数
    def cross_alpha_changer(self, x, img):
        # 断面のy座標
        if self.use_spline:
            y = self.spline(x)
        else:
            y = self.cross_polynomial(x)

        for j in range(int(y) - 5, int(y) + 10):
            if 0 <= j < self.h and 0 <= x < self.w:
                img[int(j), x, 3] = 0
        for n, j in enumerate(range(int(y) + 10, int(y + self.scope) + 10)):
            if 0 <= j < self.h and 0 <= x < self.w:
                if img[int(j), x, 3] > self.diff_alpha * n:
                    img[int(j), x, 3] = self.diff_alpha * n

    # 輪郭と断面のランドマークをプロットする関数
    def plot_contour_cross(self):
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                # 輪郭
                for idx in self.face_contour_indices:
                    landmark = face_landmarks.landmark[idx]
                    # ランドマークの座標を画像の座標に変換
                    x, y = int(landmark.x * self.w), int(landmark.y * self.h)
                    cv2.circle(self.contour_plot_img, (x, y), 2, (0, 255, 0, 255), -1)
                # 断面
                for idx in self.cross_section_indices:
                    landmark = face_landmarks.landmark[idx]
                    # ランドマークの座標を画像の座標に変換
                    x, y = int(landmark.x * self.w), int(landmark.y * self.h)
                    cv2.circle(self.contour_plot_img, (x, y), 2, (0, 0, 255, 255), -1)
        cv2.imwrite('1_landmark_plot.png', self.contour_plot_img)

    # 高次多項式をプロットする関数
    def plot_polynomial(self):
        contour_x = []
        contour_y = []
        if self.use_spline:
            cross_x = []
            cross_y = []
            spline_x = []
            spline_y = []
        else:
            cross_x = []
            cross_y = []
        for x in self.x_range:
            y = self.contour_polynomial(x)
            contour_x.append(x)
            contour_y.append(y)
            cv2.circle(self.contour_polynomial_img, (x, int(y)), 1, (0, 255, 0, 255), -1)
            if self.use_spline:
                y = self.cross_polynomial(x)
                cross_x.append(x)
                cross_y.append(y)
                cv2.circle(self.contour_polynomial_img, (x, int(y)), 1, (0, 0, 255, 255), -1)
                y = self.spline(x)
                spline_x.append(x)
                spline_y.append(y)
                cv2.circle(self.contour_polynomial_img, (x, int(y)), 1, (255, 0, 0, 255), -1)
            else:
                y = self.cross_polynomial(x)
                cross_x.append(x)
                cross_y.append(y)
                cv2.circle(self.contour_polynomial_img, (x, int(y)), 1, (0, 0, 255, 255), -1)
        cv2.imwrite('2_polynomial.png', self.contour_polynomial_img)
        # グラフ作成
        p1,  = self.ax1.plot(contour_x, contour_y, color='green')
        if self.use_spline:
            p2,  = self.ax1.plot(cross_x, cross_y, color='red')
            p3,  = self.ax1.plot(spline_x, spline_y, color='blue')
        else:
            p2,  = self.ax1.plot(cross_x, cross_y, color='red')
        self.ax1.set_title('高次多項式', fontproperties=self.font_prop)
        self.ax1.set_ylim(round(max(contour_y), -1)+10, round(min(cross_y), -1)-10)
        if self.use_spline:
            self.ax1.legend([p1, p2, p3], ['顔の輪郭', '高次多項式', 'スプライン補間'], loc='center', prop=self.font_prop)
        else:
            self.ax1.legend([p1, p2], ['顔の輪郭', 'モデルの断面'], loc='center', prop=self.font_prop)
        self.figure1.savefig('2_polynomial_graph.png')

    # 断面に垂直にプロットする関数
    def plot_contour_cross_normal(self):
        if self.use_spline:
            spline_x = []
            spline_y = []
        else:
            cross_x = []
            cross_y = []
        for x in self.contour_x_list:
            if self.use_spline:
                y = self.spline(x)
                spline_x.append(x)
                spline_y.append(y)
            else:
                y = self.cross_polynomial(x)
                cross_x.append(x)
                cross_y.append(y)
            vertical_x = []
            vertical_y = []
            for incremental in range(-5, 30):
                j = int(y) + incremental
                vertical_x.append(x)
                vertical_y.append(j)
                cv2.circle(self.contour_polynomial_img, (x, j), 1, (0, 0, 0, 255), -1)
            p1,  = self.ax1.plot(vertical_x, vertical_y, color='black')

        cv2.imwrite('3_alpha-range_cross-section.png', self.contour_polynomial_img)
        # グラフ作成
        self.ax1.set_title('断面のアルファ範囲', fontproperties=self.font_prop)
        self.ax1.legend([p1], ['断面に垂直な線'], loc='center', prop=self.font_prop)
        self.figure1.savefig('3_alpha-range_cross-section_graph.png')

    # 接線をプロットする関数
    def plot_tangent(self):
        max = 0
        for x in self.contour_x_list:
            # 接線の傾き
            slope_tangent = self.derivative(x)
            # y座標
            y = self.contour_polynomial(x)
            tangent_x = []
            tangent_y = []
            # 顎部分
            if np.abs(slope_tangent) <= 1.0:
                for i in range(x-30, x+30):
                    j = slope_tangent * (i - x) + y
                    cv2.circle(self.tangent_normal_img, (i, int(j)), 1, (255, 0, 0, 255), 0)
                    tangent_x.append(i)
                    tangent_y.append(int(j))
                    if max < j:
                        max = j
                p1,  = self.ax2.plot(tangent_x, tangent_y, color="blue")
            # 右側面
            elif slope_tangent > 0.0:
                for j in range(int(y)-30, int(y)+30):
                    i = 1/slope_tangent * (j - y) + x
                    cv2.circle(self.tangent_normal_img, (int(i), j), 1, (0, 255, 0, 255), 0)
                    tangent_x.append(int(i))
                    tangent_y.append(j)
                p2,  = self.ax2.plot(tangent_x, tangent_y, color="green")
            # 左側面
            else:
                for j in range(int(y)-30, int(y)+30):
                    i = 1/slope_tangent * (j - y) + x
                    cv2.circle(self.tangent_normal_img, (int(i), j), 1, (0, 0, 255, 255), 0)
                    tangent_x.append(int(i))
                    tangent_y.append(j)
                p3,  = self.ax2.plot(tangent_x, tangent_y, color="red")
        cv2.imwrite('4_tangent.png', self.tangent_normal_img)
        # グラフ作成
        self.ax2.set_title('高次多項式の接線', fontproperties=self.font_prop)
        self.min = round(min(tangent_y), -1)-10
        self.ax2.set_ylim(round(max, -1)+10, self.min)
        self.ax2.legend(handles=[p2, p1, p3], labels=['右側面', '顎', '左側面'], loc='center', prop=self.font_prop)
        self.figure2.savefig('4_tangent_graph.png')
    
    # 接線と垂直にプロットする関数
    def plot_tangent_normal(self):
        max = 0
        self.plot_tangent()
        for x in self.contour_x_list:
            # 接線の傾き
            slope_tangent = self.derivative(x)
            # 接線に垂直な傾き
            slope_normal = -1 / slope_tangent
            # y座標
            y = self.contour_polynomial(x)

            normal_x = []
            normal_y = []
            # 顎部分
            if np.abs(slope_normal) >= 1.0:
                for j in range(int(y)-self.scope, int(y)+self.scope):
                    i = 1/slope_normal * (j - y) + x
                    cv2.circle(self.tangent_normal_img, (int(i), j), 1, (255, 255, 0, 255), 0)
                    normal_x.append(int(i))
                    normal_y.append(j)
                    if max < j:
                        max = j
                p1,  = self.ax2.plot(normal_x, normal_y, color="#0ff")
            # 右側面
            elif slope_normal < 0.0:
                for i in range(x-self.scope, x+self.scope):
                    j = slope_normal * (i - x) + y
                    cv2.circle(self.tangent_normal_img, (i, int(j)), 1, (0, 255, 255, 255), 0)
                    normal_x.append(i)
                    normal_y.append(int(j))
                p2,  = self.ax2.plot(normal_x, normal_y, color="#ff0")
            # 左側面
            else:
                for i in range(x-self.scope, x+self.scope):
                    j = slope_normal * (i - x) + y
                    cv2.circle(self.tangent_normal_img, (i, int(j)), 1, (255, 0, 255, 255), 0)
                    normal_x.append(i)
                    normal_y.append(int(j))
                p3,  = self.ax2.plot(normal_x, normal_y, color="#f0f")
        cv2.imwrite('5_tangent_normal.png', self.tangent_normal_img)
        # グラフ作成
        self.ax2.set_title('接線に垂直な線', fontproperties=self.font_prop)
        self.ax2.set_ylim(round(max, -1)+10, self.min)
        self.ax2.legend(handles=[p2, p1, p3], labels=['右側面', '顎', '左側面'], loc='center', prop=self.font_prop)
        self.figure2.savefig('5_tangent_normal_graph.png')

if __name__ == '__main__':
    contour_alpha = ContourAlpha('test_programs/nomask.jpg', True, True, False)
    contour_alpha.plot_contour_cross()
    contour_alpha.plot_polynomial()
    contour_alpha.plot_contour_cross_normal()
    contour_alpha.plot_tangent()
    contour_alpha.plot_tangent_normal()