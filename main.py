# main.py
# editor : tagawa kota, sugano yasuyuki
# last updated : 2023/6/9
# overview : 
# Input and set 3dpoint, 3dmodel and imagefile. 
# Run main loop of app.

import sys
import cv2
import glfw
from mqoloader.loadmqo import LoadMQO
import Application

# 追加
from create_MQO import CreateMQO
import os
#import PySimpleGUI as sg
import TkEasyGUI as sg
import argparse
import time

#
# メインクラス
#        
class Main:
    
    #
    # コンストラクタ
    # (@param kwargs : image = "image_filename"
    #                  texture = "texture_filename")
    #
    def __init__(self, texture, draw_landmark, use_facelandmark=False, use_alpha=False):
        
        if texture is not None:
            self.take_texture = False
            texture_filename = texture
        else:
            self.take_texture = True
            texture_filename = "default.png"
        
        # アルファ処理フラグを保持
        self.use_alpha = use_alpha
        
        # ディスプレイサイズ
        width  = 640
        height = 480
            
        #use_api = cv2.CAP_DSHOW # Windowsで使用する場合こちらを使う
        use_api = 0 # Linuxで使用する場合はこちらを使う
        
        #
        # アプリケーション設定
        # Applicationクラスのインスタンス生成
        #
        title = 'Face marker AR (press w...change model, press q...quit)'
        self.app = Application.Application(
            title, 
            width, height, use_api,
            draw_landmark, use_facelandmark)
        
        #
        # テクスチャ撮影
        # 初期テクスチャを使用する場合、回転の補正は使用しない(モデルの向きがおかしくなるため)
        # 
        if self.take_texture == True:
            # GUI設定
            
            sg.theme("clam")

            #sg.theme("DefaultNoMoreNagging")
            layout = [[sg.Text("テクスチャ画像を撮影しますか？")]
                    ,[sg.Button('Yes'), sg.Button('No')]]
            self.window = sg.Window('main.py', layout)
            
            # ウィンドウ読み込み
            event, values = self.window.read()
            
            # イベント処理
            if event in (None, 'Cancel'):
                # 右上のボタンでウィンドウを閉じた場合、プログラムを終了
                sys.exit("強制終了")
            elif event == "Yes":
                while True:
                    # カメラ画像読み込み
                    success, image = self.app.camera.CaptureImage()
                    if not success:
                        print("error : video open error")
                        return
                    # 画像を表示
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.imshow("take a texture image", image)
                    # `s`キーを押すと画像を撮影しループ終了
                    if cv2.waitKey(1) == ord('s'):
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        self.app.camera.SaveImage(image, 'mqodata/nomask.jpg')
                        self.app.camera.SaveImage(image, 'mqodata/model/nomask.jpg')
                        texture_filename = "nomask.jpg"
                        break
                    # `q`キーを押すと画像を撮影せずにループ終了
                    if cv2.waitKey(1) == ord('q'):
                        break
                
                # カメラウィンドウを閉じる
                cv2.destroyAllWindows()
                
                # 画像撮影後、アルファ処理を行うか確認（引数で指定されていない場合のみ）
                if not use_alpha:  # コマンドライン引数でuse_alphaが指定されていない場合
                    self.window.close()
                    layout_alpha = [[sg.Text("テクスチャにアルファ処理を行いますか？")]
                                   ,[sg.Button('Yes'), sg.Button('No')]]
                    window_alpha = sg.Window('アルファ処理確認', layout_alpha)
                    event_alpha, values_alpha = window_alpha.read()
                    
                    if event_alpha in (None, 'Cancel', 'No'):
                        self.use_alpha = False
                    elif event_alpha == "Yes":
                        self.use_alpha = True
                    
                    window_alpha.close()
                else:
                    # コマンドライン引数で指定されている場合はそのまま使用
                    self.use_alpha = use_alpha
                    
            elif event == "No":
                # 画像を撮影せずに、デフォルトのテクスチャ画像を使用
                texture_filename = "nomask.jpg"
                pass
            # カメラを閉じる
            self.app.camera.Close()
            # ウィンドウクローズ
            self.window.close()
        
        #
        # モデル読み込み
        # CreateMQOクラスのインスタンス生成（アルファ処理フラグを渡す）
        #
        print("ok")
        start = time.time()
        mqo = CreateMQO(texture_filename, use_alpha=self.use_alpha)
        model_filename = os.getcwd() +"/"+ mqo.model_filename
        end = time.time()
        msg = 'Creating %s' % model_filename
        print(msg + " " +str(end-start)+'sec.')

        # 3次元データをアプリケーションにセット
        self.app.set_3D_point(mqo.data, mqo.datalist)
        self.app.set_3D_point_1(mqo.data1, mqo.datalist1) # 一部の特徴点を使用
        self.app.set_3D_point_2(mqo.data2, mqo.datalist2) # 一部の特徴点を使用
       
        # モデルを生成
        self.display(model_filename)
        
        if self.take_texture:
            # アプリケーションのカメラオープン
            self.app.camera.Open(width, height, None, use_api)
        else:
            pass
   
        #
        # アプリケーションのメインループ
        #
        while not self.app.glwindow.window_should_close():
            # カメラ映像の表示(メインの処理が記述される)
            self.app.display_func(self.app.glwindow.window) 
            # イベントを待つ
            glfw.poll_events()
        
        # glfwの終了処理
        glfw.terminate()
        # Applicationクラスのインスタンス削除
        del self.app
    
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
        
        start = time.time()
        #
        # 第3引数をTrueにすると面の法線計算を行い、陰影がリアルに描画されます（実行できない）
        # その代わりに計算にかなり時間がかかります
        #
        self.app.use_normal = False
        model_scale = 10.0
        model = LoadMQO(model_filename, model_scale, self.app.use_normal)
        end = time.time()
        print('Done. '+str(end-start)+'sec.')
        self.app.set_mqo_model(model)
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage =  "python main.py <-t texture_filename> <-i image_filename>",
        description = "description for commandline arguments",
        epilog = "end",
        add_help = True,
    )
    parser.add_argument("--texture", default=None, help = "texture_filename")
    parser.add_argument('--draw_landmark', action='store_true', help = "draw landmark")
    parser.add_argument('--use_facelandmark', action='store_true', help = "enable FaceLandmark processing")
    parser.add_argument('--use_alpha', action='store_true', help = "enable alpha channel processing for texture")
    args = parser.parse_args()
    
    Main(args.texture, args.draw_landmark, args.use_facelandmark, args.use_alpha)