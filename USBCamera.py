import cv2
import datetime

class USBCamera:

    INPUT_CAMERA = 0
    INPUT_VIDEO  = 1

    #
    # コンストラクタ
    #
    # @param deviceID : カメラ番号
    #
    def __init__(self, width, height, use_api):
        self.inputMode     = self.INPUT_CAMERA
        self.use_api       = use_api
        self.deviceID      = 0
        self.vflip         = False
        self.hflip         = True
        self.save_original = True

        self.Open(width, height, None, use_api)
        
        # 出力ファイル用のヘッダー生成
        date = datetime.datetime.now()
        self.output_header = "%04d-%02d-%02d-%02d:%02d:%02d" % (date.year, date.month, date.day, date.hour, date.minute, date.second)
        # 出力ファイル用の連番
        self.image_count = 0

        # ビデオ出力フラグ(True: ビデオ出力中, False: ビデオ出力停止中)
        self.video_out = False

    #
    # デストラクタ:オブジェクトが削除される際に呼び出される
    def __del__(self):
        self.Close()

    #
    # カメラをクローズする関数
    #
    def Close(self):
        if self.capture.isOpened() is True:
            self.capture.release()

    #
    # カメラまたはビデオをオープンする関数
    #
    # @param width  : 画像の横サイズ
    # @param height : 画像の縦サイズ
    # @param name   : 動画ファイル名
    #
    def Open(self, width, height, name, use_api):
        if self.inputMode == self.INPUT_CAMERA:
            return self.OpenCamera(width, height, use_api)
        else:
            print("video")
            return self.OpenVideo (name, use_api)

    #
    # カメラをオープンする関数
    #
    # @param width  : 画像の横サイズ
    # @param height : 画像の縦サイズ
    #
    def OpenCamera(self, width, height, use_api):
        self.inputMode = self.INPUT_CAMERA
        # 画像の読み込み
        self.capture = cv2.VideoCapture(self.deviceID, use_api)

        if not self.capture:
            print('Camera open error')
            return False

        if self.capture.isOpened() is False:
            message = "Camera ID %d is not found." % self.deviceID
            print(message)
            return False

        self.image = self.capture.read()
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width    = width
        self.height   = height
        self.nchannels = 3

        return True

    #
    # ビデオをオープンする関数
    #
    # @param name : 動画ファイル名
    #
    def OpenVideo(self, name, use_api):
        self.inputMode = self.INPUT_VIDEO
        self.capture = cv2.VideoCapture(name, use_api)
        if self.capture.isOpened() is False:
            message = "Video %s is not found." % name
            print(message)
            return False

        self.width  = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        return True

    #
    # カメラまたはビデオから画像を取得する関数
    #
    def CaptureImage(self):
        ret, self.image = self.capture.read()
        if not ret:
            print("カメラの読み取りに失敗しました。")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        #
        # カメラ画像の回転
        #
        if self.vflip is True and self.hflip is True:
            self.image = cv2.flip(self.image, -1)
        elif self.vflip is True:
            self.image = cv2.flip(self.image, 0)
        elif self.hflip is True:
            self.image = cv2.flip(self.image, 1)
        return ret, self.image

    #
    # 画像を出力する関数
    #
    def SaveImage(self, image, filename):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image)
        
    #
    # 録画を開始する関数
    #
    def SaveRecord(self, filename):
        fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(filename, fourcc, fps, (self.width, self.height))
        return video

    #
    # 画像のフリップ設定を行う関数
    # 
    def SetFlip (self, hflip, vflip):
        self.hflip = hflip
        self.vflip = vflip
