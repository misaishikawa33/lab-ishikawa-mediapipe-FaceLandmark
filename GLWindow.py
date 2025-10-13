import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *


class GLWindow:
    #
    # コンストラクタ
    #
    def __init__(self, title, width, height, display_func, keyboard_func):
        # GLFWの初期化
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW3")
        
        # ウィンドウ生成
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create a window")

        # ウィンドウ更新関数の設定
        if display_func != None:
            glfw.set_window_refresh_callback(self.window, display_func)
        else:
            glfw.set_window_refresh_callback(self.window, self.display_func)

        glfw.set_window_size_callback(self.window, self.reshape_func)
            
        # キーボード関数
        if keyboard_func != None:
            glfw.set_key_callback(self.window, keyboard_func)
        else:
            glfw.set_key_callback(self.window, self.keyboard_func)
            
        # ウィンドウ表示
        glfw.make_context_current(self.window)

        # OpenGLの設定(20251013)
        # アルファブレンディングを有効化（透明度を有効にする）
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # デプステストを有効化
        glEnable(GL_DEPTH_TEST)

        # メンバ変数の初期化
        self.image  = np.array([])
        self.width  = width
        self.height = height

        
    #
    # ウィンドウを閉じるかどうかを返す関数
    #
    def window_should_close(self):
        return glfw.window_should_close(self.window)
    

    #
    # タイムアウト関数
    #
    def wait_events_timeout(self):
        glfw.wait_events_timeout(1e-03)

    #
    # ディスプレイ関数
    #
    def display_func(self, window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.image.size != 0:
            self.draw_image(self.image)
        
        glfw.swap_buffers(window)

    def reshape_func(self, window, width, height):
        glViewport(0, 0, width, height)
    
    #
    # 画像の描画関数
    #
    def draw_image(self, image):
        glDisable(GL_DEPTH_TEST)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()

        glBindTexture(GL_TEXTURE_2D, 0)        
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, image)

        glColor3f(1.0, 1.0, 1.0)        
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glBegin(GL_QUADS)
        
        glTexCoord2d(0.0, 1.0)
        glVertex3d(-1.0, -1.0, 0.0)

        glTexCoord2d(1.0, 1.0)
        glVertex3d(1.0, -1.0, 0.0)

        glTexCoord2d(1.0, 0.0)
        glVertex3d(1.0, 1.0, 0.0)

        glTexCoord2d(0.0, 0.0)
        glVertex3d(-1.0, 1.0, 0.0)

        glEnd()
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        glEnable(GL_DEPTH_TEST)

    #
    # 世界座標系の軸を描画する関数
    #
    def draw_xyz_axis(self):

        glLineWidth (5.0)
        
        glBegin(GL_LINES)
        
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glVertex3d(5000.0, 0.0, 0.0)
        glVertex3d(0.0, 0.0, 0.0)

        glColor4f(0.0, 1.0, 0.0, 1.0)
        glVertex3d(0.0, 5000.0, 0.0)
        glVertex3d(0.0, 0.0, 0.0)

        glColor4f(0.0, 0.0, 1.0, 1.0)
        glVertex3d(0.0, 0.0, 5000.0)
        glVertex3d(0.0, 0.0, 0.0)
        
        glEnd()

        glColor4f(1.0, 1.0, 1.0, 1.0)        


    #
    # xz平面のグリッドを記述
    # 
    def draw_XZ_plane(self, mesh_size, mesh_skip):
        glLineWidth (5.0)
        
        # 線分の描画
        glBegin(GL_LINES)
        
        glColor4f(1.0, 1.0, 1.0, 0.5)

        x = -mesh_size
        while x < mesh_size:
            # 頂点を指定
            glVertex3d(x, 0.0, -mesh_size)
            glVertex3d(x, 0.0, mesh_size)
            x += mesh_skip

        z = -mesh_size
        while z < mesh_size:
            glVertex3d(-mesh_size, 0.0, z)
            glVertex3d(mesh_size, 0.0, z)
            z += mesh_skip
        
        glEnd()

        glColor4f(1.0, 1.0, 1.0, 1.0)
        
    #
    # 直線を記述
    #       
    def draw_line(self, vector):
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glLineWidth(5.0)
        
        # 線分の描画
        glBegin(GL_LINES)
        
        glVertex2d(0, 0)
        glVertex2d(vector[0], -vector[1])
        
        glEnd()
        
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glPointSize(10.0)
        
        # 始点の描画
        glBegin(GL_POINTS)
        glVertex2i(0, 0)
        glEnd()
        glColor4f(1.0, 1.0, 1.0, 1.0)
    
    #
    # 頂点を記述
    #
    def draw_point(self, vector):
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glPointSize(10.0)
        
        # 始点の描画
        glBegin(GL_POINTS)
        glVertex2i(vector[0], -vector[1])
        glEnd()
        glColor4f(1.0, 1.0, 1.0, 1.0)
    
    #
    # キー関数
    # 
    def keyboard_func(self, window, key, scancode, action, mods):
        if key == glfw.KEY_Q:
            glfw.set_window_should_close(self.window, GL_TRUE)

    #
    # 表示用画像を設定する関数
    # 
    def set_image (self, image):
        self.image = image
