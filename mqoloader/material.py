import cv2
import numpy as np
from OpenGL.GL import *

def vec(*args):
    return (GLfloat * len(args))(*args)

class Material():

    def __init__(self,name,col,dif,amb,emi,spc,power,textureID,tex=None):
        self.name = name
        self.col = col
        self.dif = dif
        self.amb = amb
        self.emi = emi
        self.spc = spc
        self.diffuse = vec(col[0] * dif, col[1] * dif, col[2] * dif, col[3])
        self.ambient = vec(0.25 * amb, 0.25 * amb, 0.25 * amb, 1)
        self.emission = vec(emi, emi, emi, 1)
        self.spcular = vec(spc, spc, spc, 1)
        self.power = power
        self.tex = tex
        self.source_texture_img = None
        self.texture_format = None
        self.texture_internal_format = None
        if tex != None:
            self.load_texture(tex, textureID)
    
    def set_material(self):
        # glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   self.diffuse)
        # glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   self.ambient)
        # glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION,  self.emission)
        # glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  self.spcular)
        # glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, self.power)
        # glColor3f(self.col[0],self.col[1],self.col[2])

        if self.tex == None:
            glDisable(GL_TEXTURE_2D)
        else:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D,self.textureID)

    def load_texture(self,filename, textureID):
        # アルファチャンネルを含めて画像を読み込む
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img = cv2.flip(img, 0)
        
        # チャンネル数を確認してRGBまたはRGBAに変換(20251013)
        if img.shape[2] == 4:
            # RGBA画像の場合
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            internal_format = GL_RGBA8
            format_type = GL_RGBA
        else:
            # RGB画像の場合
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            internal_format = GL_RGB8
            format_type = GL_RGB
        
        self.source_texture_img = img.copy()
        self.texture_internal_format = internal_format
        self.texture_format = format_type
        self.textureID = textureID
        glBindTexture(GL_TEXTURE_2D,self.textureID)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR)
        glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_MODULATE)
        height, width = img.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height,
                     0, format_type, GL_UNSIGNED_BYTE, img)

    @staticmethod
    def _rgb_to_scalar_luminance(rgb, mode):
        rgb = np.asarray(rgb, dtype=np.uint8).reshape(1, 1, 3)
        if mode == 'lab_luminance':
            return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0, 0])
        return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[0, 0, 0])

    @staticmethod
    def adjust_texture_color(source_texture_img, source_rgb, target_rgb, mode='rgb'):
        source_rgb = source_rgb.astype('float32')
        target_rgb = target_rgb.astype('float32')
        adjusted = source_texture_img.astype('float32')

        if mode == 'rgb':
            rgb = adjusted[:, :, :3]
            rgb[:] = (rgb - source_rgb) + target_rgb
            adjusted[:, :, :3] = np.clip(rgb, 0, 255)
            return adjusted.astype('uint8')

        if mode not in ('ycrcb_luminance', 'lab_luminance'):
            raise ValueError(f"Unsupported color adjustment mode: {mode}")

        rgb_uint8 = source_texture_img[:, :, :3].copy()
        if mode == 'lab_luminance':
            converted = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype('float32')
        else:
            converted = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2YCrCb).astype('float32')

        source_luma = Material._rgb_to_scalar_luminance(source_rgb, mode)
        target_luma = Material._rgb_to_scalar_luminance(target_rgb, mode)
        converted[:, :, 0] = np.clip(converted[:, :, 0] + (target_luma - source_luma), 0, 255)

        converted = converted.astype('uint8')
        if mode == 'lab_luminance':
            adjusted[:, :, :3] = cv2.cvtColor(converted, cv2.COLOR_LAB2RGB)
        else:
            adjusted[:, :, :3] = cv2.cvtColor(converted, cv2.COLOR_YCrCb2RGB)

        return adjusted.astype('uint8')

    def update_color_adjustment(self, source_rgb, target_rgb, mode='rgb'):
        if self.source_texture_img is None:
            return

        adjusted = self.adjust_texture_color(
            self.source_texture_img,
            source_rgb,
            target_rgb,
            mode)

        height, width = adjusted.shape[:2]
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        glTexImage2D(GL_TEXTURE_2D, 0, self.texture_internal_format, width, height,
                     0, self.texture_format, GL_UNSIGNED_BYTE, adjusted)
