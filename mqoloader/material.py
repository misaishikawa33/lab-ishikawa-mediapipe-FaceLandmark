import cv2
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
