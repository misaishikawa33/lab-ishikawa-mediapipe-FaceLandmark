import cv2
from pyglet.gl import *
from OpenGL.GL import *
from OpenGL.GLU import *

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
		self.diffuse = vec(col[0]*dif,col[1]*dif,col[2]*dif,col[3])
		self.ambient = vec(0.25*amb,0.25*amb,0.25*amb,1)
		self.emission = vec(emi,emi,emi,1)
		self.spcular = vec(spc,spc,spc,1)
		self.power = power
		self.tex = tex
		if tex != None:
			self.load_texture(tex, textureID)

	def set_material(self):
#		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   self.diffuse)
#		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   self.ambient)
#		glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION,  self.emission)
#		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  self.spcular)
#		glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, self.power)
#		glColor3f(self.col[0],self.col[1],self.col[2])
		if self.tex == None:
			glDisable(GL_TEXTURE_2D)
		else:
			glEnable(GL_TEXTURE_2D)
			glBindTexture(GL_TEXTURE_2D,self.textureID)
	
	def load_texture(self,filename, textureID):
		img = cv2.imread(filename)
		img = cv2.flip(img, 0)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.textureID = textureID
		glBindTexture(GL_TEXTURE_2D,self.textureID)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR)
		glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_MODULATE)
		height, width = img.shape[:2]
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height,
			     0, GL_RGB, GL_UNSIGNED_BYTE, img)
