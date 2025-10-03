from pyglet.gl import *
from OpenGL.GL import *
from OpenGL.GLU import *

class UV():

	def __init__(self,u,v):
		self.u = u
		self.v = -v

	def setUV(self):
		glTexCoord2f(self.u,self.v)
