from pyglet.gl import *
import math
from OpenGL.GL import *
from OpenGL.GLU import *

class Vector3D():

	def __init__(self,x,y,z,exist=True):
		self.x = x
		self.y = y
		self.z = z
		self.exist = exist

	def length(self):
		return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

	def normalize(self):
		l = self.length()
		if l == 0:
			return None
		else:
			self.x /= l
			self.y /= l
			self.z /= l
		return True

	def dot_product(self,v):
 		return (v.x*self.x + v.y*self.y + v.z*self.z)

	def cross_product(self,v):
		x = (self.y*v.z) - (v.y*self.z)
		y = (v.x*self.z) - (self.x*v.z)
		z = (self.x*v.y) - (v.x*self.y)
		return Vector3D(x, y, z)

	def __add__(self, other):
		x = self.x + other.x
		y = self.y + other.y
		z = self.z + other.z
		return Vector3D(x,y,z)

	def __sub__(self, other):
		x = self.x - other.x
		y = self.y - other.y
		z = self.z - other.z
		return Vector3D(x,y,z)

	def __mul__(self, other):
		x = self.x * other
		y = self.y * other
		z = self.z * other
		return Vector3D(x,y,z)

	def __neg__(self):
		return Vector3D(-self.x,-self.y,-self.z)

	def copy(self):
		return Vector3D(self.x,self.y,self.z)

	def check(self):
		if self.x < 0:
			self.x += 360
		if self.x >= 360:
			self.x -= 360
		if self.y < 0:
			self.y += 360
		if self.y >= 360:
			self.y -= 360
		if self.z < 0:
			self.z += 360
		if self.z >= 360:
			self.z -= 360

	def set_vertex(self):
		glVertex3f(self.x,self.y,self.z)

	def set_normal(self):
		glNormal3f(self.x,self.y,self.z)
