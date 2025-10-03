from pyglet.gl import *
from kantanengine.vector3d import Vector3D
from kantanengine.face3d import Face3D
import math
from OpenGL.GL import *
from OpenGL.GLU import *

class Model3D():
	def __init__(self):
		self.meshes = []
		self.materials = []
		self.current = 0

	def draw(self):
#		glEnable(GL_LIGHTING)
		for mesh in self.meshes:
			mesh.draw(self.materials)

	def calc_normals(self,flat=True):
		for mesh in self.meshes:
			mesh.calc_face_normals()
			mesh.calc_normals()

	def create_triangle(self,current,material):
		mesh = self.meshes[current]
		mesh.set_facet(89.5)
		one = len(mesh.vertices)
		two = one + 1
		three = two + 1
		mesh.vertices.append(Vector3D(0,100,0))
		mesh.vertices.append(Vector3D(100,-100,0))
		mesh.vertices.append(Vector3D(-100,-100,0))
		mesh.faces.append(
			Face3D([three,two,one],material,[0,1,1,1,0.5,0]))
