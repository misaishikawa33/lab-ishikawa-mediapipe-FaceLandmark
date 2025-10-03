#オブジェクト
from pyglet.gl import *
from kantanengine.vector3d import Vector3D
from kantanengine.material import Material
import math
from OpenGL.GL import *
from OpenGL.GLU import *

class Mesh3D():

	def __init__(self,name):
		self.name = name
		self.vertices = []
		self.faces = []
		self.set_facet(59.5)

	def set_facet(self,facet):
		self.facet = math.radians(facet)

	def calc_face_normals(self):
		length = len(self.faces)
		for i in range(length):
			face = self.faces[i]
			if face.exist == False:
				continue
			indices = face.indices
			if len(indices) >= 3:
				v = self.vertices[indices[0]]
				v1 = self.vertices[indices[1]]
				v2 = self.vertices[indices[2]]
				face.normal = self.get_normal(v,v1,v2)
				if face.normal == None:
					face.exist = False
					face.normal = Vector3D(0,0,-1)

	def get_normal(self,v,v1,v2):
		x = v1.x - v.x
		y = v1.y - v.y
		z = v1.z - v.z
		A = Vector3D(x,y,z)
		x = v2.x - v.x
		y = v2.y - v.y
		z = v2.z - v.z
		B = Vector3D(x,y,z)
		n = B.cross_product(A)
		if n.normalize() == None:
			return None
		return -n

	def calc_normals(self):
		for i in range(len(self.faces)):
			face = self.faces[i]
			if face.exist == False:
				continue
			indices = face.indices
			for j in range(len(indices)):
				index = indices[j]
				face.normals[j] += face.normal
				for k in range(len(self.faces)):
					if i == k:
						continue
					face2 = self.faces[k]
					if face2.exist == False:
						continue
					indices2 = face2.indices
					for l in range(len(indices2)):
						index2 = indices2[l]
						if index == index2:
							d = face.normal.dot_product(face2.normal)
							if d < -1 or d > 1:
								continue
							angle = math.acos(d)
							if angle < self.facet:
								face.normals[j] += face2.normal
							break
				face.normals[j].normalize()

	def draw(self,materials):
		for m in range(len(materials)):
			materials[m].set_material()
			for i in range(len(self.faces)):
				face = self.faces[i]
				if face.material != m or face.exist == False:
					continue
				indices = face.indices
				if len(indices) == 3:
					glBegin(GL_TRIANGLES)
					for k in range(3):
						face.uvs[k].setUV()
						face.normals[k].set_normal()
						self.vertices[indices[k]].set_vertex()
					glEnd()
		glFlush()
