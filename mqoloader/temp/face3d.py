from kantanengine.uv import UV
from kantanengine.vector3d import Vector3D

class Face3D():

	def __init__(self,indices,material,uvs):
		self.indices = indices
		self.material = material
		self.normal = None
		self.normals = []
		for i in range(len(indices)):
			self.normals.append(Vector3D(0.0,0.0,0.0))
		self.uvs = []
		for i in range(0,len(uvs),2):
			self.uvs.append(UV(uvs[i],uvs[i+1]))
		self.exist = True
