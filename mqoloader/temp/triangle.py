from kantanengine.mesh3d import Mesh3D
from kantanengine.material import Material
from kantanengine.vector3d import Vector3D
from kantanengine.model3d import Model3D
from kantanengine.face3d import Face3D

class Triangle(Model3D):

	def __init__(self,main_dir):
		Model3D.__init__(self)

		self.materials.append(Material("White",[1,1,1,1],1,0,0,0,1))
		self.meshes.append(Mesh3D("Triangle"))
		self.create_triangle(0,0)
		self.calc_normals()
