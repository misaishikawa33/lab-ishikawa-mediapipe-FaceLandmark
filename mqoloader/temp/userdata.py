from pyglet.gl import *
from kantanengine.material import Material
from kantanengine.triangle import Triangle
from kantanengine.vector3d import Vector3D
from kantanengine.vector2d import Vector2D

def vec(*args):
	return (GLfloat * len(args))(*args)

class UserData():

	def __init__(self,main_dir):
		self.model = Triangle(main_dir)
		self.trans = Vector3D(0,0,-300)
		self.rotate = Vector3D(0,0,0)
		self.filename = None
		self.size = Vector2D(1000.0,700.0)
		self.projection = vec(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)
		self.camera = vec(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)
		self.rgb = []
		self.rgb.append(Material("Red",[1,0,0,1],1,0,0,0,1))
		self.rgb.append(Material("Green",[0,1,0,1],1,0,0,0,1))
		self.rgb.append(Material("Blue",[0,0,1,1],1,0,0,0,1))
