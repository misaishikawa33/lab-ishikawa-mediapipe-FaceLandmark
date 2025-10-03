from mqoloader.vector3d import Vector3D
from mqoloader.face3d import Face3D
import math
from OpenGL.GL import *

class Model3D():
    def __init__(self):
        self.meshes = []
        self.materials = []
        self.current = 0
    
    def draw(self):
        for mesh in self.meshes:
            mesh.draw(self.materials)

    def calc_normals(self,flat=True):
        for mesh in self.meshes:
            mesh.calc_face_normals()
            mesh.calc_normals()
