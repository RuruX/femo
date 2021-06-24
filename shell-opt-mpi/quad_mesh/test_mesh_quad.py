
from dolfinx import *
from dolfinx.io import *
from mpi4py import MPI
import ufl
from ufl import *

encoding = dolfinx.cpp.io.XDMFFile.Encoding.ASCII
# Import manifold mesh of topological dimension 2 and geometric dimension 3:
with XDMFFile(MPI.COMM_WORLD,'mesh.xdmf','r',encoding=encoding) as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

#with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "meshquad_attempt.xdmf", "r") as xdmf:
#       mesh = xdmf.read_mesh(name="Grid")

# Write it out in a visualizable format for Paraview:
#File("meshout.pvd") << mesh
dolfinx.io.VTKFile("meshout_quad.pvd").write(mesh)

# Make sure the necessary finite element spaces are supported for this mesh:
cell = mesh.ufl_cell()
VE = VectorElement("Lagrange",cell,1)
WE = MixedElement([VE,VE])
W = FunctionSpace(mesh,WE)
x = SpatialCoordinate(mesh)
#w = project(as_vector(6*[Constant(1),]),W)

with XDMFFile(MPI.COMM_WORLD, "meshout_quad.xdmf", "w") as file:
    file.write_mesh(mesh)

