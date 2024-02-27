
#############################################
#Code snippet to compute the shape derivatives
#############################################
from femo.fea.fea_dolfinx import *
mesh_2D = createUnitSquareMesh(2)
X = SpatialCoordinate(mesh_2D)

VX = VectorFunctionSpace(mesh_2D,("CG",1))

VT = FunctionSpace(mesh_2D,("CG",1))
T = Function(VT)
T.x.array[:] = 1.
vol = T*dx
args = vol.arguments()
# UFL arguments need unique indices within a form
n = max(a.number() for a in args) if args else -1
du = ufl.Argument(VX, n+1)
dLdX = ufl.derivative(vol, X, du)
print("Derivatives of volume w.r.t. spatial coordinates")
print(assembleVector(dLdX).reshape(-1,2))
fdim = 0
num_facets_owned_by_proc = mesh_2D.topology.index_map(fdim).size_local
geometry_entities = dolfinx.cpp.mesh.entities_to_geometry(mesh_2D, fdim, np.arange(num_facets_owned_by_proc, dtype=np.int32), False)
points = mesh_2D.geometry.x
print('Node id, Coords')
for e, entity in enumerate(geometry_entities):
    print(e, points[entity])
#############################################