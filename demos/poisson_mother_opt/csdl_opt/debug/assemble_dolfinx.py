import dolfinx
import ufl
import numpy as np
from mpi4py import MPI

n = 2
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
V = dolfinx.fem.FunctionSpace(mesh, ('CG', 1))
VF = dolfinx.fem.FunctionSpace(mesh, ('DG', 0))

u = dolfinx.fem.Function(V)
v = ufl.TestFunction(V)

# Define the source term

class Expression_f:
    def __init__(self):
        self.alpha = 1e-6

    def eval(self, x):
        return (1/(1+self.alpha*4*np.power(np.pi,4))*
                np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
                        
f_analytic = Expression_f()
f = dolfinx.fem.Function(VF)
f.interpolate(f_analytic.eval)

# Apply zero boundary condition on the outer boundary
tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)
boundary_facets = np.flatnonzero(
                    dolfinx.mesh.compute_boundary_facets(
                        mesh.topology))

boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
ubc = dolfinx.fem.Function(V)
ubc.vector.set(0.0)
bc = [dolfinx.fem.dirichletbc(ubc, boundary_dofs)]

# Variational form of Poisson's equation
res = (ufl.inner(ufl.grad(u),ufl.grad(v))-f*v)*ufl.dx

# Quantity of interest: will be used as the Jacobian in adjoint method
dRdu = ufl.derivative(res, u)

# Option 1: assemble A and b seperately
A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(dRdu), bcs=bc)


def convertToDense(A_petsc):
    """
    Convert the PETSc matrix to a dense numpy array
    (super unefficient, only used for debugging purposes)
    """
    A_petsc.assemble()
    A_dense = A_petsc.convert("dense")
    return A_dense.getDenseArray()

print(" ------ Matrix A by DOLFINx ------- ")
print(convertToDense(A))




