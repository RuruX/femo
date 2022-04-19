"""
The FEniCS wrapper for the variational forms and the partial derivatives computation
"""

from fea_utils_dolfinx import *
from dolfinx.io import XDMFFile
import ufl

from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, apply_lifting)
from dolfinx.fem import (set_bc, Function, FunctionSpace, form, dirichletbc,   
                        assemble_scalar, locate_dofs_topological)
from dolfinx.mesh import compute_boundary_facets
from ufl import (TestFunction, TrialFunction, dx, inner, derivative,
                    grad, SpatialCoordinate)
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

PI = np.pi

def pdeRes(u,v,f):
    """
    The variational form of the PDE residual for the Poisson's problem
    """
    return (inner(grad(u),grad(v))-f*v)*dx


class FEA(object):
    """
    The class of the FEniCS wrapper for the motor problem,
    with methods to compute the variational forms, partial derivatives,
    and solve the nonlinear/linear subproblems.
    """
    def __init__(self, mesh, coords_bc=[]):

        self.mesh = mesh
        # # Import the initial mesh from the mesh file to FEniCS
        # self.initMesh()
        # Define the function spaces based on the initial mesh
        self.initFunctionSpace()

        # Get the indices of the vertices that would move during optimization
        # self.bc_ind = self.locateBC(coords_bc)

        self.u = Function(self.V) # Function for the solution of the magnetic vector potential
        self.v = TestFunction(self.V)
        self.dR = Function(self.V) # Function used in the CSDL model
        self.du = Function(self.V) # Function used in the CSDL model
        
        self.f = Function(self.VF)
        self.df = Function(self.VF)

        # self.total_dofs_bc = len(self.bc_ind)
        self.total_dofs_u = len(self.u.vector.getArray())
        self.total_dofs_f = len(self.f.vector.getArray())
        # Partial derivatives in the magnetostatic problem
        self.dR_du = derivative(self.R(), self.u)
        self.dR_df = derivative(self.R(), self.f)

        self.u_ex, self.f_ex = self.exactSolution()

    def initFunctionSpace(self):
        """
        Preprocessor 2 to define the function spaces for the mesh motion (VHAT)
        and the problem solution (V)
        """
        self.V = FunctionSpace(self.mesh, ('CG', 1))
        self.VF = FunctionSpace(self.mesh, ('DG', 0))

    # def locateBC(self,coords_bc):
    #     """
    #     Find the indices of the dofs for setting up the boundary condition
    #     in the mesh motion subproblem
    #     """
    #     V0 = FunctionSpace(self.mesh, 'CG', 1)
    #     coordinates = V0.tabulate_dof_coordinates()

    #     # Use KDTree to find the node indices of the points on the edge
    #     # in the mesh object in FEniCS
    #     node_indices = findNodeIndices(np.reshape(coords_bc, (-1,2)),
    #                                     coordinates)

    #     # Convert the node indices to edge indices, where each node has 2 dofs
    #     dofs = np.empty(2*len(node_indices))
    #     for i in range(len(node_indices)):
    #         dofs[2*i] = 2*node_indices[i]
    #         dofs[2*i+1] = 2*node_indices[i]+1

    #     return dofs.astype('int')

    def R(self):
        """
        Formulation of the magnetostatic problem
        """
        res = pdeRes(
                self.u,self.v,self.f)
        return res

    def bc(self):
        # Create facet to cell connectivity required to determine boundary facets
        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        self.mesh.topology.create_connectivity(fdim, tdim)
        boundary_facets = np.flatnonzero(
                            compute_boundary_facets(
                                self.mesh.topology))
        boundary_dofs = locate_dofs_topological(self.V, fdim, boundary_facets)
        ubc = Function(self.V)
        ubc.vector.set(0.0)
        bc = [dirichletbc(ubc, boundary_dofs)]
        return bc

    def exactSolution(self):
        """
        Exact solutions for the problem
        """
        x = SpatialCoordinate(self.mesh)
        class Expression_f:
            def __init__(self):
                pass
            def eval(self, x):
                # Added some spatial variation here. Expression is sin(t)*x
                return (1/(1+alpha*4*pow(PI,4))*
                        np.sin(PI*x[0])*np.sin(PI*x[1]))
        
        class Expression_u:
            def __init__(self):
                pass
            def eval(self, x):
                # Added some spatial variation here. Expression is sin(t)*x
                return (1/(2*pow(PI, 2))*
                        1/(1+alpha*4*pow(PI,4))*
                        np.sin(PI*x[0])*np.sin(PI*x[1]))

        alpha = 1e-6
        f_analytic = Expression_f()
        u_analytic = Expression_u()
        f_ex = Function(self.VF)
        u_ex = Function(self.V)
        f_ex.interpolate(f_analytic.eval)
        u_ex.interpolate(u_analytic.eval)
        return u_ex, f_ex

    def objective(self):
        x = SpatialCoordinate(self.mesh)
        w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
        d = 1/(2*pi**2)
        d = Expression("d*w", d=d, w=w, degree=3)
        alpha = 1e-6
        return 0.5*inner(self.u-d, self.u-d)*dx + alpha/2*self.f**2*dx

    def getBCDerivatives(self):
        """
        Compute the derivatives of the PDE residual of the mesh motion
        subproblem wrt the BCs, which is a fixed sparse matrix with "-1"s
        on the entries corresponding to the edge indices.
        """

        row_ind = self.bc_ind
        col_ind = np.arange(self.total_dofs_bc)
        data = -1.0*np.ones(self.total_dofs_bc)
        M = csr_matrix((data, (row_ind, col_ind)),
                        shape=(self.total_dofs_uhat, self.total_dofs_bc))
        return M


    def solve(self, report=False):
        """
        Solve the PDE problem
        """
        if report == True:
            print(80*"=")
            print(" FEA: Solving the PDE problem")
            print(80*"=")
        bc = self.bc()
        res = self.R()

        solveNonlinear(res, self.u, bc)


    def solveLinearFwd(self, A, dR):
        """
        solve linear system dR = dR_du (A) * du in DOLFIN type
        """
        setArray(self.dR, dR)

        self.du.vector.set(0.0)

        solveKSP(A, self.du.vector, self.dR.vector)
        self.du.vector.assemble()
        self.du.vector.ghostUpdate()
        return self.du.vector.getArray()

    def solveLinearBwd(self, A, du):
        """
        solve linear system du = dR_du.T (A_T) * dR in DOLFIN type
        """
        setArray(self.du, du)

        self.dR.vector.set(0.0)

        solveKSP(transpose(A), self.dR.vector, self.du.vector)
        self.dR.vector.assemble()
        self.dR.vector.ghostUpdate()
        return self.dR.vector.getArray()

if __name__ == "__main__":
    n = 16
    mesh = createUnitSquareMesh(n)
    fea = FEA(mesh)
    print(mesh)
    f_ex = fea.f_ex
    u_ex = fea.u_ex

    fea.f.x.array[:] = f_ex.x.array
    fea.solve(report=False)
    state_error = errorNorm(u_ex, fea.u)
    print("Error in solve():", state_error)
    # Visualization with Paraview
    with XDMFFile(MPI.COMM_WORLD, "solutions/u.xdmf", "w") as xdmf:
        xdmf.write_mesh(fea.mesh)
        xdmf.write_function(fea.u)

