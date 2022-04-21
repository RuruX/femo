"""
The FEniCS wrapper for the variational forms and the partial derivatives computation
in the magnetostatic problem for the motor on a deformable mesh.
"""
from fea_utils_old_dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

"""
Definition of the variational form of the motor problem
"""

def pdeRes(u,v,f):
    """
    The variational form of the PDE residual for the Poisson's problem
    """
    return (inner(grad(u),grad(v))-f*v)*dx

class OuterBoundary(SubDomain):
    """
    Define the subdomain for the outer boundary
    """
    def inside(self, x, on_boundary):
        return on_boundary

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
        self.bc_ind = self.locateBC(coords_bc)

        self.u = Function(self.V) # Function for the solution of the magnetic vector potential
        self.v = TestFunction(self.V)
        self.dR = Function(self.V) # Function used in the CSDL model
        self.du = Function(self.V) # Function used in the CSDL model
        
        self.f = Function(self.VF)
        self.df = Function(self.VF)

        self.local_dof_f = len(self.VF.dofmap().dofs())
        self.local_dof_u = len(self.V.dofmap().dofs())
        self.ind_f = getGlobalIndices(self.f)
        # Ghost points are not included in the indices of u
        self.ind_u = getGlobalIndices(self.u)[:self.local_dof_u]
        
        self.u_ex, self.f_ex = self.exactSolution()
        self.total_dofs_bc = len(self.bc_ind)
        self.total_dofs_u = len(self.u.vector().get_local())
        self.total_dofs_f = len(self.f.vector().get_local())
        # Partial derivatives in the magnetostatic problem
        self.dR_du = derivative(self.R(), self.u)
        self.dR_df = derivative(self.R(), self.f)
        self.dC_du = derivative(self.objective(), self.u)
        self.dC_df = derivative(self.objective(), self.f)

    def initFunctionSpace(self):
        """
        Preprocessor 2 to define the function spaces for the mesh motion (VHAT)
        and the problem solution (V)
        """
        self.V = FunctionSpace(self.mesh, 'CG', 1)
        self.VF = FunctionSpace(self.mesh, 'DG', 0)

    def locateBC(self,coords_bc):
        """
        Find the indices of the dofs for setting up the boundary condition
        in the mesh motion subproblem
        """
        V0 = FunctionSpace(self.mesh, 'CG', 1)
        coordinates = V0.tabulate_dof_coordinates()

        # Use KDTree to find the node indices of the points on the edge
        # in the mesh object in FEniCS
        node_indices = findNodeIndices(np.reshape(coords_bc, (-1,2)),
                                        coordinates)

        # Convert the node indices to edge indices, where each node has 2 dofs
        dofs = np.empty(2*len(node_indices))
        for i in range(len(node_indices)):
            dofs[2*i] = 2*node_indices[i]
            dofs[2*i+1] = 2*node_indices[i]+1

        return dofs.astype('int')


    def R(self):
        """
        Formulation of the magnetostatic problem
        """
        res = pdeRes(
                self.u,self.v,self.f)
        return res

    def bc(self):
        bc = DirichletBC(self.V, Constant(0.0), "on_boundary")
        return bc

    def exactSolution(self):
        """
        Exact solutions for the problem
        """
        x = SpatialCoordinate(self.mesh)
        w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
        alpha = Constant(1e-6)
        # f_analytic = Expression("sin(pi*x[0])", degree=3)
        f_analytic = Expression("1/(1+alpha*4*pow(pi,4))*w", w=w, alpha=alpha, degree=3)
        u_analytic = Expression("1/(2*pow(pi, 2))*f", f=f_analytic, degree=3)
        f_ex = interpolate(f_analytic, self.VF)
        u_ex = interpolate(u_analytic, self.V)
        return u_ex, f_ex

    def objective(self):
        x = SpatialCoordinate(self.mesh)
        w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
        d = 1/(2*pi**2)
        d = Expression("d*w", d=d, w=w, degree=3)
        alpha = Constant(1e-6)
        return (Constant(0.5)*inner(self.u-d, self.u-d))*dx + alpha/2*self.f**2*dx

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
        u_ = TrialFunction(self.V)
        Dres = derivative(res, self.u, u_)

        # Nonlinear solver parameters
        ABS_TOL_M = 1e-6
        REL_TOL_M = 1e-6
        MAX_ITERS_M = 100

        problem = NonlinearVariationalProblem(res, self.u,
                                                bc, Dres)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['nonlinear_solver']='snes'
        solver.parameters['snes_solver']['line_search'] = 'bt'
        solver.parameters['snes_solver']['absolute_tolerance'] = ABS_TOL_M
        solver.parameters['snes_solver']['relative_tolerance'] = REL_TOL_M
        solver.parameters['snes_solver']['maximum_iterations'] = MAX_ITERS_M
        solver.parameters['snes_solver']['linear_solver']='mumps'
        solver.parameters['snes_solver']['error_on_nonconvergence'] = False
        solver.parameters['snes_solver']['report'] = report
        solver.solve()


    def solveLinearFwd(self, A, dR):
        """
        solve linear system dR = dR_du (A) * du in DOLFIN type
        """
        self.dR.vector().set_local(dR)
        v2p(self.dR.vector()).assemble()
        v2p(self.dR.vector()).ghostUpdate()

        self.du.vector().set_local(np.zeros(self.local_dof_u))
        v2p(self.du.vector()).assemble()
        v2p(self.du.vector()).ghostUpdate()

        solverFwd = LUSolver("mumps")
        solverFwd.solve(A, self.du.vector(), self.dR.vector())
        v2p(self.du.vector()).assemble()
        v2p(self.du.vector()).ghostUpdate()
        return self.du.vector().get_local()

    def solveLinearBwd(self, A, du):
        """
        solve linear system du = dR_du.T (A_T) * dR in DOLFIN type
        """
        self.du.vector().set_local(du)
        v2p(self.du.vector()).assemble()
        v2p(self.du.vector()).ghostUpdate()

        self.dR.vector().set_local(np.zeros(self.local_dof_u))
        v2p(self.dR.vector()).assemble()
        v2p(self.dR.vector()).ghostUpdate()

        A_T = transpose(A)

        solverBwd = LUSolver("mumps")
        solverBwd.solve(A_T, self.dR.vector(), self.du.vector())
        v2p(self.dR.vector()).assemble()
        v2p(self.dR.vector()).ghostUpdate()
        return self.dR.vector().get_local()

if __name__ == "__main__":
    n = 2
    mesh = UnitSquareMesh(n, n)
    fea = FEA(mesh)

    fea.f.assign(fea.f_ex)
    print(getFuncArray(fea.f))
    print(SpatialCoordinate(fea.mesh))
    
    # fea.solve(report=False)
    state_error = errorNorm(fea.u_ex, fea.u)
    # A = assembleMatrix(fea.dR_du, bcs=fea.bc())
    A = assemble(fea.dR_du)
    b = assemble(fea.R())
    fea.bc().apply(A,b)
    A_,b_ = assembleSystem(fea.dR_du, fea.R(), bcs=fea.bc())
    print(convertToDense(m2p(A)))
    print(v2p(b).getArray())
    print(convertToDense(m2p(A_)))
    print(v2p(b_).getArray())
    # print(getFuncArray(fea.u))
    # print(getFuncArray(fea.f))
    # print(fea.VF.tabulate_dof_coordinates())
    print("Error in solve():", state_error)
    print("number of states dofs:", fea.total_dofs_u)
    File('f_dolfin.pvd') << fea.f
    File('u_dolfin.pvd') << fea.u
    plot(fea.u)
    # plt.show()
