from dolfin import *
import numpy as np
import ufl
from scipy.sparse import csr_matrix
from petsc4py import PETSc

def m2p(A):
    return as_backend_type(A).mat()

def v2p(v):
    return as_backend_type(v).vec()

def transpose(A):
    """
    Transpose for matrix of DOLFIN type
    """
    return PETScMatrix(as_backend_type(A).mat().transpose(PETSc.Mat(MPI.comm_world)))

def computeMatVecProductFwd(A, x):
    """
    Compute y = A * x
    A: ufl form matrix
    x: ufl function
    """
    A_p = m2p(A)
    y = A_p * v2p(x.vector())
    y.assemble()
#    y.ghostUpdate()
    return y.getArray()


def computeMatVecProductBwd(A, R):
    """
    Compute y = A.T * R
    A: ufl form matrix
    R: ufl function
    """
    row, col = m2p(A).getSizes()
    y = PETSc.Vec().create()
    y.setSizes(col)
    y.setUp()
    m2p(A).multTranspose(v2p(R.vector()),y)
    y.assemble()
#    y.ghostUpdate()
    return y.getArray()

def update(f, f_values):
    """
    f: dolfin function
    f_values: numpy array
    """
    f.vector().set_local(f_values)
    v2p(f.vector()).assemble()
    v2p(f.vector()).ghostUpdate()
    
    
class set_fea(object):

    def __init__(self, num_elements):
        
        self.num_elements = num_elements
        k = 1
        self.mesh = UnitCubeMesh(self.num_elements,self.num_elements,self.num_elements)
        UE = FiniteElement('CG', self.mesh.ufl_cell(), k)
        FE = FiniteElement('DG', self.mesh.ufl_cell(), 0)

        self.dof_f = 6*num_elements**3
        self.dof_u = (num_elements+1)**3
        
        self.V = FunctionSpace(self.mesh, UE)
        self.F = FunctionSpace(self.mesh, FE)
        self.u = Function(self.V)
        self.v = TestFunction(self.V)
        self.f = Function(self.F)
        self.n = FacetNormal(self.mesh)
        self.R = self.pdeRes(self.u, self.v, self.f)
        self.dR_du = derivative(self.R,self.u)
        self.dR_df = derivative(self.R,self.f)
        self.dJ_du = derivative(self.objective(self.u,self.f), self.u)
        self.dJ_df = derivative(self.objective(self.u,self.f), self.f)

        self.local_dof_f = len(self.F.dofmap().dofs())
        self.local_dof_u = len(self.V.dofmap().dofs())
        self.ind_f = self.getGlobalIndices(self.f)
        # Ghost points are not included in the indices of u
        self.ind_u = self.getGlobalIndices(self.u)[:self.local_dof_u]
        
        self.du = Function(self.V)
        self.dR = Function(self.V)
        self.df = Function(self.F)
        self.nonlinearSolver = self.setUpNonlinearSolver()
        
        
    def getGlobalIndices(self, u_):
        comm = MPI.comm_world
        rank = comm.Get_rank()
        u_PETSc = v2p(u_.vector())
        ind = u_PETSc.getLGMap().getIndices()
        return ind
    
    def updateF(self, f_new):
        update(self.f, f_new)
        
    def updateU(self, u_new):
        update(self.u, u_new)
        
    def initializeU(self):
        self.updateU(np.zeros(self.local_dof_u))
    
    def pdeRes(self,u,v,f):
        return (inner(grad(u),grad(v))-f*v)*dx
    
    def bc(self):
        bc = DirichletBC(self.V, Constant(0.0), "on_boundary")
        return bc
    
    def objective(self,u,f):
        x = SpatialCoordinate(self.mesh)
        w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
        d = 1/(2*pi**2)
        d = Expression("d*w", d=d, w=w, degree=3)
        alpha = Constant(1e-6)
        return (Constant(0.5)*inner(u-d, u-d))*dx + alpha/2*f**2*dx

    def solveLinearFwd(self, A, dR):
        """
        solve linear system dR = dR_du (A) * du
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
        solve linear system du = dR_du.T (A_T) * dR
        """
        self.du.vector().set_local(du)
        v2p(self.du.vector()).assemble()
        v2p(self.du.vector()).ghostUpdate()

        self.dR.vector().set_local(np.zeros(self.local_dof_u))
        v2p(self.dR.vector()).assemble()
        v2p(self.dR.vector()).ghostUpdate()
        
        A_T = transpose(A)
        as_backend_type(A_T).mat().assemble()

#        solverBwd = LUSolver("mumps")
        solverBwd = KrylovSolver(method="gmres", preconditioner="hypre_parasails")
        solverBwd.solve(A_T, self.dR.vector(), self.du.vector())
        v2p(self.dR.vector()).assemble()
        v2p(self.dR.vector()).ghostUpdate()
        return self.dR.vector().get_local()

    
    def setUpNonlinearSolver(self):
        problem = NonlinearVariationalProblem(self.R, self.u, self.bc(), J=self.dR_du)
        solver  = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm['newton_solver']['relative_tolerance'] = 1E-3
#        prm['newton_solver']['linear_solver'] = 'mumps'
        prm['newton_solver']['linear_solver'] = 'gmres'
        prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-6
        prm['newton_solver']['preconditioner'] = 'hypre_parasails'
#        list_krylov_solver_preconditioners()
#        info(prm, True)
        return solver
            
    def solveNonlinear(self): 
        solver = self.nonlinearSolver
        set_log_active(False)
        solver.solve()
        
if __name__ == '__main__':
    fea = set_fea(num_elements=16)
    rank = MPI.comm_world.Get_rank()
    fea.u.vector().set_local(np.ones(fea.local_dof_u))
    fea.dR.vector().set_local(np.ones(fea.local_dof_u))
    A,_ = assemble_system(fea.dR_du, fea.R, bcs=[fea.bc()])
#    b = computeMatVecProductFwd(A, fea.u)
#    x = computeMatVecProductBwd(A, fea.dR)
    import timeit
    start = timeit.default_timer()
    du = fea.solveLinearBwd(A, np.ones(fea.local_dof_u))
    stop = timeit.default_timer()
    print(rank, stop - start)
