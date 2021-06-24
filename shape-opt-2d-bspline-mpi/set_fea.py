from dolfin import *
import numpy as np
import ufl
from petsc4py import PETSc

"""
A discussion on the choice of linear solvers for both solve for the next step of Newton iterations' and solve for the 
states of the PDE:

Direct solver: MUMPS (state-of-art)
Iterative solver: KSP.GRMES with ASM as the preconditioner and LU as the sub preconditioner 

(Wikipedia) -- ASM (Additive Schwarz Method) solves a boundary value problem for a partial differential equation approximately by splitting it into boundary value problems on smaller domains and adding the results.

The direct solver needs much memory and is said to be not suitable for applications in 3D. But MUMPS works fine until now.

The Krylov solver sometimes takes (much) more time in parallel, where the number of iterations varies from 2 to 600. (It always takes only one iteration in serial since it is solved actually by LU).
"""


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

def solveKSP(A,b,u):
    """
    solve linear system A*u=b
    """

    ksp = PETSc.KSP().create() 
    ksp.setType(PETSc.KSP.Type.GMRES)
    A.assemble()
    ksp.setOperators(A)
    
    ksp.setFromOptions()
    pc = ksp.getPC()
    pc.setType("asm")
    pc.setASMOverlap(1)
    ksp.setUp()
    localKSP = pc.getASMSubKSP()[0]
    localKSP.setType(PETSc.KSP.Type.GMRES)
    localKSP.getPC().setType("lu")
    ksp.setGMRESRestart(50)
    ksp.setConvergenceHistory()
    ksp.solve(b,u)
    history = ksp.getConvergenceHistory()
    print('Converged in', ksp.getIterationNumber(), 'iterations.')



class set_fea(object):

    def __init__(self, num_elements):

        self.num_elements = num_elements
        self.length = length = 1.0
        N_fac = 8
        self.height = height = self.length/N_fac
        d = 2
        k = 1
        self.mesh = RectangleMesh(Point(0,0),Point(length,height),N_fac*num_elements,num_elements)

        cell = self.mesh.ufl_cell()

        VE = VectorElement("CG",cell,k)
        LE = FiniteElement("R",cell,0)


        self.I = Identity(d)
        x = SpatialCoordinate(self.mesh)
        self.dX = dx(metadata={"quadrature_degree":2*k})
        
        topChar = conditional(gt(x[1],Constant(self.height-DOLFIN_EPS)),1.0,Constant(0.0))
        self.h = Constant((0,-1))*topChar         # h: applied downward force on the top

        self.V = FunctionSpace(self.mesh, VE)     # V: the function space of the displacements
        self.VHAT = FunctionSpace(self.mesh, VE)  # VE: the function space of the displacements of the mesh
        self.L = FunctionSpace(self.mesh, LE)     

        self.uh = TrialFunction(self.V)
        self.u = Function(self.V)
        self.v = TestFunction(self.V)
        self.uhat = Function(self.VHAT)
        self.lam = Function(self.L)

        self.R = self.pdeRes(self.u,self.v,self.uhat)
        self.dR_du = derivative(self.R, self.u)
        self.dR_df = derivative(self.R, self.uhat)
        self.dJ_du = derivative(self.objective(self.u,self.uhat), self.u)
        self.dJ_df = derivative(self.objective(self.u,self.uhat), self.uhat)
        self.dC_df = derivative(self.constraint(self.uhat,self.lam), self.uhat)

        self.local_dof_u = len(self.V.dofmap().dofs())
        self.local_dof_f = len(self.VHAT.dofmap().dofs())
        
        # Ghost points are not included in the indices of u
        self.ind_u = self.getGlobalIndices(self.u)[:self.local_dof_u]
        self.dof_u = 2*(num_elements+1)*(N_fac*num_elements+1)
        
        self.ind_f = self.ind_u
        self.dof_f = self.dof_u
        
        self.du = Function(self.V)
        self.dR = Function(self.V)
        self.df = Function(self.VHAT)
        
            
    def getGlobalIndices(self, u_):
        comm = MPI.comm_world
        rank = comm.Get_rank()
        u_PETSc = v2p(u_.vector())
        ind = u_PETSc.getLGMap().getIndices()
        return ind

    def gradx(self,f,u):
        return dot(grad(f), inv(self.I + grad(u)))

    def Ctimes(self,eps):
        K = Constant(1.0e6)
        mu = Constant(1.0e6)
        return K*tr(eps)*self.I + 2.0*mu*(eps - tr(eps)*self.I/3.0)

    def J(self,uhat):
        return det(self.I + grad(uhat))

    def pdeRes(self,u,v,uhat):
        epsu = sym(self.gradx(u,uhat))
        gradv = self.gradx(v,uhat)
        sigma = self.Ctimes(epsu)
        return inner(sigma,gradv)*self.J(uhat)*self.dX - dot(self.h,v)*ds

    def bcu(self):
        x = SpatialCoordinate(self.mesh)
        leftStr = "x[0] < DOLFIN_EPS"
        bcu = DirichletBC(self.V,Constant((0,0)),leftStr)
        return bcu

    def objective(self,u,uhat):
        return Constant(1.0)*dot(self.h,u)*ds
        
    def constraint(self,uhat,lam):
        return lam*(self.J(uhat)-Constant(1.0))*self.dX

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
    
    # ---------------- Solve with Direct Solver ------------------
        solverFwd = LUSolver("mumps")
        solverFwd.solve(A, self.du.vector(), self.dR.vector())
        v2p(self.du.vector()).assemble()
        v2p(self.du.vector()).ghostUpdate()
        return self.du.vector().get_local()

    # ---------------- Solve with Iterative Solver ------------------
#        A_p = m2p(A)
#        A_p.assemble()
#        b_p = v2p(self.dR.vector())
#        b_p.assemble()
#        u_p = PETSc.Vec().create()
#        u_p.createNest([v2p(self.du.vector())])
#        u_p.setUp()
#        u_p.assemble()
#        
#        import timeit
#        start = timeit.default_timer()
#        solveKSP(A_p,b_p,u_p)
#        stop = timeit.default_timer()
#        print('Time for solveLinearFwd:', stop-start)
#        
#        v2p(self.du.vector()).assemble()
#        v2p(self.du.vector()).ghostUpdate()
#        return self.du.vector().get_local()


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

# ---------------- Solve with Direct Solver ------------------
        solverBwd = LUSolver("mumps")
        solverBwd.solve(A_T, self.dR.vector(), self.du.vector())
        v2p(self.dR.vector()).assemble()
        v2p(self.dR.vector()).ghostUpdate()
        return self.dR.vector().get_local()

# ---------------- Solve with Iterative Solver ------------------
#        A_p = m2p(A_T)
#        A_p.assemble()
#        b_p = v2p(self.du.vector())
#        b_p.assemble()
#        u_p = PETSc.Vec().create()
#        u_p.createNest([v2p(self.dR.vector())])
#        u_p.setUp()
#        u_p.assemble()
#        
#        import timeit
#        start = timeit.default_timer()
#        solveKSP(A_p,b_p,u_p)
#        stop = timeit.default_timer()
#        print('rank', MPI.comm_world.Get_rank(), 
#                'Time for solveLinearBwd:', stop-start)
#
#        v2p(self.dR.vector()).assemble()
#        v2p(self.dR.vector()).ghostUpdate()
#        return self.dR.vector().get_local()

    
    def solveNonlinear(self):
    
# ---------------- Solve with Direct Solver ------------------
        Rh = self.pdeRes(self.uh,self.v,self.uhat)
        a = lhs(Rh)
        L = rhs(Rh)
        problem = LinearVariationalProblem(a, L, self.u, self.bcu())
        solver  = LinearVariationalSolver(problem)
        prm = solver.parameters
        prm['linear_solver'] = 'mumps'
        solver.solve()


# ---------------- Solve with Iterative Solver ------------------
#        Rh = self.pdeRes(self.uh,self.v,self.uhat)
#        a = lhs(Rh)
#        L = rhs(Rh)
#        A,b = assemble_system(a, L, bcs=[self.bcu()])
#        A_p = m2p(A)
#        b_p = v2p(b)

#        u_p = PETSc.Vec().create()
#        u_p.createNest([v2p(self.u.vector())])
#        u_p.setUp()
#        
#        solveKSP(A_p,b_p,u_p)
#        
#        v2p(self.u.vector()).ghostUpdate()

        
if __name__ == '__main__':

    fea = set_fea(num_elements=200)
    rank = MPI.comm_world.Get_rank()

    fea.u.vector().set_local(np.ones(fea.local_dof_u))
    A,B = assemble_system(fea.dR_du, fea.R, bcs=[fea.bcu()])
#    b = computeMatVecProductFwd(A, fea.u)
#    x = computeMatVecProductBwd(A, fea.dR)
    dR = fea.solveLinearFwd(A, B.get_local())
    print(rank, fea.du.vector().get_local())
    dR = fea.solveLinearBwd(A, B.get_local())
    print(rank, fea.dR.vector().get_local())
    
