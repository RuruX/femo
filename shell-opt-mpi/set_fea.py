from dolfin import *
import numpy as np
from ufl import indices, Jacobian, shape
from petsc4py import PETSc

"""
Shape optimization problem of a cantilever Reissner-Mindlin plate under bending. Minimizing the tip deflection of the plate with respect to the thickness distribution.
This example uses triangle elements.
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
    return y.getArray()

def update(f, f_values):
    """
    f: dolfin function
    f_values: numpy array
    """
    f.vector().set_local(f_values)
    v2p(f.vector()).assemble()
    v2p(f.vector()).ghostUpdate()
    
def getGlobalIndices(u_):
    comm = MPI.comm_world
    rank = comm.Get_rank()
    u_PETSc = v2p(u_.vector())
    ind = u_PETSc.getLGMap().getIndices()
    return ind

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
    
def getQuadRule(n):
    """
    (copy-pasted from tIGAr and ShNAPr)
    
    Return a list of points and a list of weights for integration over the
    interval (-1,1), using ``n`` quadrature points.  
    """
    if(n==1):
        xi = [Constant(0.0),]
        w = [Constant(2.0),]
        return (xi,w)
    if(n==2):
        xi = [Constant(-0.5773502691896257645091488),
              Constant(0.5773502691896257645091488)]
        w = [Constant(1.0),
             Constant(1.0)]
        return (xi,w)
    if(n==3):
        xi = [Constant(-0.77459666924148337703585308),
              Constant(0.0),
              Constant(0.77459666924148337703585308)]
        w = [Constant(0.55555555555555555555555556),
             Constant(0.88888888888888888888888889),
             Constant(0.55555555555555555555555556)]
        return (xi,w)
    if(n==4):
        xi = [Constant(-0.86113631159405257524),
              Constant(-0.33998104358485626481),
              Constant(0.33998104358485626481),
              Constant(0.86113631159405257524)]
        w = [Constant(0.34785484513745385736),
             Constant(0.65214515486254614264),
             Constant(0.65214515486254614264),
             Constant(0.34785484513745385736)]
        return (xi,w)
    
    print("ERROR: invalid number of quadrature points requested.")
    exit()

def getQuadRuleInterval(n,L):
    """
    Returns an ``n``-point quadrature rule for the interval 
    (-``L``/2,``L``/2), consisting of a list of points and list of weights.
    """
    xi_hat, w_hat = getQuadRule(n)
    xi = []
    w = []
    for i in range(0,n):
        xi += [L*xi_hat[i]/2.0,]
        w += [L*w_hat[i]/2.0,]
    return (xi,w)

class ThroughThicknessMeasure:
    """
    Class to represent a local integration through the thickness of a shell.
    The ``__rmul__`` method is overloaded for an instance ``dxi2`` to be
    used like ``volumeIntegral = volumeIntegrand*dxi2*dx``, where
    ``volumeIntegrand`` is a python function taking a single parameter,
    ``xi2``.
    """
    def __init__(self,nPoints,h):
        """
        Integration uses a quadrature rule with ``nPoints`` points, and assumes
        a thickness ``h``.
        """
        self.nPoints = nPoints
        self.h = h
        self.xi2, self.w = getQuadRuleInterval(nPoints,h)

    def __rmul__(self,integrand):
        """
        Given an ``integrand`` that is a Python function taking a single
        ``float`` parameter with a valid range of ``-self.h/2`` to 
        ``self.h/2``, return the (numerical) through-thickness integral.
        """
        integral = 0.0
        for i in range(0,self.nPoints):
            integral += integrand(self.xi2[i])*self.w[i]
        return integral


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 20.0)

class Left(SubDomain):
   def inside(self, x, on_boundary):
      return near(x[0], 0.0)


class set_fea(object):

    def __init__(self, mesh):

        self.mesh = mesh
        self.cell = cell = mesh.ufl_cell()

        # Problem parameters:
        self.E = E = Constant(4.32e8) # Young's modulus
        self.nu = nu = Constant(0.) # Poisson ratio
        self.f = f = Constant((0,0,-10)) # Body force per unit volume

        # Reference configuration of midsurface:
        X_mid = SpatialCoordinate(mesh)

        # Normal vector to each element is the third basis vector of the
        # local orthonormal basis (indexed from zero for consistency with Python):
        self.E2 = E2 = CellNormal(mesh)

        # Local in-plane orthogonal basis vectors, with 0-th basis vector along
        # 0-th parametric coordinate direction (where Jacobian[i,j] is the partial
        # derivatiave of the i-th physical coordinate w.r.t. to j-th parametric
        # coordinate):
        A0 = as_vector([Jacobian(mesh)[j,0] for j in range(0,3)])
        self.E0 = E0 = A0/sqrt(dot(A0,A0))
        self.E1 = E1 = cross(E2,E0)

        # Matrix for change-of-basis to/from local/global Cartesian coordinates;
        # E01[i,j] is the j-th component of the i-th basis vector:
        self.E01 = E01 = as_matrix([[E0[i] for i in range(0,3)],
                         [E1[i] for i in range(0,3)]])

        # Voigt notation material stiffness matrix for plane stress:
        self.D = D = (E/(1.0 - nu*nu))*as_matrix([[1.0,  nu,   0.0         ],
                                         [nu,   1.0,  0.0         ],
                                         [0.0,  0.0,  0.5*(1.0-nu)]])


        right = Right()
        left = Left()
        print(type(self.mesh))
        boundaries = MeshFunction("size_t", self.mesh, 1)
        boundaries.set_all(0) 
        right.mark(boundaries, 1) 
        left.mark(boundaries, 2) 
        self.ds = Measure('ds',subdomain_data=boundaries)
        
        self.boundaries = boundaries
        self.rightChar = conditional(gt(X_mid[0],1-DOLFIN_EPS),1,Constant(0))
        
        VE = VectorElement("Lagrange",cell,1)
#        VE_CR = VectorElement("CR",cell,1)
        WE = MixedElement([VE,VE])
        self.W = W = FunctionSpace(self.mesh,WE)
#        self.VT = VT = FunctionSpace(self.mesh,'DG',0)
        self.VT = VT = FunctionSpace(self.mesh,'CG',1)

        self.dX = dX = dx(metadata={"quadrature_degree":0})
        self.n = CellNormal(mesh)

        self.w = Function(self.W)
        self.u_mid, self.theta = split(self.w)
        self.dw = TestFunction(self.W)
        self.du_mid, self.dtheta = split(self.dw)
        self.h = Function(VT)
        
        # Integrate through the thickness numerically to get total energy (where dx
        # is a UFL Measure, integrating over midsurface area).
        dxi2 = ThroughThicknessMeasure(4,self.h)
        self.elasticEnergy = self.energyPerUnitVolume*dxi2*dX
        self.E_b = self.e_b*dxi2*dX
        self.E_s = self.e_s*dxi2*dX
        self.R = self.pdeRes()


        self.wh = TrialFunction(W)

        self.dR_du = derivative(self.R, self.w)
        self.dR_df = derivative(self.R, self.h)

        self.dJ_du = derivative(self.objective(self.u_mid), self.w)
        self.dJ_df = derivative(self.objective(self.u_mid), self.h)
        self.dC_df = derivative(self.constraint(self.h), self.h)

        self.local_dof_u = len(W.dofmap().dofs())
        self.local_dof_f = len(VT.dofmap().dofs())
        
        # Ghost points are not included in the indices of u
        # for plate1, dof_u = 396, dof_f = 86
        self.ind_u = getGlobalIndices(self.w)[:self.local_dof_u]
        self.dof_u = W.dofmap().global_dimension()
        
        self.ind_f = getGlobalIndices(self.h)[:self.local_dof_f]
        self.dof_f = VT.dofmap().global_dimension()
        
        # solving the next step of Newton's iteration.
        self.du = Function(self.W)
        self.dR = Function(self.W)
        self.df = Function(VT)
        

    def u(self, xi2):
        """
        Displacement at through-thickness coordinate xi2:
        Formula (7.1) from http://www2.nsysu.edu.tw/csmlab/fem/dyna3d/theory.pdf
        """
        return self.u_mid - xi2*cross(self.E2,self.theta)


    def gradu_local(self, xi2):
        """
        In-plane gradient components of displacement in the local orthogonal
        coordinate system:
        """
        gradu_global = grad(self.u(xi2)) # (3x3 matrix, zero along E2 direction)
        i,j,k,l = indices(4)
        return as_tensor(self.E01[i,k]*gradu_global[k,l]*self.E01[j,l],(i,j))


    def eps(self, xi2):
        """
        In-plane strain components of local orthogonal coordinate system at
        through-thickness coordinate xi2, in Voigt notation:
        """
        eps_mat = sym(self.gradu_local(xi2))
        return as_vector([eps_mat[0,0], eps_mat[1,1], 2*eps_mat[0,1]])

    def gamma_2(self, xi2):
        """
        Transverse shear strains in local coordinates at given xi2, as a vector
        such that gamma_2(xi2)[i] = 2*eps[i,2], for i in {0,1}
        """
        dudxi2_global = -cross(self.E2,self.theta)
        i,j = indices(2)
        dudxi2_local = as_tensor(dudxi2_global[j]*self.E01[i,j],(i,))
        gradu2_local = as_tensor(dot(self.E2,grad(self.u(xi2)))[j]*self.E01[i,j],(i,))
        return dudxi2_local + gradu2_local


    def energyPerUnitVolume(self, xi2):
        """
        Elastic energy per unit volume at through-thickness coordinate xi2:
        """
        G = self.E/(2*(1+self.nu))
        return 0.5*(dot(self.eps(xi2), self.D*self.eps(xi2))\
                    + G*inner(self.gamma_2(xi2),self.gamma_2(xi2)))
                    
    def e_b(self, xi2):
        G = self.E/(2*(1+self.nu))
        return 0.5*dot(self.eps(xi2), self.D*self.eps(xi2))
                        
    def e_s(self, xi2):
        G = self.E/(2*(1+self.nu))
        return 0.5*G*inner(self.gamma_2(xi2),self.gamma_2(xi2))
                    
    def pdeRes(self):
        beta = Constant(1e-1)
#        
#        """ Tip Load """
#        F = derivative(self.elasticEnergy,self.w,self.dw) \
#            - self.rightChar*inner(self.f,self.du_mid)*self.ds \
#            + beta*self.E*self.h*dot(self.theta,self.n)*dot(self.dtheta,self.n)*self.dX
##            
        """ Force per unit area """
        F = derivative(self.elasticEnergy,self.w,self.dw) \
            - inner(self.f,self.du_mid)*self.dX \
            + beta*self.E*self.h*dot(self.theta,self.n)*dot(self.dtheta,self.n)*self.dX
#            
#        """ Force per unit volume """
#        F = derivative(self.elasticEnergy,self.w,self.dw) \
#            - inner(self.f,self.du_mid)*self.h*self.dX \
#            + beta*self.E*self.h*dot(self.theta,self.n)*dot(self.dtheta,self.n)*self.dX
        return F

    def penalty(self):
        beta = Constant(1e-1)
        VE = VectorElement("CG",self.cell,1)
        V = FunctionSpace(self.mesh, VE)
        n = project(self.n,V)
        _,theta = split(self.w)
        return beta*self.E*self.h*dot(theta,n)*dot(theta,n)*self.dX

    def bc(self):
        boundaries = self.boundaries
        bcs = DirichletBC(self.W, Constant(6*(0,)), boundaries, 2)
        return bcs

    def regularization(self):
        alpha = Constant(1e-1)
        # Tikhonov regularization: Induces artificial BCs
#        return 0.5*alpha*inner(grad(self.h),grad(self.h))*self.dX
        
#         L2 regularization
#        return 0.5*alpha*inner(self.h,self.h)*self.dX

#         L2 + jumps
#        return 0.5*alpha*(inner(self.h,self.h)*self.dX + 
#                    avg(self.h)*(jump(self.h)**2)*dS)

#        # H1 regularization
        return 0.5*alpha*dot(grad(self.h), grad(self.h))*self.dX


    def objective(self, u_mid):
        """ Minimizing the tip deflection """

        return self.rightChar*Constant(0.5)*dot(self.u_mid,self.u_mid)*self.ds \
                + self.regularization()

    def constraint(self, h):
        return self.h*self.dX

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

        solverBwd = LUSolver("mumps")
        solverBwd.solve(A_T, self.dR.vector(), self.du.vector())
        v2p(self.dR.vector()).assemble()
        v2p(self.dR.vector()).ghostUpdate()
        return self.dR.vector().get_local()

    
    def solveNonlinear(self):
        problem = NonlinearVariationalProblem(self.R, self.w, self.bc(), J=self.dR_du)
        solver  = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-8
        prm['newton_solver']['relative_tolerance'] = 1E-3
        prm['newton_solver']['linear_solver'] = 'mumps'
        solver.solve()
        
#        solve(self.R == 0, self.w, self.bc(), J = self.dR_du,
#                solver_parameters={"newton_solver":{"maximum_iterations":1, "error_on_nonconvergence":False}})
#        
#        print(assemble(inner(self.w,self.w)*self.dX))
#        print(assemble(dot(self.u_mid,self.u_mid)*self.dX))
#        print('elastic energy:', assemble(self.elasticEnergy))
        
#        print('penalty term:', assemble(self.penalty()))
#        print('bending energy:', assemble(self.E_b))
#        print('shearing energy:', assemble(self.E_s))
#        print(self.h.vector().get_local())
        
        
if __name__ == '__main__':

    mesh = Mesh()
    filename = "plate3.xdmf"
    file = XDMFFile(mesh.mpi_comm(),filename)
    file.read(mesh)


    fea = set_fea(mesh)
    rank = MPI.comm_world.Get_rank()
    print('number of elements:', fea.dof_f)
    fea.h.vector().set_local(0.02*np.ones(fea.local_dof_f))
#    dRdf_petsc = m2p(assemble(fea.dR_df)).convert("dense")
#    print(dRdf_petsc.getDenseArray())
    fea.solveNonlinear()
    print('objective:', assemble(fea.objective(fea.u_mid)))
#    A,B = assemble_system(fea.dR_du, fea.R, bcs=[fea.bc()])
#    b = computeMatVecProductFwd(A, fea.w)
#    x = computeMatVecProductBwd(A, fea.dR)
#    dR = fea.solveLinearFwd(A, B.get_local())
#    print(rank, fea.du.vector().get_local())
#    dR = fea.solveLinearBwd(A, B.get_local())
#    print(rank, fea.dR.vector().get_local())
    print('penalty term:', assemble(fea.penalty()))





