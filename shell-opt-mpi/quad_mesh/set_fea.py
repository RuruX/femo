from dolfinx import *
from dolfinx.io import XDMFFile
from dolfinx.fem import (locate_dofs_geometrical, assemble_matrix, assemble_vector,
                         apply_lifting, set_bc, locate_dofs_topological,assemble_scalar)
import dolfinx.la
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
import numpy as np
import ufl
from ufl import (MixedElement, VectorElement, FiniteElement, TestFunctions,
                 TestFunction, TrialFunction,
                 derivative, diff, dx, grad, inner, variable, SpatialCoordinate,
                 CellNormal,as_vector,Jacobian,sqrt,dot,cross,as_matrix,sym,indices,
                 as_tensor,split,Cell)
from petsc4py import PETSc
import dolfinx.geometry

"""
Shape optimization problem of a cantilever Reissner-Mindlin plate under bending. Minimizing the tip deflection of the plate with respect to the thickness distribution.
This example uses quadrilateral elements with dolfinx.
"""

def computeMatVecProductFwd(A, x):
    """
    Compute y = A * x
    A: ufl form matrix
    x: ufl function
    """
    y = A * x
    return assemble_vector(y).getArray()


def computeMatVecProductBwd(A, x):
    """
    Compute y = A.T * x
    A: ufl form matrix
    x: ufl function
    """
    A_p = assemble_matrix(A)
    x_p = x.vector
    y = x_p.copy()
    A_p.assemble()
    A_p.multTranspose(x_p, y)  # y = A'*x
    return y.getArray()

def update(f, f_ind, f_values):
    """
    f: dolfin function
    f_values: numpy array
    """
    f.vector.setValues(f_ind, f_values)
    f.vector.assemble()
    f.vector.ghostUpdate()
    
def getGlobalIndices(u_):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    u_PETSc = u_.vector
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
    Return a list of points and a list of weights for integration over the
    interval (-1,1), using ``n`` quadrature points.  
    """
    if(n==1):
        xi = [Constant(mesh,0.0),]
        w = [Constant(mesh,2.0),]
        return (xi,w)
    if(n==2):
        xi = [Constant(mesh,-0.5773502691896257645091488),
              Constant(mesh,0.5773502691896257645091488)]
        w = [Constant(mesh,1.0),
             Constant(mesh,1.0)]
        return (xi,w)
    if(n==3):
        xi = [Constant(mesh,-0.77459666924148337703585308),
              Constant(mesh,0.0),
              Constant(mesh,0.77459666924148337703585308)]
        w = [Constant(mesh,0.55555555555555555555555556),
             Constant(mesh,0.88888888888888888888888889),
             Constant(mesh,0.55555555555555555555555556)]
        return (xi,w)
    if(n==4):
        xi = [Constant(mesh,-0.86113631159405257524),
              Constant(mesh,-0.33998104358485626481),
              Constant(mesh,0.33998104358485626481),
              Constant(mesh,0.86113631159405257524)]
        w = [Constant(mesh,0.34785484513745385736),
             Constant(mesh,0.65214515486254614264),
             Constant(mesh,0.65214515486254614264),
             Constant(mesh,0.34785484513745385736)]
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


def free_end(x):
    """Marks the leftmost points of the cantilever"""
    return np.isclose(x[0], 20.0)

def left(x):
    """Marks left part of boundary, where cantilever is attached to wall"""
    return np.isclose(x[0], 0.0)
    

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

    def __init__(self, mesh):
        self.mesh = mesh
        self.cell = cell = mesh.ufl_cell()
        
        # Problem parameters:
        self.E = E = Constant(mesh,4.32e8) # Young's modulus
        self.nu = nu = Constant(mesh,0.) # Poisson ratio
        self.f = f = Constant(mesh,(0,0,-10)) # Body force per unit volume
        self.DOLFIN_EPS = Constant(mesh,1.0e-8)

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

        # Locate all facets at the free end and assign them value 1
        free_end_facets = locate_entities_boundary(self.mesh, 1, free_end)
        mt = dolfinx.mesh.MeshTags(self.mesh, 1, free_end_facets, 1)
        self.ds = ufl.Measure("ds", subdomain_data=mt)
        
        self.rightChar = ufl.conditional(ufl.gt(X_mid[0],1-self.DOLFIN_EPS),1,Constant(mesh,0))
        
        VE = VectorElement("Lagrange",cell,1)
        WE = MixedElement([VE,VE])
        TE = FiniteElement("CG",cell,1)
        self.W = W = FunctionSpace(self.mesh,WE)
#        self.VT = VT = FunctionSpace(self.mesh,'DG',0)
        self.VT = VT = FunctionSpace(self.mesh,TE)

        self.dX = dX = dx(metadata={"quadrature_degree":0})
        self.n = CellNormal(mesh)

        self.w = Function(self.W)
        self.u_mid, self.theta = split(self.w)
        self.dw = TestFunction(self.W)
        self.du_mid, self.dtheta = split(self.dw)
        self.h = Function(VT)
        
        # Integrate through the thickness numerically to get total energy (where dx
        # is a UFL Measure, integrating over midsurface area).
        dxi2 = ThroughThicknessMeasure(3,self.h)
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
        
        self.num_elements = mesh.topology.index_map(mesh.topology.dim).size_global
        
        self.local_dof_u = len(self.w.vector.getArray())
        self.local_dof_f = len(self.h.vector.getArray())
        # Ghost points are not included in the indices of u
        # for plate1, dof_u = 396, dof_f = 86
        self.ind_u = getGlobalIndices(self.w)[:self.local_dof_u]
        self.dof_u = self.W.dofmap.index_map.size_global
        self.ind_f = getGlobalIndices(self.h)[:self.local_dof_f]
        self.dof_f = self.VT.dofmap.index_map.size_global
#        print('global indices of local entities of f:',self.ind_f)
#        print('global indices of local entities of u:',self.ind_u)
#        print('global dofs of f:',self.dof_f)
#        print('global dofs of u:',self.dof_u)
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
        beta = Constant(self.mesh,1e-1)
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

#    def penalty(self):
#        beta = Constant(1e-1)
#        VE = VectorElement("CG",self.cell,1)
#        V = FunctionSpace(self.mesh, VE)
#        n = project(self.n,V)
#        _,theta = split(self.w)
#        return beta*self.E*self.h*dot(theta,n)*dot(theta,n)*self.dX

    def bc(self):
        u0 = Function(self.W)
        u0.vector.set(0.0)
        locate_fixed_BC = [locate_dofs_geometrical((self.W.sub(0),self.W.sub(0).collapse()),
                            lambda x: np.isclose(x[0],0.,atol=1e-6)),
                           locate_dofs_geometrical((self.W.sub(1),self.W.sub(1).collapse()),
                            lambda x: np.isclose(x[0], 0. ,atol=1e-6)),]
#        print("#bc1 ",locate_fixed_BC)
        bcs = [DirichletBC(u0,locate_fixed_BC)]
        return bcs

    def regularization(self):
        alpha = Constant(self.mesh,1e-1)
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

        return self.rightChar*Constant(self.mesh,0.5)*dot(self.u_mid,self.u_mid)*self.ds \
                + self.regularization()

    def constraint(self, h):
        return self.h*self.dX

    def solveLinearFwd(self, A, dR):
        """
        solve linear system dR = dR_du (A) * du
        """
        dR_p = self.dR.vector
        dR_p.setValues(self.ind_u, dR)
        dR_p.assemble()
        dR_p.ghostUpdate()

        self.du.vector.set(0.0)
        self.du.vector.assemble()
        self.du.vector.ghostUpdate()
        
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setOperators(A)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')

        ksp.solve(self.du.vector, dR_p)
        self.du.vector.assemble()
        self.du.vector.ghostUpdate()
        return self.du.vector.getArray()


    def solveLinearBwd(self, A, du):
        """
        solve linear system du = dR_du.T (A_T) * dR
        """
        dR_array = self.solveLinearFwd(A.transpose(), du)
        return dR_array
    
    def solveNonlinear(self):

        problem = fem.LinearProblem(self.dR_du, -self.R, bcs=self.bc(), 
                                    petsc_options={
                                    "ksp_type": "gmres", 
                                    "pc_type": "jacobi"})
#        problem = fem.LinearProblem(self.dR_du, -self.R, bcs=self.bc(), 
#                                    petsc_options={
#                                     "ksp_type": "preonly",
#                                     "pc_type": "lu",
#                                     "pc_factor_mat_solver_type": "mumps"})
        self.w = problem.solve()
        u_vec = self.w.sub(0).compute_point_values()
        print(u_vec)
        
#        # Create nonlinear problem and Newton solver
#        solver = NewtonSolver(MPI.COMM_WORLD)
#        solver.setF(problem.F, problem.vector())
#        solver.setJ(problem.J, problem.matrix())
#        solver.set_form(problem.form)
#        solver.convergence_criterion = "incremental"
#        solver.rtol = 1e-6

#        problem = NonlinearVariationalProblem(self.R, self.w, self.bc(), J=self.dR_du)
#        solver  = NonlinearVariationalSolver(problem)
#        prm = solver.parameters
#        prm['newton_solver']['absolute_tolerance'] = 1E-8
#        prm['newton_solver']['relative_tolerance'] = 1E-3
#        prm['newton_solver']['linear_solver'] = 'mumps'
#        solver.solve()
        
#        print(assemble(inner(self.w,self.w)*self.dX))
#        print(assemble(dot(self.u_mid,self.u_mid)*self.dX))
#        print('elastic energy:', assemble(self.elasticEnergy))
        
#        print('penalty term:', assemble(self.penalty()))
#        print('bending energy:', assemble(self.E_b))
#        print('shearing energy:', assemble(self.E_s))
#        print(self.h.vector().get_local())
        
        
if __name__ == '__main__':

# Import manifold mesh of topological dimension 2 and geometric dimension 3:

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "meshout_quad.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")

    fea = set_fea(mesh)
    rank = MPI.COMM_WORLD.Get_rank()
    fea.h.vector.set(0.25)
#    dRdf_petsc = m2p(assemble(fea.dR_df)).convert("dense")
#    print(dRdf_petsc.getDenseArray())
    fea.solveNonlinear()
    
    dolfinx.io.VTKFile("u.pvd").write(fea.w.sub(0))   
    # Save solution in XDMF format
    with XDMFFile(MPI.COMM_WORLD, "output.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(fea.w.sub(0))

    print('objective:', assemble_scalar(fea.objective(fea.u_mid)))
    
    # Assemble linear system
    a = fea.dR_du
    L = fea.R
    bcs = fea.bc()
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    
#   set values for a petsc vector 
#    n = fea.local_dof_u
#    ind = fea.ind_u
#    val = 0.1*ind
##    print(n,ind,val)
#    b.setValues(ind,val)
#    b.assemblyBegin()
#    b.assemblyEnd()
#    print(b.getArray())
#    fea.w.vector.set(1.0)
#    b = computeMatVecProductFwd(fea.dR_du, fea.w)
#    print(b)
#    fea.dR.vector.set(1.0)
#    x = computeMatVecProductBwd(fea.dR_du, fea.dR)
#    print(x)
    dR = fea.solveLinearFwd(A, b.getArray())
    print(rank, fea.du.vector.getArray())
    dR = fea.solveLinearBwd(A, b.getArray())
    print(rank, fea.dR.vector.getArray())
#    print('penalty term:', assemble_scalar(fea.penalty()))





