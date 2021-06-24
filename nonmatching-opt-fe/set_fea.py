import numpy as np 
from dolfin import *
import ufl
from scipy.sparse import csr_matrix, vstack, hstack, block_diag
#from nonmatching_FE import nonmatching_FE
from petsc4py import PETSc


def m2p(A):
    return as_backend_type(A).mat()

def v2p(v):
    return as_backend_type(v).vec()

def AT_R_B(A,R,B):
    """
    Compute A^T*R*B.
    """
    return (m2p(A).transposeMatMult(m2p(assemble(R))).matMult(m2p(B)))
    
def AT_R(A,R):
    """
    Compute y = A^T*R.
    """
    y = PETSc.Vec().create()
    row, col = m2p(A).getSizes()
    y.setSizes(col)
    y.setUp()
    m2p(A).multTranspose(v2p(assemble(R)),y)
    y.assemble()
    return y

def zero_PETSc_v(num_el):
    """
    Create zeros PETSc vector with size num_el.
    """
    v = PETSc.Vec().create()
    v.setSizes(num_el)
    v.setType('seq')
    v.setUp()
    v.assemble()
    return v

def zero_PETSc_M(row, col):
    """
    Create zeros PETSc matrix with shape (row, col).
    """
    A = PETSc.Mat().create()
    A.setSizes([row, col])
    A.setType('seqaij')
    A.setUp()
    A.assemble()
    return A
    
def transferMatrix(V,Vm):
    """
    Create the transder matrix from function space V to Vm.
    """
    return PETScDMCollection.create_transfer_matrix(V,Vm)
    

class set_fea(object):

    def __init__(self, num_elements):
        
        self.num_elements = num_elements

        k = 1
        self.mesh_1 = UnitSquareMesh(self.num_elements,self.num_elements)
        self.mesh_2 = UnitSquareMesh(self.num_elements+7,self.num_elements)
        ALE.move(self.mesh_2,Constant((0,-1)))
        EPS = 1e-10
        self.mesh_m = BoundaryMesh(RectangleMesh(Point(0,-EPS),
                            Point(1,EPS),10*self.num_elements,1),"exterior")

        self.V1 = FunctionSpace(self.mesh_1, "CG", 1)
        self.V2 = FunctionSpace(self.mesh_2, "CG", 1)
        self.Vm = FunctionSpace(self.mesh_m, "DG", 0)
        self.F1 = FunctionSpace(self.mesh_1, "DG", 0)
        self.F2 = FunctionSpace(self.mesh_2, "DG", 0)
        
        assemble(TestFunction(self.Vm)*dx)
        
        self.u1 = Function(self.V1)
        self.u2 = Function(self.V2)
        self.u1m = Function(self.Vm)
        self.u2m = Function(self.Vm)
        self.v1 = TestFunction(self.V1)
        self.v2 = TestFunction(self.V2)
        self.f1 = Function(self.F1)
        self.f2 = Function(self.F2)

        self.dx = dx(metadata={"quadrature_degree":2})
        x1 = SpatialCoordinate(self.mesh_1)
        x2 = SpatialCoordinate(self.mesh_2)
        self.d1 = self.u_ex(x1)
        self.d2 = self.u_ex(x2)
        self.A_1m = transferMatrix(self.V1,self.Vm)
        self.A_2m = transferMatrix(self.V2,self.Vm)
        self.dR_du_coo_0, self.dR_df_coo_0 = self.get_coo()
        
    def updateF(self, f_new):
        self.f1.vector().set_local(f_new[:len(self.F1.dofmap().dofs())])
        self.f2.vector().set_local(f_new[-len(self.F2.dofmap().dofs()):])
        
    def updateU(self, u_new):
        self.u1.vector().set_local(u_new[:len(self.V1.dofmap().dofs())])
        self.u2.vector().set_local(u_new[-len(self.V2.dofmap().dofs()):])
        self.u1m.vector()[:] = self.A_1m*self.u1.vector()
        self.u2m.vector()[:] = self.A_2m*self.u2.vector()
        
    def initializeU(self):
        self.u1.vector().set_local(np.zeros(len(self.V1.dofmap().dofs())))
        self.u2.vector().set_local(np.zeros(len(self.V2.dofmap().dofs())))
        self.u1m.vector()[:] = self.A_1m*self.u1.vector()
        self.u2m.vector()[:] = self.A_2m*self.u2.vector()
   
    def penaltyEnergy(self, u1m, u2m):
        K = Constant(1.0*self.num_elements)
        return 0.5*K*((u1m-u2m)**2)*self.dx
   
    def setUpKSP(self):
        A_1m = self.A_1m
        A_2m = self.A_2m
        penaltyEnergy = self.penaltyEnergy(self.u1m,self.u2m)
        R1m = derivative(penaltyEnergy,self.u1m)
        R2m = derivative(penaltyEnergy,self.u2m)
        dR1m_du1m = derivative(R1m,self.u1m)
        dR1m_du2m = derivative(R1m,self.u2m)
        dR2m_du1m = derivative(R2m,self.u1m)
        dR2m_du2m = derivative(R2m,self.u2m)
        
        dR1_du1 = AT_R_B(A_1m,dR1m_du1m,A_1m)
        dR1_du2 = AT_R_B(A_1m,dR1m_du2m,A_2m)
        dR2_du1 = AT_R_B(A_2m,dR2m_du1m,A_1m)
        dR2_du2 = AT_R_B(A_2m,dR2m_du2m,A_2m)
        
        x1 = SpatialCoordinate(self.mesh_1)
        x2 = SpatialCoordinate(self.mesh_2)
        
        r1 = self.pdeRes(self.u1,self.v1,self.f1)
        r2 = self.pdeRes(self.u2,self.v2,self.f2)
        A11 = m2p(assemble(derivative(r1,self.u1)))
        A22 = m2p(assemble(derivative(r2,self.u2)))
        self.R1 = v2p(assemble(r1))
        self.R2 = v2p(assemble(r2))
        
#        --------Adding the penalty contribution to the PDE residual vectors----------
        self.R1 += AT_R(A_1m,R1m)
        self.R2 += AT_R(A_2m,R2m)
        dR1_du1 += A11
        dR2_du2 += A22

        self.A = PETSc.Mat()
        self.A.createNest([[dR1_du1,dR1_du2],
                      [dR2_du1,dR2_du2]])
        self.A.setUp()
        self.b = PETSc.Vec()
        self.b.createNest([-self.R1,-self.R2]) # add negative sign to the residuals
        self.b.setUp()
        self.u = PETSc.Vec()
        self.u.createNest([v2p(self.u1.vector()),v2p(self.u2.vector())])
        self.u.setUp()

        dR1_df1 = m2p(assemble(derivative(r1,self.f1)))
        dR2_df2 = m2p(assemble(derivative(r2,self.f2)))
        dR1_df1_csr = csr_matrix(dR1_df1.getValuesCSR()[::-1], shape=dR1_df1.size)
        dR2_df2_csr = csr_matrix(dR2_df2.getValuesCSR()[::-1], shape=dR2_df2.size)
        self.dR_df_csr = block_diag((dR1_df1_csr,dR2_df2_csr),"csr",)
        
        dR1_du1_csr = csr_matrix(dR1_du1.getValuesCSR()[::-1], shape=dR1_du1.size)
        dR1_du2_csr = csr_matrix(dR1_du2.getValuesCSR()[::-1], shape=dR1_du2.size)
        dR2_du1_csr = csr_matrix(dR2_du1.getValuesCSR()[::-1], shape=dR2_du1.size)
        dR2_du2_csr = csr_matrix(dR2_du2.getValuesCSR()[::-1], shape=dR2_du2.size)
        self.dR_du_csr = hstack([vstack([dR1_du1_csr, dR2_du1_csr]),
                                vstack([dR1_du2_csr, dR2_du2_csr])])


    def pdeRes(self,u,v,f):
        return inner(grad(u),grad(v))*self.dx + u*v*self.dx - f*v*self.dx

    def bc(self):
        bc = DirichletBC(self.V, Constant(0.0), "on_boundary")
        return bc
            
    def objective(self):
        return (assemble(self.err(self.u1,self.d1,self.f1)) +
                assemble(self.err(self.u2,self.d2,self.f2)))
                    
    def err(self,u,d,f):
        alpha = Constant(1e-3)
        return (0.5*inner(u-d, u-d) + alpha/2*f**2)*self.dx

    def compute_derivative(self):
        self.setUpKSP()
        return self.dR_du_csr.tocoo(), self.dR_df_csr.tocoo()

    def get_coo(self):
        self.setUpKSP()
        return self.dR_du_csr.tocoo(), self.dR_df_csr.tocoo()
    
    def solveKSP(self):
    # Ru: the KSP solver (Ax = b) requires that x has zero initial values
        self.initializeU()
        self.setUpKSP()
        ksp = PETSc.KSP().create()
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.setTolerances(rtol=1e-15)
        ksp.setOperators(self.A)
        ksp.setFromOptions()
        ksp.solve(self.b,self.u)
        v2p(self.u1.vector()).ghostUpdate()
        v2p(self.u2.vector()).ghostUpdate()

    def u_ex(self,x):
        xp = as_vector([x[0],x[1]-x[0]])
        return sin(pi*xp[0])*sin(pi*xp[1])*(sin(pi*x[0])**2)*(cos(0.5*pi*x[1])**2)



if __name__ == '__main__':

    fea = set_fea(num_elements=16)
    x1 = SpatialCoordinate(fea.mesh_1)
    x2 = SpatialCoordinate(fea.mesh_2)
    import sympy as sy
    from sympy.printing import ccode
    
    x_ = sy.Symbol('x[0]')
    y_ = sy.Symbol('x[1]')
    u_ = (sy.sin(pi*x_)*sy.sin(pi*(y_-x_))*(sy.sin(pi*x_)*
                        sy.sin(pi*x_))*(sy.cos(0.5*pi*y_)*sy.cos(0.5*pi*y_)))
    f_ = - sy.diff(u_, x_, x_) - sy.diff(u_, y_, y_) + u_
    f = Expression(ccode(f_), degree=2)
    f_ex_1 = interpolate(f,fea.F1)
    f_ex_2 = interpolate(f,fea.F2)
    fea.f1.vector()[:] = f_ex_1.vector().get_local()
    fea.f2.vector()[:] = f_ex_2.vector().get_local()
    
    fea.solveKSP()

    e1 = fea.u1-fea.u_ex(x1)
    print('the L2 error of u1:')
    print(sqrt(assemble((e1**2)*fea.dx)))
    e2 = fea.u2-fea.u_ex(x2)
    print('the L2 error of u2:')
    print(sqrt(assemble((e2**2)*fea.dx)))


    print('The objective:')
    print(fea.objective())
#    print('u1:')
#    print(fea.u1.vector().get_local())
    from matplotlib import pyplot as plt
    plt.figure()
    plot(fea.u1)
    plt.show()
    plot(fea.u2)
    plt.show()



