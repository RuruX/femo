import numpy as np
from dolfin import *
import ufl
from scipy.sparse import csr_matrix, vstack, hstack, block_diag
from petsc4py import PETSc

def m2p(A):
    return as_backend_type(A).mat()

def v2p(v):
    return as_backend_type(v).vec()

class set_fea(object):

    def __init__(self, num_elements):
        
        self.num_elements = num_elements

        k = 1
        self.mesh = UnitSquareMesh(self.num_elements,self.num_elements)
        UE = FiniteElement('CG', self.mesh.ufl_cell(), k)
        FE = FiniteElement('DG', self.mesh.ufl_cell(), 0)

        self.V = FunctionSpace(self.mesh, UE)
        self.F = FunctionSpace(self.mesh, FE)
        self.u = Function(self.V)
        self.v = TestFunction(self.V)
        self.f = Function(self.F)
        self.n = FacetNormal(self.mesh)
        self.R = self.pdeRes(self.u, self.v, self.f)
        self.dR_du = derivative(self.R,self.u)
        self.dR_df = derivative(self.R,self.f)
        
    def updateF(self, f_new):
        self.f.vector().set_local(f_new[:len(self.F.dofmap().dofs())])
        
    def updateU(self, u_new):
        self.u.vector().set_local(u_new[:len(self.V.dofmap().dofs())])
        
    def initializeU(self):
        self.updateU(np.zeros(len(self.V.dofmap().dofs())))
    
    def pdeRes(self,u,v,f):
       return (inner(grad(u),grad(v))-f*v)*dx
    
    def bc(self):
        bc = DirichletBC(self.V, Constant(DOLFIN_EPS), "on_boundary")
        return bc
    
    def objective(self,u,f):
        x = SpatialCoordinate(self.mesh)
        w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
        d = 1/(2*pi**2)
        d = Expression("d*w", d=d, w=w, degree=3)
        alpha = Constant(1e-6)
        return (Constant(0.5)*inner(u-d, u-d))*dx + alpha/2*f**2*dx

    def compute_derivative(self):
        A = assemble(derivative(self.R, self.u))
        dR_du_sparse = as_backend_type(A).mat()
        A1 =assemble(derivative(self.R, self.f))
        dR_df_sparse = as_backend_type(A1).mat()
        dR_du_csr = csr_matrix(dR_du_sparse.getValuesCSR()[::-1], shape=dR_du_sparse.size)
        dR_df_csr = csr_matrix(dR_df_sparse.getValuesCSR()[::-1], shape=dR_df_sparse.size)
        return dR_du_csr.tocoo(), dR_df_csr.tocoo()

    def get_coo(self):
        self.u.interpolate(Constant(1.0))
        self.f.interpolate(Constant(1.0))
        dR_du_sparse_0 = as_backend_type(assemble(self.dR_du)).mat()
        dR_df_sparse_0 = as_backend_type(assemble(self.dR_df)).mat()
        dR_du_csr_0 = csr_matrix(dR_du_sparse_0.getValuesCSR()[::-1], shape=dR_du_sparse_0.size)
        dR_df_csr_0 = csr_matrix(dR_df_sparse_0.getValuesCSR()[::-1], shape=dR_df_sparse_0.size)

        return dR_du_csr_0.tocoo(), dR_df_csr_0.tocoo()

    def setUpKSP(self):
        L = -self.R
        A,B = assemble_system(self.dR_du, L, bcs=[self.bc()])
        self.A = m2p(A)
        self.b = v2p(B)
        self.u_new = PETSc.Vec()
        self.u_new.createNest([v2p(self.u.vector())])
        self.u_new.setUp()

    def solveKSP(self):
        """
        Ru: the KSP solver (Ax = b) requires that x has zero initial values
        """
        self.initializeU()
        self.setUpKSP()
        ksp = PETSc.KSP().create()
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.setTolerances(rtol=1e-15)
        ksp.setOperators(self.A)
        ksp.setFromOptions()
        ksp.solve(self.b,self.u_new)
        v2p(self.u.vector()).ghostUpdate()
        
        
if __name__ == '__main__':

    fea = set_fea(num_elements=5)
    fea.f.interpolate(Constant(1.0))
    fea.u.interpolate(Constant(1.0))
    
    solve(fea.R==0, fea.u, bcs=fea.bc(), J=fea.dR_du)
    print('*'*20, fea.u.vector().get_local(), '*'*20)




