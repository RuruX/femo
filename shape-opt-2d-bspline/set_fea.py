import numpy as np
from dolfin import *
from ufl import shape, Max
from scipy.sparse import csr_matrix


class set_fea(object):

    def __init__(self, num_elements):

        self.num_elements = num_elements
        self.length = 1.0
        N_fac = 8
        self.height = self.length/N_fac
        d = 2
        k = 1
        self.mesh = RectangleMesh(Point(0,0),Point(self.length,self.height),N_fac*self.num_elements,self.num_elements)

        cell = self.mesh.ufl_cell()

        VE = VectorElement("CG",cell,k)
        LE = FiniteElement("R",cell,0)


        self.I = Identity(d)
        x = SpatialCoordinate(self.mesh)
        self.dX = dx(metadata={"quadrature_degree":2*k})
        # Traction on top of beam:
        self.leftChar = conditional(lt(x[0],DOLFIN_EPS),1.0,Constant(0.0))
        topChar = conditional(gt(x[1],Constant(self.height-DOLFIN_EPS)),1.0,Constant(0.0))
        self.h = Constant((0,-1))*topChar

        self.V = FunctionSpace(self.mesh, VE)
        self.VHAT = FunctionSpace(self.mesh, VE)
        self.L = FunctionSpace(self.mesh, LE)

        self.u = Function(self.V)
        self.v = TestFunction(self.V)
        self.uhat = Function(self.VHAT)
        self.lam = Function(self.L)

        self.R = self.pdeRes(self.u,self.v,self.uhat)
        self.dR_du = derivative(self.R, self.u)
        self.dR_df = derivative(self.R, self.uhat)

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
        gradu = self.gradx(u,uhat)
        gradv = self.gradx(v,uhat)
        sigma = self.Ctimes(epsu)
        sigmav = self.Ctimes(sym(gradv))
        n = FacetNormal(self.mesh)
        F = self.I + grad(uhat)
        N = dot(inv(F.T),n)
            
        return inner(sigma,gradv)*self.J(uhat)*self.dX - dot(self.h,v)*ds

        # [DK] Add penalty to enforce BC:  Note that, for a consistent
        #      Nitsche method, would need to transform ds using Nanson's
        #      formula (due to the deformation by uhat).  With the current
        #      penalty formulation, the result is surprisingly sensitivie
        #      to the choice of the penalty constant.


#        penalty = Constant(1e21)
#        bc_pen_term = penalty*dot(u,v)*self.leftChar*ds
#        return inner(sigma,gradv)*self.J(uhat)*self.dX - dot(self.h,v)*ds + bc_pen_term

#        return inner(sigma,gradv)*self.J(uhat)*self.dX - dot(self.h,v)*ds \
#            - self.leftChar*dot(dot(sigma,N),v)*self.J(uhat)*ds \
#            + self.leftChar*dot(dot(sigmav,N),u)*self.J(uhat)*ds


    def bcu(self):
        x = SpatialCoordinate(self.mesh)

        leftChar = conditional(lt(x[0],DOLFIN_EPS),1.0,Constant(0.0))
        rightChar = conditional(gt(x[0],self.length-DOLFIN_EPS),1.0,Constant(0.0))
        leftStr = "x[0] < DOLFIN_EPS"
        bcu = DirichletBC(self.V,Constant((0,0)),leftStr)
        return bcu

    def objective(self,u,uhat):
        alpha = Constant(4e-9)
        return Constant(1.0)*dot(self.h,u)*ds \
            + 0.5*alpha*(inner(self.Ctimes(sym(grad(uhat))),grad(uhat)))*self.dX

    def constraint(self,uhat,lam):
        return lam*(self.J(uhat)-Constant(1.0))*self.dX

    def compute_derivative(self):

        # Strong enforcement of bcs on u
        L = -self.R
        A,B = assemble_system(self.dR_du, L, bcs=[self.bcu()])
        dR_du_sparse = as_backend_type(A).mat()
        A1,B1 = assemble_system(self.dR_df, L, bcs=[self.bcu()])
        dR_df_sparse = as_backend_type(A1).mat()

        dR_du_csr = csr_matrix(dR_du_sparse.getValuesCSR()[::-1], shape=dR_du_sparse.size)
        dR_df_csr = csr_matrix(dR_df_sparse.getValuesCSR()[::-1], shape=dR_df_sparse.size)
        return dR_du_csr.tocoo(), dR_df_csr.tocoo()

    def get_coo(self):

        dR_du_sparse_0 = as_backend_type(assemble(self.dR_du)).mat()
        dR_df_sparse_0 = as_backend_type(assemble(self.dR_df)).mat()
        dR_du_csr_0 = csr_matrix(dR_du_sparse_0.getValuesCSR()[::-1], shape=dR_du_sparse_0.size)
        dR_df_csr_0 = csr_matrix(dR_df_sparse_0.getValuesCSR()[::-1], shape=dR_df_sparse_0.size)

        return dR_du_csr_0.tocoo(), dR_df_csr_0.tocoo()

    def applyBCuhat(self):

        self.uhat.interpolate(Constant((1.0,1.0)))
        bcuhat0 = DirichletBC(self.VHAT.sub(0),Constant(0),"true","pointwise")
        bcuhat1 = DirichletBC(self.VHAT.sub(1),Constant(0),"x[1] > "
                              +str(self.height)+"-DOLFIN_EPS")
        bcuhat0.apply(self.uhat.vector())
        bcuhat1.apply(self.uhat.vector())
        return self.uhat.vector().get_local()




if __name__ == '__main__':

    fea = set_fea(num_elements=3)
    print(fea.V)
    fea.u.interpolate(Constant((1.0,1.0)))
    fea.bcu().apply(fea.u.vector())
    print(fea.u.vector().get_local())
#    solve(fea.pdeRes(fea.u,fea.v,fea.uhat)==0, fea.u, bcs=fea.bc())
#    print(assemble(fea.objective(fea.u,fea.uhat)))







