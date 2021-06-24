import numpy as np 
from dolfin import *
import ufl
from scipy.sparse import csr_matrix

class set_fea(object):

    def __init__(self, num_elements):
        self.num_elements = num_elements
        self.num_nodes = self.num_elements+1

        self.mesh = UnitIntervalMesh(self.num_elements)
        self.k = 1 # polynomial order
        self.n = FacetNormal(self.mesh)

        self.UE = FiniteElement('CG', self.mesh.ufl_cell(), self.k)
        self.VE = FiniteElement('CG', self.mesh.ufl_cell(), self.k)
        self.TE = FiniteElement('DG', self.mesh.ufl_cell(), 0)
        self.FE = FiniteElement('DG', self.mesh.ufl_cell(), 0)

        self.W = FunctionSpace(self.mesh, MixedElement([self.UE, self.VE]))
        self.w = Function(self.W)
        self.u, self.v = split(self.w)
        self.du, self.dv = split(TestFunction(self.W))

        self.T = FunctionSpace(self.mesh, self.TE)
        self.t = Function(self.T) # beam thickness
        # self.EI = self.t**3

        self.x = SpatialCoordinate(self.mesh)
        self.leftChar = 1.0 - self.x[0]
        self.rightChar = self.x[0]

        self.F = FunctionSpace(self.mesh, self.FE)
        self.force_vector = self.rightChar*Constant(-1.)

    def pdeRes(self, u, v, du, dv, t):
        alpha = Constant(1e-2)
        EI = t**3
        return inner(grad(u),grad(du))*dx + inner(v/EI,du)*dx \
            + inner(grad(v),grad(dv))*dx + inner(self.force_vector,dv)*dx \
            + self.leftChar*u*inner(self.n,grad(dv))*ds \
            + self.rightChar*v*inner(self.n,grad(du))*ds \
            - self.leftChar*dot(grad(v),self.n)*dv*ds \
            - self.rightChar*dot(grad(u),self.n)*du*ds

    def compute_derivative(self): #, inputs_t, outputs_w):

        dR_dw = derivative(self.pdeRes(self.u, self.v, self.du, self.dv, self.t), self.w)
        dR_dw_matrix = assemble(dR_dw)
        
        dR_dt = derivative(self.pdeRes(self.u, self.v, self.du, self.dv, self.t), self.t)
        dR_dt_matrix = assemble(dR_dt)

        dR_dw_sparse = as_backend_type(dR_dw_matrix).mat()
        dR_dt_sparse = as_backend_type(dR_dt_matrix).mat()

        dR_dw_csr = csr_matrix(dR_dw_sparse.getValuesCSR()[::-1], shape=dR_dw_sparse.size)
        dR_dt_csr = csr_matrix(dR_dt_sparse.getValuesCSR()[::-1], shape=dR_dt_sparse.size)

        return dR_dw_csr.tocoo(), dR_dt_csr.tocoo()

    def get_coo(self):
        wh = Function(self.W)

        uh, vh = split(wh)
        duh, dvh = split(TestFunction(self.W))
        th = Function(self.T)
        wh.interpolate(Constant((1,1)))
        th.interpolate(Constant(1))
        dR_dw_0 = derivative(self.pdeRes(uh, vh, duh, dvh, th), wh)
        dR_dw_matrix_0 = assemble(dR_dw_0)

        dR_dt_0 = derivative(self.pdeRes(uh, vh, duh, dvh, th), th)
        dR_dt_matrix_0 = assemble(dR_dt_0)

        dR_dw_sparse_0 = as_backend_type(dR_dw_matrix_0).mat()
        dR_dt_sparse_0 = as_backend_type(dR_dt_matrix_0).mat()

        dR_dw_csr_0 = csr_matrix(dR_dw_sparse_0.getValuesCSR()[::-1], shape=dR_dw_sparse_0.size)
        dR_dt_csr_0 = csr_matrix(dR_dt_sparse_0.getValuesCSR()[::-1], shape=dR_dt_sparse_0.size)

        return dR_dw_csr_0.tocoo(), dR_dt_csr_0.tocoo()


if __name__ == '__main__':

    fea = set_fea(num_elements=5)
    dR_dw_size, dR_dt_size = fea.get_coo()
    print(dR_dt_size, '*'*20)
    print(dR_dw_size, '*'*20)






