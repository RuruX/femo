from __future__ import division
from six.moves import range

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, hstack
from scipy.sparse.linalg import splu
from petsc4py import PETSc

import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt

from dolfin import *
import ufl
from set_fea import set_fea


class StatesComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('fea')
    
    def setup(self):
        self.fea = self.options['fea']
        self.num_elements = self.fea.num_elements
        self.add_input('f',
            shape=len(self.fea.F1.dofmap().dofs())+len(self.fea.F2.dofmap().dofs()))
        self.add_output('displacements',
            shape=len(self.fea.V1.dofmap().dofs())+len(self.fea.V2.dofmap().dofs()))
        #get indices of partials
        dR_du_coo_0 = self.fea.dR_du_coo_0
        dR_df_coo_0 = self.fea.dR_df_coo_0
        self.declare_partials('displacements', 'f',
                    rows=dR_df_coo_0.row, cols=dR_df_coo_0.col)
        self.declare_partials('displacements', 'displacements',
                    rows=dR_du_coo_0.row, cols=dR_du_coo_0.col)
                    
    def apply_nonlinear(self, inputs, outputs, residuals):
        f = inputs['f']
        u = outputs['displacements']
        self.fea.updateF(f)
        self.fea.updateU(u)
        self.fea.setUpKSP()
        R1 = self.fea.R1.getArray()
        R2 = self.fea.R2.getArray()
        print("calling apply_nonlinear ...")
        residuals['displacements'] = np.append([R1],[R2])


    def solve_nonlinear(self, inputs, outputs,):
        f = inputs['f']
        u = outputs['displacements']
        self.fea.updateF(f)
        self.fea.solveKSP()
        print("calling solve_nonlinear ...")
        outputs['displacements'] = self.fea.u.getArray()
        u_new = outputs['displacements']
        self.fea.updateU(u_new)

    def linearize(self, inputs, outputs, partials):
        f = inputs['f']
        u = outputs['displacements']
        self.fea.updateF(f)
        self.fea.updateU(u)
        dR_du_coo, dR_df_coo = self.fea.compute_derivative()
        self.lu = splu(dR_du_coo.tocsc())
        print("calling linearize ...")
        partials['displacements','f'] = dR_df_coo.data
        partials['displacements','displacements'] = dR_du_coo.data

    def solve_linear(self, d_outputs, d_residuals, mode):

#        from petsc4py import PETSc
#        dR_du_sparse = self.fea.A
#        # solve dR_du * du = dR
#        # A - dR_du
#        # dR - d_residuals['displacements']
#        # du - d_outputs['displacements']
#
#        ksp = PETSc.KSP().create()
#        ksp.setType(PETSc.KSP.Type.GMRES)
#        ksp.setTolerances(rtol=1e-15)
#        ksp.setOperators(dR_du_sparse)
#        ksp.setFromOptions()
#
#        size = len(self.fea.V1.dofmap().dofs()) + len(self.fea.V2.dofmap().dofs())
#
#        dR = PETSc.Vec().create()
#        dR.setSizes(size)
#        dR.setType('seq')
#        dR.setValues(range(size), d_residuals['displacements'])
#        dR.setUp()
#
#        du = PETSc.Vec().create()
#        du.setSizes(size)
#        du.setType('seq')
#        du.setValues(range(size), d_outputs['displacements'])
#        du.setUp()
#
#        print("calling ksp...")
#
#        if mode == 'fwd':
#            ksp.solve(dR,du)
#            d_outputs['displacements'] = du.getValues(range(size))
#        else:
#            ksp.solveTranspose(du,dR)
#            d_residuals['displacements'] = dR.getValues(range(size))
#

        print("calling solve linear ...")
        if mode == 'fwd':
            d_outputs['displacements'] = self.lu.solve(d_residuals['displacements'])
        else:
            d_residuals['displacements'] = self.lu.solve(d_outputs['displacements'],trans='T')



if __name__ == '__main__':

	num_elements = 5
	fea = set_fea(num_elements=num_elements)
	prob = Problem()

	comp = StatesComp(fea=fea)
	prob.model = comp
	prob.setup()
	prob.run_model()
	prob.model.list_outputs()
	print(prob['displacements'])
	print('check_partials:')
	prob.check_partials(compact_print=True)

