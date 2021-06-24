from __future__ import division
from six.moves import range

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
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
        self.add_input('t', shape=self.fea.num_elements)
        self.add_output('displacements', shape=len(self.fea.W.dofmap().dofs()))
        
        #get indices of partials
        dR_dw_coo_0, dR_dt_coo_0 = self.fea.get_coo()
        
        self.declare_partials('displacements', 't', rows=dR_dt_coo_0.row, cols=dR_dt_coo_0.col)
        self.declare_partials('displacements', 'displacements', rows=dR_dw_coo_0.row, cols=dR_dw_coo_0.col)

    
    def apply_nonlinear(self, inputs, outputs, residuals):
        self.fea.t.vector().set_local(inputs['t'])
        self.fea.w.vector().set_local(outputs['displacements'])
        residuals['displacements'] = assemble(self.fea.pdeRes(self.fea.u,self.fea.v,self.fea.du,self.fea.dv,self.fea.t)).get_local()

    
    def solve_nonlinear(self, inputs, outputs,):
        self.fea.t.vector().set_local(inputs['t'])
        self.fea.w.vector().set_local(outputs['displacements'])

        dR_dw = derivative(self.fea.pdeRes(self.fea.u,self.fea.v,self.fea.du,self.fea.dv,self.fea.t),self.fea.w)
        solve(self.fea.pdeRes(self.fea.u,self.fea.v,self.fea.du,self.fea.dv,self.fea.t)==0, self.fea.w, bcs=[], J=dR_dw)

        outputs['displacements'] = self.fea.w.vector().get_local()

    
    def linearize(self, inputs, outputs, partials):

        dR_dw_coo, dR_dt_coo = self.fea.compute_derivative()
        self.lu = splu(dR_dw_coo.tocsc())

        partials['displacements','t'] = dR_dt_coo.data
        partials['displacements','displacements'] = dR_dw_coo.data
    


    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['displacements'] = self.lu.solve(d_residuals['displacements'])
        else:
            d_residuals['displacements'] = self.lu.solve(d_outputs['displacements'],trans='T')



if __name__ == '__main__':

	num_elements = 20
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

