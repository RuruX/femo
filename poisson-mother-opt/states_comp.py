from __future__ import division
from dolfin import *

from six.moves import range

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from petsc4py import PETSc

import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt
from openmdao.utils.mpi import MPI

import ufl
from set_fea import set_fea


class StatesComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('fea')
        self.options['distributed'] = True
        
    def setup(self):
        self.fea = self.options['fea']
        self.num_elements = self.fea.num_elements
        self.add_input('f', shape=len(self.fea.F.dofmap().dofs()))
        self.add_output('displacements', shape=len(self.fea.V.dofmap().dofs()))
        
        #get indices of partials
        dR_du_coo_0, dR_df_coo_0 = self.fea.get_coo()
        
        self.declare_partials('displacements', 'f', rows=dR_df_coo_0.row, cols=dR_df_coo_0.col)
        self.declare_partials('displacements', 'displacements', rows=dR_du_coo_0.row, cols=dR_du_coo_0.col)

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.fea.f.vector().set_local(inputs['f'])
        self.fea.u.vector().set_local(outputs['displacements'])
        A,B = assemble_system(self.fea.dR_du, self.fea.R, bcs=[])
        residuals['displacements'] = B.get_local()
        
    def solve_nonlinear(self, inputs, outputs,):
        self.fea.f.vector().set_local(inputs['f'])
        self.fea.u.vector().set_local(outputs['displacements'])
        
#        solve(self.fea.R==0,
#              self.fea.u,
#              bcs=self.fea.bc(),J=self.fea.dR_du,
#              solver_parameters={"newton_solver":{"relative_tolerance":1e-3}, "linear_solver": "lu"})
#        

        problem = NonlinearVariationalProblem(self.fea.R, self.fea.u, self.fea.bc(), J=self.fea.dR_du)
        solver  = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm['newton_solver']['relative_tolerance'] = 1E-3
        prm['newton_solver']['linear_solver'] = 'mumps'
#        prm['newton_solver']['linear_solver'] = 'gmres'
#        prm['newton_solver']['preconditioner'] = 'hypre_amg'
        solver.solve()


        outputs['displacements'] = self.fea.u.vector().get_local()
        self.fea.updateU(outputs['displacements'])


    def linearize(self, inputs, outputs, partials):

        self.fea.f.vector().set_local(inputs['f'])
        self.fea.u.vector().set_local(outputs['displacements'])
        
        dR_du_coo, dR_df_coo = self.fea.compute_derivative()
#        self.lu = splu(dR_du_coo.tocsc())
        partials['displacements','f'] = dR_df_coo.data
        partials['displacements','displacements'] = dR_du_coo.data
  
  
    def solve_linear(self, d_outputs, d_residuals, mode):
        L = -self.fea.R
        A,B = assemble_system(self.fea.dR_du, L, bcs=[self.fea.bc()])
        dR_du_sparse = as_backend_type(A).mat()
        dR_du_csr = csr_matrix(dR_du_sparse.getValuesCSR()[::-1], shape=dR_du_sparse.size)
        dR_du_coo = dR_du_csr.tocoo()
        self.lu = splu(dR_du_coo.tocsc())
        
        if mode == 'fwd':
            d_outputs['displacements'] = self.lu.solve(d_residuals['displacements'])
        else:
            d_residuals['displacements'] = self.lu.solve(d_outputs['displacements'],trans='T')

if __name__ == '__main__':

    num_elements = 4
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

