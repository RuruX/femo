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
        self.add_input('uhat', shape=len(self.fea.VHAT.dofmap().dofs()))
        self.add_output('displacements', shape=len(self.fea.V.dofmap().dofs()))

        #get indices of partials
        dR_du_coo_0, dR_df_coo_0 = self.fea.get_coo()

        self.declare_partials('displacements', 'uhat', rows=dR_df_coo_0.row, cols=dR_df_coo_0.col)
        self.declare_partials('displacements', 'displacements', rows=dR_du_coo_0.row, cols=dR_du_coo_0.col)

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.fea.uhat.vector().set_local(inputs['uhat'])
        self.fea.u.vector().set_local(outputs['displacements'])

        A,B = assemble_system(self.fea.dR_du, self.fea.R, bcs=[self.fea.bcu()])
        residuals['displacements'] = B.get_local()


    def solve_nonlinear(self, inputs, outputs):
        self.fea.uhat.vector().set_local(inputs['uhat'])
        self.fea.u.vector().set_local(outputs['displacements'])

        solve(self.fea.R==0,
              self.fea.u,
              bcs=self.fea.bcu(),J=self.fea.dR_du,
              solver_parameters={"newton_solver":{"relative_tolerance":1e-3}})
        outputs['displacements'] = self.fea.u.vector().get_local()


    def linearize(self, inputs, outputs, partials):

#        # [DK] Affects derivatives:
        self.fea.uhat.vector().set_local(inputs['uhat'])
        self.fea.u.vector().set_local(outputs['displacements'])

        dR_du_coo, dR_df_coo = self.fea.compute_derivative()

        # [DK] This SciPy direct solve is clearly the performance bottleneck;
        # we need to find a way to replace this with an efficient
        # iterative Krylov solver using PETSc.  SNOpt will require fewer
        # calls to it, but it will still be very inefficient.

        self.lu = splu(dR_du_coo.tocsc())
    
        partials['displacements','uhat'] = dR_df_coo.data
        partials['displacements','displacements'] = dR_du_coo.data


    def solve_linear(self, d_outputs, d_residuals, mode):

#        from petsc4py import PETSc
#
#        dR_du_sparse = self.dR_du_sparse
#
#        # solve dR_du * du = dR
#        # A - dR_du
#        # dR - d_residuals['displacements']
#        # du - d_outputs['displacements']
#
#
#        ksp = PETSc.KSP().create()
#        ksp.setType(PETSc.KSP.Type.GMRES)
#        ksp.setTolerances(rtol=1e-15)
#        ksp.setOperators(dR_du_sparse)
#        ksp.setFromOptions()
#
#        size = len(self.fea.V.dofmap().dofs())
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
#        print("Calling ksp...")
#        if mode == 'fwd':
#            ksp.solve(dR,du)
#            d_outputs['displacements'] = du.getValues(range(size))
#        else:
#            ksp.solveTranspose(du,dR)
#            d_residuals['displacements'] = dR.getValues(range(size))


        print("Calling splu...")
        if mode == 'fwd':
            d_outputs['displacements'] = self.lu.solve(d_residuals['displacements'])
        else:
            d_residuals['displacements'] = self.lu.solve(d_outputs['displacements'],trans='T')



if __name__ == '__main__':

    num_elements = 3
    fea = set_fea(num_elements=num_elements)
    prob = Problem()



    comp = StatesComp(fea=fea)
    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['uhat'])
    print('check_partials:')
    prob.check_partials(compact_print=True)

#v = PETSc.Vec().create()
#v.setSizes(num_el)
#v.setType('seq')
#v.setValues(range(x), numpy_araray)
#v.setUp()
#v.assemble()

#v.getValues(range(10))


