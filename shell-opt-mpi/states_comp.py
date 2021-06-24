from dolfin import *
import numpy as np
from petsc4py import PETSc

import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt

import ufl
from set_fea import *
import timeit


class StatesComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('fea')
        self.options['distributed'] = True

    def setup(self):
        self.fea = self.options['fea']
        self.local_var_size = self.fea.local_dof_f
        self.local_states_size = self.fea.local_dof_u
        self.global_ind_f = self.fea.ind_f
        self.global_ind_u = self.fea.ind_u
        self.add_input('h', shape=self.fea.dof_f)
        self.add_output('displacements', shape=self.local_states_size)
        self.comm = MPI.comm_world
        self.rank = self.comm.Get_rank()


    def apply_nonlinear(self, inputs, outputs, residuals):
#        print("Call apply nonlinear ...")
        update(self.fea.h, inputs['h'][self.global_ind_f])
        update(self.fea.w, outputs['displacements'])

        L = self.fea.R
        A,B = assemble_system(self.fea.dR_du, L, bcs=[])
        residuals['displacements'] = B.get_local()


    def solve_nonlinear(self, inputs, outputs):
        update(self.fea.h, inputs['h'][self.global_ind_f])
        update(self.fea.w, outputs['displacements'])

        start = timeit.default_timer()
        self.fea.solveNonlinear()
        stop = timeit.default_timer()
               
        outputs['displacements'] = self.fea.w.vector().get_local()
        update(self.fea.w, outputs['displacements'])


    def linearize(self, inputs, outputs, partials):
#        print("Call linearize ...")
        update(self.fea.h, inputs['h'][self.global_ind_f])
        update(self.fea.w, outputs['displacements'])

        self.A,_ = assemble_system(self.fea.dR_du, self.fea.R, bcs=[self.fea.bc()])
        self.dRdu,_ = assemble_system(self.fea.dR_du, self.fea.R, bcs=[])
        self.dRdf,_ = assemble_system(self.fea.dR_df, self.fea.R, bcs=[])


    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        update(self.fea.h, inputs['h'][self.global_ind_f])
        update(self.fea.w, outputs['displacements'])

        if mode == 'fwd':
            if 'displacements' in d_residuals:
                if 'displacements' in d_outputs:
                    update(self.fea.du, d_outputs['displacements'])
                    d_residuals['displacements'] += computeMatVecProductFwd(
                            self.dRdu, self.fea.du)
                if 'h' in d_inputs:
                    update(self.fea.df, d_inputs['h'][self.global_ind_f])
                    d_residuals['displacements'] += computeMatVecProductFwd(
                            self.dRdf, self.fea.df)

        if mode == 'rev':
            if 'displacements' in d_residuals:
                update(self.fea.dR, d_residuals['displacements'])
                if 'displacements' in d_outputs:
                    d_outputs['displacements'] += computeMatVecProductBwd(
                            self.dRdu, self.fea.dR)
                if 'h' in d_inputs:
                    d_inputs['h'][self.global_ind_f] += computeMatVecProductBwd(
                            self.dRdf, self.fea.dR)


    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['displacements'] = self.fea.solveLinearFwd(self.A, d_residuals['displacements'])
        else:
            d_residuals['displacements'] = self.fea.solveLinearBwd(self.A, d_outputs['displacements'])
            



if __name__ == '__main__':

    mesh = Mesh()
    filename = "plate1.xdmf"
    file = XDMFFile(mesh.mpi_comm(),filename)
    file.read(mesh)
    
    fea = set_fea(mesh)
    prob = Problem()
    fea.h.interpolate(Constant(0.25))
    comp = StatesComp(fea=fea)
    prob.model = comp
    prob.setup()
    import timeit
    start = timeit.default_timer()
    prob.run_model()
    stop = timeit.default_timer()
#    prob.check_partials(compact_print=True)
#    prob.model.list_outputs()
    print('time for solve_nonlinear: ', stop-start)

