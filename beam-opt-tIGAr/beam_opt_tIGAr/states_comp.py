from __future__ import division
from six.moves import range

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from petsc4py import PETSc
from scipy.sparse import find
from scipy.sparse import coo_matrix

import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt

from dolfin import *
import ufl
from tIGAr import *
from set_IGA import set_IGA


class StatesComp(om.ImplicitComponent):

	def initialize(self):
		self.options.declare('iga')

	def setup(self):
		self.iga = self.options['iga']
		self.add_input('t', shape=self.iga.num_var)
		self.add_output('displacements', shape=self.iga.num_dof)

		# self.iga.u.interpolate(Constant(1.0))
		# self.iga.t.interpolate(Constant(1.0))
		dR_du_coo, dR_dt_coo = self.iga.compute_derivative(self.iga.u, self.iga.v, self.iga.t)

		self.declare_partials('displacements', 't', rows=dR_dt_coo.row, cols=dR_dt_coo.col)
		self.declare_partials('displacements', 'displacements', rows=dR_du_coo.row, cols=dR_du_coo.col)

	def apply_nonlinear(self, inputs, outputs, residuals):
		# print('Run apply_nonlinear()-----------------------')
		self.iga.t.vector().set_local(inputs['t'])
		# self.iga.u.vector()[:] = self.iga.iga2feDoFs(outputs['displacements'])
		self.iga.iga2feDoFs(outputs['displacements'])
		pde_res = self.iga.pdeRes(self.iga.u,self.iga.v,self.iga.t)
		
		residuals['displacements'] = self.iga.spline.assembleVector(pde_res).get_local()

	def solve_nonlinear(self, inputs, outputs,):
		# print('Run solve_nonlinear()-----------------------')
		self.iga.t.vector().set_local(inputs['t'])
		# self.iga.u.vector()[:] = self.iga.iga2feDoFs(outputs['displacements'])
		self.iga.iga2feDoFs(outputs['displacements'])
		pde_res = self.iga.pdeRes(self.iga.u,self.iga.v,self.iga.t)
		dR_du = derivative(pde_res,self.iga.u)

		u_temp = Function(self.iga.spline.V)
		u_petsc = multTranspose(self.iga.spline.M, u_temp.vector())
		self.iga.spline.setSolverOptions(maxIters=30)
		self.iga.spline.solveNonlinearVariationalProblem(pde_res,J=dR_du,u=self.iga.u,igaDoFs=u_petsc)

		outputs['displacements'] = u_petsc.get_local()

	def linearize(self, inputs, outputs, partials):
		# print('Run linearize()-----------------------')
		self.iga.t.vector().set_local(inputs['t'])
		# self.iga.u.vector()[:] = self.iga.iga2feDoFs(outputs['displacements'])
		self.iga.iga2feDoFs(outputs['displacements'])
		dR_du_coo, dR_dt_coo = self.iga.compute_derivative(self.iga.u, self.iga.v, self.iga.t)
		self.lu = splu(dR_du_coo.tocsc())

		partials['displacements', 't'] = dR_dt_coo.data
		partials['displacements', 'displacements'] = dR_du_coo.data

	def solve_linear(self, d_outputs, d_residuals, mode):
		# print('Run solve_linear()-----------------------')
		if mode == 'fwd':
			d_outputs['displacements'] = self.lu.solve(d_residuals['displacements'])
		else:
			d_residuals['displacements'] = self.lu.solve(d_outputs['displacements'],trans='T')


if __name__ == '__main__':

	num_elements = 10
	iga = set_IGA(num_elements=num_elements)
	prob = Problem()

	comp = StatesComp(iga=iga)
	prob.model = comp
	prob.setup()
	prob.run_model()
	prob.model.list_outputs()
	print(prob['displacements'])
	print('check_partials:')
	prob.check_partials(compact_print=True)









