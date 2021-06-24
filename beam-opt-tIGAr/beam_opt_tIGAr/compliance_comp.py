from __future__ import division
from six.moves import range

import numpy as np 
import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt
from petsc4py import PETSc

from dolfin import *
import ufl
from set_IGA import set_IGA

class ComplianceComp(om.ExplicitComponent):

	def initialize(self):
		self.options.declare('iga')

	def setup(self):
		self.iga = self.options['iga']
		self.add_input('displacements', shape=self.iga.num_dof)
		self.add_input('thickness', shape=self.iga.num_var)
		self.add_output('compliance')
		self.declare_partials('compliance', 'displacements')
		self.declare_partials('compliance', 'thickness')
		
	def compute(self, inputs, outputs):
		# print('Run compliance compute()-----------------------')
		# self.iga.u.vector()[:] = self.iga.iga2feDoFs(inputs['displacements'])
		self.iga.iga2feDoFs(inputs['displacements'])
		self.iga.t.vector().set_local(inputs['thickness'])
		outputs['compliance'] = assemble(self.compute_compliance(self.iga.u, self.iga.t))

	def compute_partials(self, inputs, partials):
		# print('Run compliance compute_partials()-----------------------')
		# self.iga.u.vector()[:] = self.iga.iga2feDoFs(inputs['displacements'])
		self.iga.iga2feDoFs(inputs['displacements'])
		self.iga.t.vector().set_local(inputs['thickness'])
		C = self.compute_compliance(self.iga.u, self.iga.t)

		dC_du = self.iga.spline.assembleVector(derivative(C, self.iga.u))
		dC_dt = assemble(derivative(C, self.iga.t))

		partials['compliance', 'displacements'] = dC_du.get_local()
		partials['compliance', 'thickness'] = dC_dt.get_local()

	def compute_compliance(self,u, t):
		alpha = Constant(1e-2)
		# If using spline order k=2, don't need regularization term
		return self.iga.rightChar*0.5*u*u*self.iga.spline.ds \
			# + 0.5*alpha*(inner(t,t)*self.iga.spline.dx + (jump(t)**2)*dS)



if __name__ == '__main__':

	num_elements = 10
	iga = set_IGA(num_elements=num_elements)
	prob = Problem()

	comp = ComplianceComp(iga=iga)
	prob.model = comp
	prob.setup()
	prob.run_model()
	prob.model.list_outputs()
	print(prob['displacements'])
	print('check_partials:')
	prob.check_partials(compact_print=True)


