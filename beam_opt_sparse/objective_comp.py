from __future__ import division
from six.moves import range

import numpy as np 
import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt

from dolfin import *
import ufl
from set_fea import set_fea

class ObjectiveComp(om.ExplicitComponent):

	def initialize(self):
		self.options.declare('fea')

	def setup(self):
		self.fea = self.options['fea']
		self.force_vector = project(self.fea.force_vector, self.fea.F).vector().get_local()
		# self.force_vector = self.fea.force_vector

		self.add_input('displacements', shape=len(self.fea.W.dofmap().dofs()))
		self.add_output('compliance')
		self.declare_partials('compliance', 'displacements')

		self.W0_dof_indices = self.fea.W.dofmap().dofs()
		# W0_dof_values = self.fea.w.vector()[W0_dof_indices]
		# self.fea.w.vector().set_local(inputs['displacements'])
		

	def compute(self, inputs, outputs):
		# self.fea = self.options['fea']
		# self.fea.w.vector().set_local(inputs['displacements'])
		self.fea.w.vector()[self.W0_dof_indices] = inputs['displacements']
		self.u,self.v = split(self.fea.w)
		outputs['compliance'] = assemble(self.compute_compliance(self.u, self.fea.t))

	def compute_partials(self, inputs, partials):
		# self.fea = self.options['fea']
		# self.fea.w.vector().set_local(inputs['displacements'])
		self.fea.w.vector()[self.W0_dof_indices] = inputs['displacements']
		self.u,self.v = split(self.fea.w)
		dC_dw = assemble(derivative(self.compute_compliance(self.u, self.fea.t), self.fea.w))
		dC_dw_array = dC_dw.get_local()
		partials['compliance', 'displacements'] = dC_dw_array

	def compute_compliance(self,u, t):
		# self.fea = self.options['fea']
		alpha = Constant(1e-2)
		return self.fea.rightChar*0.5*u*u*ds 


if __name__ == '__main__':

	num_elements = 5
	fea = set_fea(num_elements=num_elements)
	prob = Problem()

	comp = ObjectiveComp(fea=fea)
	prob.model = comp
	prob.setup()
	prob.run_model()
	prob.model.list_outputs()
	print(prob['displacements'])
	print('check_partials:')
	prob.check_partials(compact_print=True)


