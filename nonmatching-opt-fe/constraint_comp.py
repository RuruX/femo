from __future__ import division
from six.moves import range

import numpy as np 
import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt

from dolfin import *
import ufl
from set_fea import set_fea

class ConstraintComp(om.ExplicitComponent):

	def initialize(self):
		self.options.declare('fea')

	def setup(self):
		self.fea = self.options['fea']
		self.lam = 1.

		self.add_input('t', shape=self.fea.num_elements)
		self.add_output('volume')

		self.declare_partials('volume', 't')

	def compute(self, inputs, outputs):
		self.fea.t.vector().set_local(inputs['t'])
		outputs['volume'] = assemble(self.fea.t*dx)
		# outputs['volume'] = assemble(self.lam*(self.fea.t)*dx)
		# print(outputs['volume'])

	def compute_partials(self, inputs, partials):
		# self.fea = self.options['fea']
		self.fea.t.vector().set_local(inputs['t'])
		dV_dt = assemble(derivative(self.fea.t*dx, self.fea.t))
		# dV_dt = assemble(derivative(self.lam*(self.fea.t)*dx, self.fea.t))
		dV_dt_array = dV_dt.get_local()
		partials['volume', 't'] = dV_dt_array



if __name__ == '__main__':

	num_elements = 5
	fea = set_fea(num_elements=num_elements)
	prob = Problem()

	comp = ConstraintComp(fea=fea)
	prob.model = comp
	prob.setup()
	prob.run_model()
	prob.model.list_outputs()
	print(prob['volume'])
	print('check_partials:')
	prob.check_partials(compact_print=True)

