from __future__ import division
from six.moves import range

import numpy as np 
import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt

from dolfin import *
import ufl
from set_IGA import set_IGA

class VolumeComp(om.ExplicitComponent):

	def initialize(self):
		self.options.declare('iga')

	def setup(self):
		self.iga = self.options['iga']

		self.add_input('t', shape=self.iga.num_var)
		self.add_output('volume')

		self.declare_partials('volume', 't')

	def compute(self, inputs, outputs):
		# print('Run volume compute()-----------------------')
		self.iga.t.vector().set_local(inputs['t'])
		vol = self.compute_volume(self.iga.t)
		outputs['volume'] = assemble(vol)

	def compute_partials(self, inputs, partials):
		# print('Run volume compute_partials()-----------------------')
		self.iga.t.vector().set_local(inputs['t'])
		vol = self.compute_volume(self.iga.t)
		dV_dt = assemble(derivative(vol, self.iga.t)).get_local()
		partials['volume', 't'] = dV_dt

	def compute_volume(self, t):
		return t*self.iga.spline.dx


if __name__ == '__main__':

	num_elements = 10
	iga = set_IGA(num_elements=num_elements)
	prob = Problem()

	comp = VolumeComp(iga=iga)
	prob.model = comp
	prob.setup()
	prob.run_model()
	prob.model.list_outputs()
	print(prob['volume'])
	print('check_partials:')
	prob.check_partials(compact_print=True)
















