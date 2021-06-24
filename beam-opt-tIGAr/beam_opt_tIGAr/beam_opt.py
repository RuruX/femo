from __future__ import division
from dolfin import *
import numpy as np 
import matplotlib.pyplot as plt
import openmdao.api as om 

from set_IGA import set_IGA
from states_comp import StatesComp 
from compliance_comp import ComplianceComp
from volume_comp import VolumeComp

class BeamGroup(om.Group):

	def initialize(self):
		self.options.declare('iga')
		self.options.declare('volume', default=.5)

	def setup(self):
		self.iga = self.options['iga']
		volume = self.options['volume']

		inputs_comp = om.IndepVarComp()
		inputs_comp.add_output('t', shape=iga.num_var, val=np.ones(iga.num_var))
		self.add_subsystem('inputs_comp', inputs_comp)

		comp_1 = StatesComp(iga=iga)
		self.add_subsystem('states_comp', comp_1)

		comp_2 = ComplianceComp(iga=iga)
		self.add_subsystem('compliance_comp', comp_2)

		comp_3 = VolumeComp(iga=iga)
		self.add_subsystem('volume_comp', comp_3)

		self.connect('inputs_comp.t', 'states_comp.t')
		self.connect('states_comp.displacements', 'compliance_comp.displacements')
		self.connect('inputs_comp.t', 'volume_comp.t')
		self.connect('inputs_comp.t', 'compliance_comp.thickness')

		self.add_design_var('inputs_comp.t', lower=1e-2, upper=10.)
		self.add_objective('compliance_comp.compliance')
		self.add_constraint('volume_comp.volume', equals=volume)



if __name__ == '__main__':

	num_el = 50
	iga = set_IGA(num_el)
	prob = om.Problem(model=BeamGroup(iga=iga))

	prob.driver = om.ScipyOptimizeDriver()
	prob.driver.options['optimizer'] = 'SLSQP'
	prob.driver.options['tol'] = 1e-9
	prob.driver.options['disp'] = True
	prob.setup()
	prob.run_driver()
	# print(prob['inputs_comp.t'],'\n')

	plt.figure()
	plot(iga.t)
	plt.title('Thickness distribution')
	plt.xlabel('Beam length')
	plt.ylabel('Thickness')
	plt.grid(True)

	plt.show()





