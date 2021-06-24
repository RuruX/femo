from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import openmdao.api as om 

from set_fea import set_fea
from states_comp import StatesComp 
from constraint_comp import ConstraintComp
from objective_comp import ObjectiveComp



class BeamGroup(om.Group):

	def initialize(self):
		self.options.declare('num_elements', types=int)
		# self.options.declare('b', default=0.1)
		# self.options.declare('L', default=1.)
		self.options.declare('volume', default=1.0)

	def setup(self):
		fea = set_fea(self.options['num_elements'])
		volume = self.options['volume']
		# L = self.options['L']
		# b = self.options['b']

		inputs_comp = om.IndepVarComp()
		inputs_comp.add_output('t', shape=fea.num_elements)
		self.add_subsystem('inputs_comp', inputs_comp)

		comp_1 = StatesComp(fea=fea)
		self.add_subsystem('states_comp', comp_1)

		comp_2 = ObjectiveComp(fea=fea)
		self.add_subsystem('objective_comp', comp_2)

		comp_3 = ConstraintComp(fea=fea)
		self.add_subsystem('constraint_comp', comp_3)

		self.connect('inputs_comp.t', 'states_comp.t')
		self.connect('states_comp.displacements', 'objective_comp.displacements')
		self.connect('inputs_comp.t', 'constraint_comp.t')

		self.add_design_var('inputs_comp.t', lower=1e-2, upper=10.)
		self.add_objective('objective_comp.compliance')
		self.add_constraint('constraint_comp.volume', equals=volume)



if __name__ == '__main__':

	num_el = 16
	prob = om.Problem(model=BeamGroup(num_elements=num_el))

	prob.driver = om.ScipyOptimizeDriver()
	prob.driver.options['optimizer'] = 'SLSQP'
	prob.driver.options['tol'] = 1e-9
	prob.driver.options['disp'] = True
	prob.setup()
	prob.run_driver()
	print(prob['inputs_comp.t'],'\n')

	plt.figure(1)
	plt.plot(np.linspace(0,1,num_el), prob['inputs_comp.t'],'-o')
	# plt.step(np.linspace(0,1,num_el), prob['inputs_comp.t'],where='pre')
	plt.title('Thickness distribution')
	plt.xlabel('Beam length')
	plt.ylabel('Thickness')
	plt.grid(True)
	plt.show()





