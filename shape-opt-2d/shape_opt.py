from __future__ import division
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from dolfin import *

from set_fea import set_fea
from states_comp import StatesComp
from objective_comp import ObjectiveComp
from constraint_comp import ConstraintComp

class Shape2DGroup(om.Group):

    def initialize(self):
        self.options.declare('num_elements', types=int)

    def setup(self):
        fea = set_fea(self.options['num_elements'])

        inputs_comp = om.IndepVarComp()

        inputs_comp.add_output('uhat', shape=len(fea.VHAT.dofmap().dofs()))
        self.add_subsystem('inputs_comp', inputs_comp)

        comp_1 = StatesComp(fea=fea)
        self.add_subsystem('states_comp', comp_1)

        comp_2 = ObjectiveComp(fea=fea)
        self.add_subsystem('objective_comp', comp_2)

        comp_3 = ConstraintComp(fea=fea)
        self.add_subsystem('constraint_comp', comp_3)

        self.connect('inputs_comp.uhat','states_comp.uhat')
        self.connect('states_comp.displacements', 'objective_comp.displacements')
        self.connect('inputs_comp.uhat', 'objective_comp.uhat')
        self.connect('inputs_comp.uhat', 'constraint_comp.uhat')

        # with bcs applied on uhat
        lower = np.empty(len(fea.VHAT.dofmap().dofs()))
        lower[:] = -np.inf
        uhatbc = np.where(fea.applyBCuhat()==0)
        lower[uhatbc] = 0.0
        upper = -lower

        self.add_design_var('inputs_comp.uhat', lower=lower, upper=upper)


        self.add_objective('objective_comp.objective')
        self.add_constraint('constraint_comp.constraint', equals=0.0)



if __name__ == '__main__':

    # [DK] The SciPy solve takes too long with num_el = 10.
    #num_el = 10
    num_el = 3

    prob = om.Problem(model=Shape2DGroup(num_elements=num_el))
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-7
    prob.driver.options['disp'] = True
#    prob.driver = om.pyOptSparseDriver()
#    prob.driver.options['optimizer']='SNOPT'
    # [DK] Needed to shrink tolerance for trivial objective function


    prob.setup()
    prob.run_model()
    prob.run_driver()

    
    print(prob['inputs_comp.uhat'],'uhat \n')
    print('The value of the objective function is: \n')
    print(prob['objective_comp.objective'],'\n')
    fea = set_fea(num_el)
    fea.uhat.vector().set_local(prob['inputs_comp.uhat'])

    ALE.move(fea.mesh, fea.uhat)
    from matplotlib import pyplot as plt
    plot(fea.mesh)
    plt.show()

