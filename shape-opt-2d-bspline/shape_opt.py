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
from bspline_comp import BsplineComp, get_bspline_mtx
from uhat_comp import uhatComp

class Shape2DGroup(om.Group):

    def initialize(self):
        self.options.declare('num_elements', types=int)

    def setup(self):
        fea = set_fea(self.options['num_elements'])
        N_fac = 8
        num_pt = self.options['num_elements']*N_fac+1
        num_cp = 5
        
        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('control_points', shape=num_cp)
        self.add_subsystem('inputs_comp', inputs_comp)

        comp_0 = BsplineComp(
                           num_pt=num_pt,
                           num_cp=num_cp,
                           jac=get_bspline_mtx(num_cp,num_pt),
                           in_name='control_points',
                           out_name='points',
                           )

        
        self.add_subsystem('bspline_comp', comp_0)
    
        comp = uhatComp(fea=fea)
        self.add_subsystem('uhat_comp', comp)

        comp_1 = StatesComp(fea=fea)
        self.add_subsystem('states_comp', comp_1)

        comp_2 = ObjectiveComp(fea=fea)
        self.add_subsystem('objective_comp', comp_2)

        comp_3 = ConstraintComp(fea=fea)
        self.add_subsystem('constraint_comp', comp_3)

        self.connect('inputs_comp.control_points',
                     'bspline_comp.control_points')
        self.connect('bspline_comp.points','uhat_comp.points')
        self.connect('uhat_comp.uhat','states_comp.uhat')
        self.connect('states_comp.displacements',
                     'objective_comp.displacements')
        self.connect('uhat_comp.uhat', 'objective_comp.uhat')
        self.connect('uhat_comp.uhat', 'constraint_comp.uhat')

        thickness = fea.length / N_fac
        self.add_design_var('inputs_comp.control_points', upper=thickness * 0.9)
        self.add_objective('objective_comp.objective')
        self.add_constraint('constraint_comp.constraint', equals=0.0)



if __name__ == '__main__':

    # tested up to 50
    num_el = 128

    prob = om.Problem(model=Shape2DGroup(num_elements=num_el))
    prob.driver = om.ScipyOptimizeDriver()
#    prob.driver.options['optimizer'] = 'SLSQP'
#    prob.driver.options['tol'] = 1e-9
#    prob.driver.options['disp'] = True

    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer']="SNOPT"


    prob.setup()
    prob.run_model()
    print(prob['objective_comp.objective'],'\n')
#    prob.check_totals(compact_print=True)

    import timeit
    start = timeit.default_timer()
    prob.run_driver()
    stop = timeit.default_timer()
    print('time', stop-start)

    
#    print(prob['inputs_comp.control_points'],'control points \n')
#    print(prob['bspline_comp.points'],'points \n')
#    print('The value of the objective function is: \n')
#    print(prob['objective_comp.objective'],'\n')

    fea = set_fea(num_el)
    fea.uhat.vector().set_local(prob['uhat_comp.uhat'])
    fea.u.vector().set_local(prob['states_comp.displacements'])
    compliance = assemble(Constant(1.0)*dot(fea.h,fea.u)*ds)
    print(compliance)
    ALE.move(fea.mesh, fea.uhat)
    from matplotlib import pyplot as plt
    plot(fea.mesh)
    plt.show()

