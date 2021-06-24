from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import openmdao.api as om 
from dolfin import *
from set_fea import set_fea
from states_comp import StatesComp 
from objective_comp import ObjectiveComp


class PoissonGroup(om.Group):

    def initialize(self):
        self.options.declare('num_elements', types=int)

    def setup(self):
        self.fea = set_fea(self.options['num_elements'])

        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('f', shape=len(self.fea.F.dofmap().dofs()))
        self.add_subsystem('inputs_comp', inputs_comp)

        comp_1 = StatesComp(fea=self.fea)
        self.add_subsystem('states_comp', comp_1)

        comp_2 = ObjectiveComp(fea=self.fea)
        self.add_subsystem('objective_comp', comp_2)

        self.connect('inputs_comp.f', 'states_comp.f')
        self.connect('states_comp.displacements', 'objective_comp.displacements')
        self.connect('inputs_comp.f', 'objective_comp.f')
        self.add_design_var('inputs_comp.f')
        self.add_objective('objective_comp.objective')

if __name__ == '__main__':

    num_el = 32
    group = PoissonGroup(num_elements=num_el)
    prob = om.Problem(model=group)
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer']="SNOPT"
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-12
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-12
    prob.setup()
    
    print('*'*20)
    print('Processor #', MPI.comm_world.Get_rank(), '-- check_totals:')
    print('*'*20)
    prob.run_model()
    prob.check_totals(compact_print=False)
    
    fea = group.fea
    x = SpatialCoordinate(fea.mesh)
    w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    alpha = Constant(1e-6)
    f_analytic = Expression("1/(1+alpha*4*pow(pi, 4))*w", w=w, alpha=alpha, degree=3)
    u_analytic = Expression("1/(2*pow(pi, 2))*f", f=f_analytic, degree=3)
    f_ex = interpolate(f_analytic, fea.F)
    u_ex = interpolate(u_analytic, fea.V)
    
    import timeit
    start = timeit.default_timer()
#    prob.run_driver()
    stop = timeit.default_timer()
    
    state_error = errornorm(u_ex, fea.u)
    control_error = errornorm(f_ex, fea.f)

    print('Time: ', stop - start)
    print("Objective: ", prob['objective_comp.objective'],'\n')
    print("Error in state:   %e." % state_error)
    print("Error in control: %e." % control_error)
    
