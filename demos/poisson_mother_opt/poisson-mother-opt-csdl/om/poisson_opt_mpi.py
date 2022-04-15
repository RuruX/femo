from __future__ import division
from dolfin import *
import numpy as np 
import matplotlib.pyplot as plt
import openmdao.api as om 
from set_fea import *
#from set_fea_3d import *
from states_comp import StatesComp 
from objective_comp import ObjectiveComp
import cProfile, pstats, io

def profile(filename=None, comm=MPI.comm_world):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.Get_rank())
                pr.dump_stats(filename_r)

            return result
        return wrap_f
    return prof_decorator

class PoissonGroup(om.Group):

    def initialize(self):
        self.options.declare('num_elements', types=int)

    def setup(self):
        num_el = self.options['num_elements']
        self.fea = set_fea(num_elements=num_el)
        
        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('f', shape=self.fea.dof_f)
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

    num_el = 16
    group = PoissonGroup(num_elements=num_el)
    prob = om.Problem(model=group)
    rank = MPI.comm_world.Get_rank()
    np = MPI.comm_world.Get_size()
    
    prob.driver = driver = om.pyOptSparseDriver()
    driver.options['optimizer']='SNOPT'
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-12
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-13
    prob.driver.options['print_results'] = False
    prob.setup()
    
#    print('*'*20)
#    print('Processor #', rank, '-- check_totals:')
#    print('*'*20)
#    prob.run_model()
#    prob.check_totals(compact_print=False)

    fea = group.fea
    x = SpatialCoordinate(fea.mesh)
    w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    alpha = Constant(1e-6)
    f_analytic = Expression("1/(1+alpha*4*pow(pi,4))*w", w=w, alpha=alpha, degree=3)
    u_analytic = Expression("1/(2*pow(pi, 2))*f", f=f_analytic, degree=3)
    f_ex = interpolate(f_analytic, fea.F)
    u_ex = interpolate(u_analytic, fea.V)
        

    @profile(filename="profile_out")
    def main(prob):

        import timeit
        start = timeit.default_timer()
        prob.run_driver()
        stop = timeit.default_timer()
        state_error = errornorm(u_ex, fea.u)
        control_error = errornorm(f_ex, fea.f)

        if rank == 0:
            File = open('check_results.txt', 'a')
            outputs = '\n{0:3d}      {1:2d}      {2:.3f}      {3:e}    {4:e}'.format(
                        num_el, np, stop-start, state_error, control_error)
            File.write(outputs)
            File.close()
        


    cProfile.run('main(prob)', "profile_out")

    File('u.pvd') << fea.u
    File('f.pvd') << fea.f

