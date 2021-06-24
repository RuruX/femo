from __future__ import division
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om

from set_fea import *
from states_comp import StatesComp
from objective_comp import ObjectiveComp
from constraint_comp import ConstraintComp
from bspline_comp import BsplineComp, get_bspline_mtx
from uhat_comp import uhatComp
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
    

class Shape2DGroup(om.Group):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('num_cp', types=int)

    def setup(self):
        self.fea = fea = set_fea(self.options['num_elements'])
        N_fac = 8
        num_pt = self.options['num_elements']*N_fac+1
        num_cp = self.options['num_cp']  # fixxed to support larger num_elements 
        
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
        self.connect('uhat_comp.uhat', 'constraint_comp.uhat')

        thickness = fea.length / N_fac
        self.add_design_var('inputs_comp.control_points', upper=0.90*thickness)
        self.add_objective('objective_comp.objective')
        self.add_constraint('constraint_comp.constraint', equals=0.0)



if __name__ == '__main__':

    num_el = 128
    num_cp = 25
    group = Shape2DGroup(num_elements=num_el, num_cp=num_cp)

    prob = om.Problem(model=group)
    rank = MPI.comm_world.Get_rank()
    num_pc = MPI.comm_world.Get_size()
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer']="SNOPT"
#    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-8
#    prob.driver.opt_settings['Major optimality tolerance'] = 1e-7
    prob.setup()
    prob.run_model()
#    print(prob['objective_comp.objective'],'\n')
#    prob.check_totals(compact_print=True)

    fea = group.fea
    @profile(filename="profile_out")
    def main(prob):

        import timeit
        start = timeit.default_timer()
        prob.run_driver()
        stop = timeit.default_timer()
        compliance = assemble(Constant(1.0)*dot(fea.h,fea.u)*ds)
        if rank == 0:
            print(prob['bspline_comp.points'])
            print("Program finished!")
            File = open('check_results.txt', 'a')
            outputs = '\n{0:3d}      {1:2d}      {2:2d}     {3:.3f}        {4:e}'.format(
                        num_el, num_cp, num_pc, stop-start, compliance)
            File.write(outputs)
            File.close()

    cProfile.run('main(prob)', "profile_out")

    ALE.move(fea.mesh, fea.uhat)
    File('uhat.pvd') << fea.uhat
    File('u.pvd') << fea.u


