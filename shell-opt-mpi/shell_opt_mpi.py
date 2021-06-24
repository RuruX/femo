from __future__ import division
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om

from set_fea import *
from states_comp import StatesComp
from objective_comp import ObjectiveComp
from constraint_comp import ConstraintComp
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
    

class ShellGroup(om.Group):

    def initialize(self):
        self.options.declare('fea')
        self.options.declare('volume')

    def setup(self):
        self.fea = fea = self.options['fea']
        self.volume = volume = self.options['volume']
        width = 2.
        length = 20.
        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('h', shape=fea.dof_f)
        self.add_subsystem('inputs_comp', inputs_comp)

        comp_1 = StatesComp(fea=fea)
        self.add_subsystem('states_comp', comp_1)

        comp_2 = ObjectiveComp(fea=fea)
        self.add_subsystem('objective_comp', comp_2)

        comp_3 = ConstraintComp(fea=fea)
        self.add_subsystem('constraint_comp', comp_3)

        self.connect('inputs_comp.h','states_comp.h')
        self.connect('inputs_comp.h','constraint_comp.h')
        self.connect('inputs_comp.h','objective_comp.h')
        self.connect('states_comp.displacements',
                     'objective_comp.displacements')

        h_bar = volume/(width*length)
        self.add_design_var('inputs_comp.h', lower=0.1, upper=3*h_bar)
        self.add_objective('objective_comp.objective')
        
        # width = 2, length = 20, average_h = 0.2
        self.add_constraint('constraint_comp.constraint', equals=volume)



if __name__ == '__main__':

    
    mesh = Mesh()
    filename = "plate3.xdmf"
    file = XDMFFile(mesh.mpi_comm(),filename)
    file.read(mesh)
    
##     Use the `refine` function on it to split every triangle into four smaller triangles
#    level = 1
#    for i in range(0,level):
#        mesh = refine(mesh)
#    
    fea = set_fea(mesh)
    volume = 8.
    width = 2.
    length = 20.
    group = ShellGroup(fea=fea,volume=volume)

    prob = om.Problem(model=group)
    rank = MPI.comm_world.Get_rank()
    num_pc = MPI.comm_world.Get_size()
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer']="SNOPT"
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-12
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-10
    prob.driver.opt_settings['Iterations limit'] = 10000000
    prob.setup()
    
    # plot the original deflection with uniform thickness
    h_bar = volume/(width*length)
    prob['inputs_comp.h'] = h_bar*np.ones(fea.dof_f)
    prob.run_model()
#    prob.check_totals(compact_print=True)
    u_mid,theta = fea.w.split(True)
    u_mid.rename("u","u")
    theta.rename("t","t")
    File('h-0.pvd') << fea.h
    File('t-0.pvd') << theta
    File('u-0.pvd') << u_mid
    h0_max = np.max(np.abs(fea.h.vector().get_local()))
    u0_max = np.max(np.abs(fea.w.vector().get_local()))

    # setting the starting point to be all ones as the default optimization setting
    prob['inputs_comp.h'] = np.ones(fea.dof_f)
    @profile(filename="profile_out")
    def main(prob):

        import timeit
        start = timeit.default_timer()
        prob.run_driver()
        stop = timeit.default_timer()
        h_max = np.max(np.abs(fea.h.vector().get_local()))
        u_max = np.max(np.abs(fea.w.vector().get_local()))

        if rank == 0:
            print('-'*10,'Before Optimization','-'*10)
            print('maximum thickness:', h0_max)
            print('maximum deflection:', u0_max)
            print('-'*10,'After Optimization','-'*10)
            print('maximum thickness:', h_max)
            print('maximum deflection:', u_max)
            print('compliance:', prob['objective_comp.objective'])
            print('volume:', prob['constraint_comp.constraint'])
            print('time for optimization:', stop-start)

#            print("Program finished!")
#            File = open('check_results.txt', 'a')
#            print(type(num_pc))
#            outputs = '\n{1:2d}     {2:.3f}     {3:f}'.format(
#                        num_pc, stop-start, compliance)
#            File.write(outputs)
#            File.close()

    cProfile.run('main(prob)', "profile_out")

    u_mid,theta = fea.w.split(True)
    u_mid.rename("u","u")
    theta.rename("t","t")
    File('h.pvd') << fea.h
    File('t.pvd') << theta
    File('u.pvd') << u_mid
    


