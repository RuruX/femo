from __future__ import division
from dolfin import *
import numpy as np 
import matplotlib.pyplot as plt
import openmdao.api as om 
# from stiffness_tensor_group_E_e import StiffnesstensorGroup
# from mdocomps.set_fea_full import set_fea
# from mdocomps.states_comp import StatesComp 
# from mdocomps.compliancecomp import ComplianceComp
# from mdocomps.penalty_comp import Penaltycomp

from set_fea_full import set_fea
from states_comp import StatesComp 
from compliancecomp import ComplianceComp
from penalty_comp import Penaltycomp
from constraintscomp import Constraintscomp
from densityfilter_comp import DensityFilterComp

from openmdao.api import pyOptSparseDriver

import cProfile, pstats, io

# mpirun -n 2 python3 script.py



def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


class TopoGroup(om.Group):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('fea')
      

    def setup(self):
        self.fea = set_fea(self.options['num_elements'])

        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('rho_e', shape=self.fea.num_var,val=np.ones(self.fea.num_var)*0.7)
        self.add_subsystem('inputs_comp', inputs_comp,promotes=['*'])

        # comp = DensityFilterComp()
        # self.add_subsystem('DensityFilterComp', comp,promotes=['*'])

        comp = Penaltycomp(num_ele=self.fea.num_var)
        self.add_subsystem('Penaltycomp', comp,promotes=['*'])

        comp_1 = StatesComp(fea=self.fea)
        self.add_subsystem('StatesCompFEniCs', comp_1,promotes=['*'])

        # inputs_comp = om.IndepVarComp()
        # inputs_comp.add_output('displacements', shape=self.fea.num_var,val=np.ones(self.fea.num_var)*0.7)
        # self.add_subsystem('StatesComp_FEniCs', inputs_comp)

        comp_2 = ComplianceComp(fea=self.fea)
        self.add_subsystem('ComplianceComp', comp_2,promotes=['*'])

        comp_3 = Constraintscomp(fea=self.fea)
        self.add_subsystem('Constraintscomp', comp_3, promotes=['*'])


        self.add_design_var('rho_e',upper=1, lower=0.)
        self.add_objective('compliance')
        self.add_constraint('volume_fraction',equals=0.5)


# @profile
if __name__ == '__main__':
    from openmdao.devtools import iprofile
    import cProfile
    import pstats
    # from pstats import SortKey
    # iprofile.setup()
    # iprofile.start()
    
    num_el = 9
    prob = om.Problem(model=TopoGroup(num_elements=num_el))

    def main(prob):

        # prob.driver = om.ScipyOptimizeDriver()
        # # prob.driver = om.SimpleGADriver()

        # prob.driver.options['optimizer'] = 'SLSQP'
        # prob.driver.options['tol'] = 1e-5
        # prob.driver.options['disp'] = True

        prob.driver = driver = pyOptSparseDriver()
        driver.options['optimizer'] = 'SNOPT'
        driver.opt_settings['Verify level'] = 0
        driver.opt_settings['Major iterations limit'] = 100000
        driver.opt_settings['Minor iterations limit'] = 100000
        driver.opt_settings['Iterations limit'] = 100000000
        driver.opt_settings['Major step limit'] = 2.0

        driver.opt_settings['Major feasibility tolerance'] = 1.0e-5
        driver.opt_settings['Major optimality tolerance'] =1.e-8
        # driver.opt_settings['Minor feasibility tolerance'] = 1.0e-6
        # driver.opt_settings['Minor optimality tolerance'] =5.e-5

        prob.setddup()
        # File('elasticity1/stiffness0.pvd') << prob.model.fea.C
        # prob.run_model()
        # File('elasticityn/stiffness0.pvd') << prob.model.fea.C

        # prob.check_partials(compact_print=True)  
        # prob.model.list_outputs()
        
        prob.run_driver()
        # prob.check_partials(compact_print=True)  
        # prob.model.list_outputs()

        # File('elasticity1/displacement.pvd') << prob.model.fea.u

        # File('elasticityn/stiffness.pvd') << prob.model.fea.C

        File('elasticity_line/displacement.pvd') << prob.model.fea.u

        File('elasticity_line/stiffness.pvd') << prob.model.fea.C

    cProfile.run('main(prob)',"output.dat")


    with open("output_time.txt","w") as f:
        p = pstats.Stats("output.dat",stream=f)
        p.sort_stats("time").print_stats()
    
    with open("output_call.txt","w") as f:
        p = pstats.Stats("output.dat",stream=f)
        p.sort_stats("calls").print_stats()    
    # iprofile.stop()



