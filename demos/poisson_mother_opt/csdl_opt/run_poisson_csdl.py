# from dolfin import *
import csdl
from csdl import Model
from csdl_om import Simulator
from matplotlib import pyplot as plt
from fea import *
from states_model import StatesModel
from scalar_output_model import ScalarOutputModel


class PoissonModel(Model):
    def initialize(self):
        self.parameters.declare('fea')

    def define(self):
        self.fea = fea = self.parameters['fea']

        f = self.create_input('f', shape=(fea.total_dofs_f,), val=1.0)

        self.add(StatesModel(fea=self.fea), 
                            name='states_model', promotes=[])
        self.add(ScalarOutputModel(fea=self.fea), 
                            name='scalar_output_model', promotes=[])
        self.connect('f', 'states_model.f')
        self.connect('f', 'scalar_output_model.f')
        self.connect('states_model.u', 'scalar_output_model.u')

        self.add_design_variable('f')
        self.add_objective('scalar_output_model.objective')


if __name__ == '__main__':

    num_el = 16
    mesh = createUnitSquareMesh(num_el)
    fea = FEA(mesh)

    f_ex = fea.f_ex
    u_ex = fea.u_ex
    model = PoissonModel(fea=fea)
    sim = Simulator(model)

    fea = model.fea
    # setting the design variable to be the exact solution
    ############## Run the simulation with the exact solution #########
    # sim['f'] = computeArray(f_ex)
    # sim.run()
    # print("="*40)
    # print("Objective value: ", sim['scalar_output_model.objective'])
    # control_error = errornorm(f_ex, fea.f)
    # print("Error in controls:", control_error)
    # state_error = errornorm(u_ex, fea.u)
    # print("Error in states:", state_error)
    # plt.figure(1)
    # plot(fea.u)
    # plt.show()

    # TODO: fix the `check_totals`
    # sim.check_partials(compact_print=True)
    # sim.prob.check_totals(compact_print=True)

    # TODO: 
    ############## Run the optimization with pyOptSparse #############
    import openmdao.api as om
    sim.prob.run_model()

    # sim.prob.driver = om.ScipyOptimizeDriver()
    # sim.prob.driver.options['optimizer'] = 'SLSQP'

    # sim.prob.run_driver()

    driver = om.pyOptSparseDriver()
    driver.options['optimizer']='SNOPT'
    driver.opt_settings['Major feasibility tolerance'] = 1e-12
    driver.opt_settings['Major optimality tolerance'] = 1e-13
    driver.options['print_results'] = False
    
    sim.prob.driver = driver
    sim.prob.run_driver()
    print("="*40)
    print("Objective value: ", sim['scalar_output_model.objective'])
    control_error = errorNorm(f_ex, fea.f)
    print("Error in controls:", control_error)
    state_error = errorNorm(u_ex, fea.u)
    print("Error in states:", state_error)
    print("="*40)
    # TODO: fix the check_first_derivatives()
    ############## Run the optimization with modOpt #############
    # from modopt.csdl_library import CSDLProblem

    # # Instantiate your problem using the csdl Simulator object and name your problem
    # prob = CSDLProblem(
    #     problem_name='poisson-mother',
    #     simulator=sim,
    # )
    
    # from modopt.snopt_library import SNOPT

    # optimizer = SNOPT(  prob, 
    #                     Major_iterations = 100, 
    #                     Major_optimality=1e-12, 
    #                     Major_feasibility=1e-13)

    # from modopt.scipy_library import SLSQP

    # # Setup your preferred optimizer (SLSQP) with the Problem object 
    # # Pass in the options for your chosen optimizer
    # optimizer = SLSQP(prob, maxiter=100)

    # # Check first derivatives at the initial guess, if needed
    # optimizer.check_first_derivatives(prob.x0)
    # # # Solve your optimization problem
    # optimizer.solve()

    # # # Print results of optimization
    # optimizer.print_results()
    
    # optimizer.print_available_outputs()