
import csdl
from csdl import Model
from csdl_om import Simulator
from matplotlib import pyplot as plt
from fea import *
from state_model import StateModel
from output_model import OutputModel

import argparse


class PoissonModel(Model):
    def initialize(self):
        self.parameters.declare('fea')

    def define(self):
        self.fea = fea = self.parameters['fea']

        f = self.create_input('f', shape=(fea.total_dofs_f,),
                            val=getFuncArray(self.fea.initial_guess_f))

        self.add(StateModel(fea=self.fea, debug_mode=False),
                            name='state_model', promotes=[])
        self.add(OutputModel(fea=self.fea),
                            name='output_model', promotes=[])
        self.connect('f', 'state_model.f')
        self.connect('f', 'output_model.f')
        self.connect('state_model.u', 'output_model.u')

        self.add_design_variable('f')
        self.add_objective('output_model.objective')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nel',dest='nel',default='16',
                        help='Number of elements')

    args = parser.parse_args()
    num_el = int(args.nel)
    mesh = createUnitSquareMesh(num_el)
    fea = FEA(mesh, weak_bc=True)

    f_ex = fea.f_ex
    u_ex = fea.u_ex
    model = PoissonModel(fea=fea)
    sim = Simulator(model)

    fea = model.fea

    ############## Run the simulation with the exact solution #########
    # setting the design variable to be the exact solution
    # sim['f'] = computeArray(f_ex)
    sim.run()
    print("Objective value: ", sim['output_model.objective'])


    ############## Check the derivatives #############
    # sim.check_partials(compact_print=True)
    # sim.prob.check_totals(compact_print=True)

    ############## Run the optimization with pyOptSparse #############
    import openmdao.api as om
    ####### Driver = SNOPT #########
    driver = om.pyOptSparseDriver()
    driver.options['optimizer']='SNOPT'
    driver.opt_settings['Major feasibility tolerance'] = 1e-12
    driver.opt_settings['Major optimality tolerance'] = 1e-13
    driver.options['print_results'] = False

    sim.prob.driver = driver
    sim.prob.run_driver()

    ############## Output ###################
    print("="*40)
    print("Objective value: ", sim['output_model.objective'])
    control_error = errorNorm(f_ex, fea.f)
    print("Error in controls:", control_error)
    state_error = errorNorm(u_ex, fea.u)
    print("Error in states:", state_error)
    print("="*40)

    ########### Postprocessing with DOLFIN #############
    # plt.figure(1)
    # plot(fea.u)
    # plt.show()
    # File('f_opt_dolfin.pvd') << fea.f
    # File('u_opt_dolfin.pvd') << fea.u

    ########### Postprocessing with DOLFINx #############
    with XDMFFile(MPI.COMM_WORLD, "solutions/u_opt_dolfinx.xdmf", "w") as xdmf:
        xdmf.write_mesh(fea.mesh)
        xdmf.write_function(fea.u)
    with XDMFFile(MPI.COMM_WORLD, "solutions/f_opt_dolfinx.xdmf", "w") as xdmf:
        xdmf.write_mesh(fea.mesh)
        xdmf.write_function(fea.f)

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
