
from fe_csdl_opt.fea.fea_dolfinx import *
from fe_csdl_opt.csdl_opt.fea_model import FEAModel
from fe_csdl_opt.csdl_opt.state_model import StateModel
from fe_csdl_opt.csdl_opt.output_model import OutputModel
import numpy as np
import csdl
from csdl_om import Simulator as om_simulator
from python_csdl_backend import Simulator as py_simulator
from matplotlib import pyplot as plt
import argparse

'''
1. Define the mesh
'''

parser = argparse.ArgumentParser()
parser.add_argument('--nel',dest='nel',default='16',
                    help='Number of elements')

args = parser.parse_args()
num_el = int(args.nel)
mesh = createUnitSquareMesh(num_el)

'''
2. Set up the PDE problem
'''


PI = np.pi
ALPHA = 1E-6

def interiorResidual(u,v,f):
    mesh = u.function_space.mesh
    n = FacetNormal(mesh)
    h_E = CellDiameter(mesh)
    x = SpatialCoordinate(mesh)
    return inner(grad(u), grad(v))*dx \
            - inner(f, v)*dx

def boundaryResidual(u,v,u_exact,
                        sym=False,
                        beta_value=0.1,
                        overPenalize=False):

    '''
    Formulation from Github:
    https://github.com/MiroK/fenics-nitsche/blob/master/poisson/
    poisson_circle_dirichlet.py
    '''
    mesh = u.function_space.mesh
    n = FacetNormal(mesh)
    h_E = CellDiameter(mesh)
    x = SpatialCoordinate(mesh)
    beta = Constant(mesh, beta_value)
    sgn = 1.0
    if (sym is not True):
        sgn = -1.0
    retval = sgn*inner(u_exact-u, dot(grad(v), n))*ds \
            - inner(dot(grad(u), n), v)*ds
    penalty = beta*h_E**(-1)*inner(u-u_exact, v)*ds
    if (overPenalize or sym):
        retval += penalty
    return retval

def pdeRes(u,v,f,u_exact=None,weak_bc=False,sym=False):
    """
    The variational form of the PDE residual for the Poisson's problem
    """
    retval = interiorResidual(u,v,f)
    if (weak_bc):
        retval += boundaryResidual(u,v,u_exact,sym=sym)
    return retval

def outputForm(u, f, u_exact):
    return 0.5*inner(u-u_exact, u-u_exact)*dx + \
                    ALPHA/2*f**2*dx

class Expression_f:
    def __init__(self):
        self.alpha = 1e-6

    def eval(self, x):
        return (1/(1+self.alpha*4*np.power(PI,4))*
                np.sin(PI*x[0])*np.sin(PI*x[1]))

class Expression_u:
    def __init__(self):
        pass

    def eval(self, x):
        return (1/(2*np.power(PI, 2))*
                np.sin(PI*x[0])*np.sin(PI*x[1]))


fea = FEA(mesh)
# Record the function evaluations during optimization process
fea.record = True
# Add input to the PDE problem:
input_name = 'f'
input_function_space = FunctionSpace(mesh, ('DG', 0))
input_function = Function(input_function_space)
# Add state to the PDE problem:
state_name = 'u'
state_function_space = FunctionSpace(mesh, ('CG', 1))
state_function = Function(state_function_space)
v = TestFunction(state_function_space)

residual_form = pdeRes(state_function, v, input_function)
u_ex = fea.add_exact_solution(Expression_u, state_function_space)
f_ex = fea.add_exact_solution(Expression_f, input_function_space)


ALPHA = 1e-6
# Add output to the PDE problem:
output_name = 'l2_functional'
output_form = outputForm(state_function, input_function, u_ex)



'''
3. Define the boundary conditions
'''

############ Strongly enforced boundary conditions #############
ubc = Function(state_function_space)
ubc.vector.set(0.0)
locate_BC1 = locate_dofs_geometrical((state_function_space, state_function_space),
                            lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC2 = locate_dofs_geometrical((state_function_space, state_function_space),
                            lambda x: np.isclose(x[0], 1. ,atol=1e-6))
locate_BC3 = locate_dofs_geometrical((state_function_space, state_function_space),
                            lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC4 = locate_dofs_geometrical((state_function_space, state_function_space),
                            lambda x: np.isclose(x[1], 1. ,atol=1e-6))
locate_BC_list = [locate_BC1, locate_BC2, locate_BC3, locate_BC4]
fea.add_strong_bc(ubc, locate_BC_list, state_function_space)


############ Weakly enforced boundary conditions #############
############### Unsymmetric Nitsche's method #################
# residual_form = pdeRes(state_function, v, input_function,
#                         u_exact=u_ex, weak_bc=True, sym=False)
##############################################################



fea.add_input(input_name, input_function)
fea.add_state(name=state_name,
                function=state_function,
                residual_form=residual_form,
                arguments=[input_name])
fea.add_output(name=output_name,
                type='scalar',
                form=output_form,
                arguments=[input_name,state_name])



'''
4. Set up the CSDL model
'''


fea.PDE_SOLVER = 'Newton'
# fea.REPORT = True
fea_model = FEAModel(fea=[fea])
fea_model.create_input("{}".format(input_name),
                            shape=fea.inputs_dict[input_name]['shape'],
                            val=0.1*np.ones(fea.inputs_dict[input_name]['shape']) * 0.86)

# fea_model.connect('f','u_state_model.f')
# fea_model.connect('f','l2_functional_output_model.f')
# fea_model.connect('u_state_model.u','l2_functional_output_model.u')

fea_model.add_design_variable(input_name)
fea_model.add_objective(output_name)

# Ru: the new Python backend of CSDL has issue for promotions or connecting
# the variables for custom operations as from Aug 30.
sim = py_simulator(fea_model)
# sim = om_simulator(fea_model)
########### Test the forward solve ##############
# sim[input_name] = getFuncArray(f_ex)

sim.run()

########### Generate the N2 diagram #############
# sim.visualize_implementation()

############# Check the derivatives #############
# sim.check_totals()
# sim.check_partials(compact_print=True)
# sim.executable.check_totals(of='l2_functional', wrt='f',compact_print=True)
'''
5. Set up the optimization problem
'''
# ############## Run the optimization with pyOptSparse #############
# import openmdao.api as om
# ####### Driver = SNOPT #########
# driver = om.pyOptSparseDriver()
# driver.options['optimizer']='SNOPT'
#
# driver.opt_settings['Major feasibility tolerance'] = 1e-12
# driver.opt_settings['Major optimality tolerance'] = 1e-14
# driver.options['print_results'] = False
#
# sim.prob.driver = driver
# sim.prob.setup()
#
# sim.prob.run_driver()

############# Run the optimization with modOpt #############
from modopt.csdl_library import CSDLProblem

prob = CSDLProblem(
    problem_name='poisson_opt',
    simulator=sim,
)

from modopt.snopt_library import SNOPT

from modopt.scipy_library import SLSQP

optimizer = SNOPT(prob,
                  Major_optimality = 1e-13,
                  Major_feasibility = 1e-12)
                  #   append2file=True)
                  # append2file=False)
# optimizer = SLSQP(
#     prob,
#     ftol=1e-13,
#     maxiter=20,
# )

# # Check first derivatives at the initial guess, if needed
# optimizer.check_first_derivatives(prob.x0)

# Solve your optimization problem
optimizer.solve()
print("="*40)
# optimizer.print_results()

print("Objective value: ", sim['l2_functional_output_model.'+output_name])
print("="*40)
control_error = errorNorm(f_ex, input_function)
print("Error in controls:", control_error)
state_error = errorNorm(u_ex, state_function)
print("Error in states:", state_error)
print("="*40)

with XDMFFile(MPI.COMM_WORLD, "solutions/state_"+state_name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(fea.mesh)
    fea.states_dict[state_name]['function'].name = state_name
    xdmf.write_function(fea.states_dict[state_name]['function'])
with XDMFFile(MPI.COMM_WORLD, "solutions/input_"+input_name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(fea.mesh)
    fea.inputs_dict[input_name]['function'].name = input_name
    xdmf.write_function(fea.inputs_dict[input_name]['function'])
