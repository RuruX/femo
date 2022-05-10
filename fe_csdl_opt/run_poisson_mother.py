

from fea_dolfinx import *
from state_model import StateModel
from output_model import OutputModel
import numpy as np
import csdl
from csdl import Model
from csdl_om import Simulator
from matplotlib import pyplot as plt
import argparse

'''
1. Define the mesh
'''

parser = argparse.ArgumentParser()
parser.add_argument('--nel',dest='nel',default='2',
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


fea = FEA(mesh, weak_bc=True)
# Add input to the PDE problem:
# name = 'input', function = input_function (function is the solution vector here)
input_function_space = FunctionSpace(mesh, ('DG', 0))
input_function = Function(input_function_space)

# Add states to the PDE problem (line 58):
# name = 'displacements', function = state_function (function is the solution vector here)
# residual_form = get_residual_form(u, v, rho_e) from atomics.pdes.thermo_mechanical_uniform_temp
# *inputs = input (can be multiple, here 'input' is the only input)

state_function_space = FunctionSpace(mesh, ('CG', 1))
state_function = Function(state_function_space)
v = TestFunction(state_function_space)
residual_form = pdeRes(state_function, v, input_function)

u_ex = fea.add_exact_solution(Expression_u, state_function_space)
f_ex = fea.add_exact_solution(Expression_f, input_function_space)

ALPHA = 1e-6

# Add output-avg_input to the PDE problem:
output_form = outputForm(state_function, input_function, u_ex)

fea.add_input('input', input_function)
fea.add_state(name='state',
                function=state_function,
                residual_form=residual_form,
                arguments=['input'])
fea.add_output('output',
                output_form,
                arguments=['input','state'])


'''
3. Define the boundary conditions
'''
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
fea.add_strong_bc(ubc, locate_BC1, state_function_space)
fea.add_strong_bc(ubc, locate_BC2, state_function_space)
fea.add_strong_bc(ubc, locate_BC3, state_function_space)
fea.add_strong_bc(ubc, locate_BC4, state_function_space)


########### Test the forward solve ##############
setFuncArray(input_function, getFuncArray(f_ex))

fea.solve(residual_form, state_function, fea.bc, report=False)
state_error = errorNorm(u_ex, state_function)
print("="*40)
control_error = errorNorm(f_ex, input_function)
print("Error in controls:", control_error)
state_error = errorNorm(u_ex, state_function)
print("Error in states:", state_error)
# print("number of controls dofs:", fea.total_dofs_f)
# print("number of states dofs:", fea.total_dofs_u)
print("="*40)

with XDMFFile(MPI.COMM_WORLD, "solutions/u.xdmf", "w") as xdmf:
    xdmf.write_mesh(fea.mesh)
    xdmf.write_function(fea.u)
with XDMFFile(MPI.COMM_WORLD, "solutions/f.xdmf", "w") as xdmf:
    xdmf.write_mesh(fea.mesh)
    xdmf.write_function(fea.f)
'''
4. Set up the CSDL model
'''
"""
4.1. write up the state Model
4.2. write up the output model
4.3. write up the fea model
"""
# prob = om.Problem()
#
# num_dof_input = fea.inputs_dict['input']['num_dof']
#
# comp = om.IndepVarComp()
# comp.add_output(
#     'input_unfiltered',
#     shape=num_dof_input,
#     val=np.random.random(num_dof_input) * 0.86,
# )
# prob.model.add_subsystem('indep_var_comp', comp, promotes=['*'])
#
# comp = GeneralFilterComp(input_function_space=input_function_space)
# prob.model.add_subsystem('general_filter_comp', comp, promotes=['*'])
#
#
# group = AtomicsGroup(fea=fea)
# prob.model.add_subsystem('atomics_group', group, promotes=['*'])
#
# prob.model.add_design_var('input_unfiltered',upper=1, lower=1e-4)
# prob.model.add_objective('compliance')
# prob.model.add_constraint('avg_input',upper=0.40)
#
'''
5. Set up the optimization problem
'''

# # set up the optimizer
# prob.driver = driver = om.pyOptSparseDriver()
# driver.options['optimizer'] = 'SNOPT'
# driver.opt_settings['Verify level'] = 0
#
# driver.opt_settings['Major iterations limit'] = 100000
# driver.opt_settings['Minor iterations limit'] = 100000
# driver.opt_settings['Iterations limit'] = 100000000
# driver.opt_settings['Major step limit'] = 2.0
#
# driver.opt_settings['Major feasibility tolerance'] = 1.0e-6
# driver.opt_settings['Major optimality tolerance'] =1.e-8
#
# prob.setup()
#
# if False:
#     prob.run_model()
#     prob.check_partials(compact_print=True)
# else:
#     prob.run_driver()
#
#
# #save the solution vector
# if method =='SIMP':
#     penalized_input  = project(input_function**3, input_function_space)
# else:
#     penalized_input  = project(input_function/(1 + 8. * (1. - input_function)), input_function_space)
#
# File('solutions/case_1/cantilever_beam/displacement.pvd') << state_function
# File('solutions/case_1/cantilever_beam/penalized_input.pvd') << penalized_input
