
from femo.fea.fea_dolfinx import *
from femo.csdl_opt.fea_model import FEAModel
from femo.csdl_opt.state_model import StateModel
from femo.csdl_opt.output_model import OutputModel
import numpy as np
import csdl
from python_csdl_backend import Simulator
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nel',dest='nel',default='16',
                    help='Number of elements')

parser.add_argument('--refine',dest='refine',default='1',
                    help='Level of mesh refinement')

args = parser.parse_args()

num_el = int(args.nel)
refine = int(args.refine)
'''
1. Define the mesh
'''
# import gmsh
# gmsh.initialize()
#
# model = gmsh.model()
# model.add("Circle")
# model.setCurrent("Circle")
# R = 1.
# xc = 0.
# yc = 0.
# disk = model.occ.addDisk(0, 0, 0, R, R)
# model.occ.synchronize()
# model.add_physical_group(2, [disk])
# # mesh refinement factor
# # mu = 2
# # gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", mu)
#
# gdim = 2
# model.mesh.generate(gdim)
# for i in range(refine):
#     model.mesh.refine()
# # gmsh.write("Circle.msh")


#
# from dolfinx.io.gmshio import model_to_mesh
# from mpi4py import MPI
# model_rank = 0
# mesh, cell_tags, facet_tags = model_to_mesh(model, MPI.COMM_SELF, model_rank, gdim=2)
# mesh.name = "Circle"
# gmsh.finalize()
# # with XDMFFile(mesh.comm, f"circle_1.xdmf", "w") as file:
# #     file.write_mesh(mesh)
# #     mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
#
#

mesh = createUnitSquareMesh(num_el)
# mesh = createIntervalMesh(num_el, -1., 1.)
# mesh = createRectangleMesh(np.array([-1.0,-1.0]),
#                             np.array([1., 1.]),
#                             num_el,
#                             num_el)
# mesh = dolfinx.mesh.create_mesh(num_el, domain=circle())
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "circle_1.xdmf", "r") as xdmf:
#     mesh = xdmf.read_mesh(name="Circle")
'''
2. Set up the PDE problem
'''


PI = np.pi
# ALPHA_1 = 2E-3
# ALPHA_2 = 3E-2
ALPHA_1 = 6E-7
ALPHA_2 = 2E-6

def P(u):
    return grad(u)

def g(u):
    return u**3

def interiorResidual(u,v,f):
    mesh = u.function_space.mesh
    n = FacetNormal(mesh)
    h_E = CellDiameter(mesh)
    x = SpatialCoordinate(mesh)
    return inner(P(u), grad(v))*dx \
            + inner(u**3,v)*dx \
            - inner(f, v)*dx

def boundaryResidual(u,v,u_exact,
                        sym=False,
                        beta_value=1e1,
                        overPenalize=False):
    mesh = u.function_space.mesh
    n = FacetNormal(mesh)
    h_E = CellDiameter(mesh)
    x = SpatialCoordinate(mesh)
    beta = Constant(mesh, beta_value)
    sgn = 1.0
    if (sym is not True):
        sgn = -1.0
    nitsche_1 = - inner(dot(P(u), n), v)*ds
    dP = derivative(P(u), u, v)
    nitsche_2 = sgn*inner(u_exact-u, dot(dP, n))*ds
    retval = nitsche_1 + nitsche_2
    penalty = beta*h_E**(-1)*inner(u-u_exact, v)*ds
    if (overPenalize or sym):
        retval += penalty
    return retval

def pdeRes(u,v,f,u_exact=None,weak_bc=False,sym=False,overPenalize=False):
    """
    The variational form of the PDE residual for the Poisson's problem
    """
    retval = interiorResidual(u,v,f)
    if (weak_bc):
        retval += boundaryResidual(u,v,u_exact,sym=sym,overPenalize=overPenalize)
    return retval

# H1 regularization
# def outputForm(u, f, u_exact):
#     return 0.5*inner(u-u_exact, u-u_exact)*dx + \
#                     ALPHA/2*(f**2+inner(grad(f),grad(f)))*dx
# L1 regularization
# def outputForm(u, f, u_exact):
#     return 0.5*inner(u-u_exact, u-u_exact)*dx + \
#                     ALPHA_2*abs(f)*dx
# # L2+L1 regularization
# def outputForm(u, f, u_exact):
#     return 0.5*inner(u-u_exact, u-u_exact)*dx + \
#                     ALPHA_1/2*f**2*dx + ALPHA_2*abs(f)*dx
# L2 regularization
def outputForm(u, f, u_exact):
    return 0.5*inner(u-u_exact, u-u_exact)*dx + \
                    ALPHA_1/2*f**2*dx

x = ufl.SpatialCoordinate(mesh)
u_ex_ufl = ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
# f_ex_ufl = 5.*ufl.pi**2*ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]) + \
#             (ufl.sin(2*ufl.pi*x[0])**3)*(ufl.sin(ufl.pi*x[1])**3)
fea = FEA(mesh)
# Record the function evaluations during optimization process
fea.record = True
# Add input to the PDE problem:
input_name = 'f'
input_function_space = FunctionSpace(mesh, ('DG', 0))
# input_function_space = FunctionSpace(mesh, ('CG', 1))
input_function = Function(input_function_space)
# Add state to the PDE problem:
state_name = 'u'
state_function_space = FunctionSpace(mesh, ('CG', 1))
state_function = Function(state_function_space)
v = TestFunction(state_function_space)

residual_form = pdeRes(state_function, v, input_function)
# u_ex = fea.add_exact_solution(Expression_u, state_function_space)
# f_ex = fea.add_exact_solution(Expression_f, input_function_space)
u_ex = Function(state_function_space)
project(u_ex_ufl,u_ex)
f_ex_ufl = -div(P(u_ex_ufl))+(u_ex_ufl)**3
f_ex = Function(input_function_space)
project(f_ex_ufl,f_ex)
# Add output to the PDE problem:
output_name = 'l2_functional'
output_form = outputForm(state_function, input_function, u_ex_ufl)



'''
3. Define the boundary conditions
'''

# ########### Strongly enforced boundary conditions #############
# locate_BC1 = locate_dofs_geometrical((state_function_space, state_function_space),
#                             lambda x: np.isclose(x[0], 0. ,atol=1e-8))
# locate_BC2 = locate_dofs_geometrical((state_function_space, state_function_space),
#                             lambda x: np.isclose(x[0], 1. ,atol=1e-8))
# locate_BC3 = locate_dofs_geometrical((state_function_space, state_function_space),
#                             lambda x: np.isclose(x[1], 0. ,atol=1e-8))
# locate_BC4 = locate_dofs_geometrical((state_function_space, state_function_space),
#                             lambda x: np.isclose(x[1], 1. ,atol=1e-8))
# locate_BC_list = [locate_BC1, locate_BC2, locate_BC3, locate_BC4]
#
# fea.add_strong_bc(u_ex, locate_BC_list, state_function_space)
# residual_form = pdeRes(state_function, v, input_function)

########### Weakly enforced boundary conditions #############
############## Unsymmetric Nitsche's method #################
residual_form = pdeRes(state_function, v, input_function,
                        u_exact=u_ex_ufl, weak_bc=True, sym=True)
#############################################################



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


# fea.PDE_SOLVER = 'Newton'
fea.PDE_SOLVER = 'SNES'
# fea.REPORT = True
x = SpatialCoordinate(mesh)
f_0_ufl = x[0]+x[1]
f_0 = lambda x: eval(str(f_0_ufl))
f_0_func = Function(input_function_space)
f_0_func.interpolate(f_0)
# f_0_func.vector[:] = 0.01*f_ex.vector

fea_model = FEAModel(fea=[fea])
fea_model.create_input("{}".format(input_name),
                            shape=fea.inputs_dict[input_name]['shape'],
                            val=0.1)
                            # val=25.*np.ones(fea.inputs_dict[input_name]['shape']))
                            # val=0.1)
                            # val=getFuncArray(f_0_func))
                            # val=10*np.ones(fea.inputs_dict[input_name]['shape']) * 0.86)

# fea_model.connect('f','u_state_model.f')
# fea_model.connect('f','l2_functional_output_model.f')
# fea_model.connect('u_state_model.u','l2_functional_output_model.u')

# fea_model.add_design_variable(input_name, lower=-12., upper=12.)
fea_model.add_design_variable(input_name)
fea_model.add_objective(output_name)

# Ru: the new Python backend of CSDL has issue for promotions or connecting
# the variables for custom operations as from Aug 30.
sim = Simulator(fea_model)
# sim = om_simulator(fea_model)
########### Test the forward solve ##############
# sim[input_name] = getFuncArray(f_ex)

sim.run()
# print("objective value:", sim[output_name])
########### Generate the N2 diagram #############
# sim.visualize_implementation()

############# Check the derivatives #############
# sim.check_totals()
# sim.check_partials(compact_print=True)
# sim.executable.check_totals(of='l2_functional', wrt='f',compact_print=True)
'''
5. Set up the optimization problem
'''
############# Run the optimization with modOpt #############
from modopt import CSDLProblem

prob = CSDLProblem(
    problem_name='nonlinear_poisson_opt',
    simulator=sim,
)

from modopt import SNOPT, SLSQP
# optimizer = SNOPT(prob,
#                   Major_iterations = 1000,
#                   Major_optimality = 1e-9,
#                   append2file=False)
optimizer = SLSQP(prob, maxiter=1000, ftol=1e-10)

# # Check first derivatives at the initial guess, if needed
# optimizer.check_first_derivatives(prob.x0)

# Solve your optimization problem
optimizer.solve()
print("="*40)
# optimizer.print_results()

print("Objective value: ", sim['l2_functional_output_model.'+output_name])
print("="*40)
control_error = errorNorm(f_ex_ufl, input_function)
print("Error in controls:", control_error)
state_error = errorNorm(u_ex_ufl, state_function)
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
with XDMFFile(MPI.COMM_WORLD, "solutions/f_ex.xdmf", "w") as xdmf:
    xdmf.write_mesh(fea.mesh)
    xdmf.write_function(f_ex)
with XDMFFile(MPI.COMM_WORLD, "solutions/u_ex.xdmf", "w") as xdmf:
    xdmf.write_mesh(fea.mesh)
    xdmf.write_function(u_ex)
