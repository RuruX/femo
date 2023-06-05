"""
Structural analysis of the classic shell obstacle course:
1/3: Scordelis-Lo Roof
"""
from femo.fea.fea_dolfinx import *
from femo.csdl_opt.fea_model import FEAModel
from femo.csdl_opt.state_model import StateModel
from femo.csdl_opt.output_model import OutputModel
from femo.csdl_opt.pre_processor.general_filter_model \
                                    import GeneralFilterModel
import numpy as np
import csdl
from csdl import Model
from csdl_om import Simulator
from matplotlib import pyplot as plt
import argparse
from mpi4py import MPI
from shell_analysis_fenicsX import *

# The shell problem setup

roof = [#### tri mesh ####
        "roof_tri_30_20.xdmf",
        "roof_tri_60_40.xdmf",
        "roof5_25882.xdmf",
        "roof6_104524.xdmf",
        "roof12_106836.xdmf",
        #### quad mesh ####
        "roof_quad_3_2.xdmf",
        "roof_quad_6_4.xdmf",
        "roof_quad_12_8.xdmf",
        "roof_quad_24_16.xdmf",
        "roof_quad_30_20.xdmf",
        "roof_quad_60_40.xdmf",
        "roof_quad_120_80.xdmf",
        "roof_quad_240_160.xdmf",
        "roof_quad_360_240.xdmf"]

filename = "../shell_analysis_fenics/mesh/mesh-examples/scordelis-lo-roof/"+roof[0]
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

E_val = 4.32e8
nu_val = 0.0
h_val = 0.25
f_d = -90.

E = Constant(mesh,E_val) # Young's modulus
nu = Constant(mesh,nu_val) # Poisson ratio
# h = Constant(mesh,h_val) # Shell thickness

f = Constant(mesh, (0,0,f_d)) # Body force per unit area

element_type = "CG2CG1" # with quad/tri elements
#element_type = "CG2CR1" # with tri elements

element = ShellElement(
                mesh,
                element_type,
#                inplane_deg=3,
#                shear_deg=3
                )
dx_inplane, dx_shear = element.dx_inplane, element.dx_shear


def pdeRes(h,w,E,f,CLT,dx_inplane,dx_shear):
    elastic_model = ElasticModel(mesh,w,CLT)
    elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane,dx_shear)
    return elastic_model.weakFormResidual(elastic_energy, f)


def compliance(u_mid,h):
    h_mesh = CellDiameter(mesh)
    alpha = 1e-1
    dX = ufl.Measure('dx', domain=mesh, metadata={"quadrature_degree":0})
    return 0.5*dot(u_mid,u_mid)*dX \
            + 0.5*alpha*dot(grad(h), grad(h))*(h_mesh**2)*dX

def volume(h):
    return h*dx

#######################################################
############## The optimization problem ###############
#######################################################
fea = FEA(mesh)
# Add input to the PDE problem:
input_name = 'thickness'
input_function_space = FunctionSpace(mesh, ("DG", 0))
input_function = Function(input_function_space)

# Add state to the PDE problem:
state_name = 'displacements'
state_function_space = element.W
state_function = Function(state_function_space)
material_model = MaterialModel(E=E,nu=nu,h=input_function) # Simple isotropic material
residual_form = pdeRes(input_function,state_function,
                        E,f,material_model.CLT,dx_inplane,dx_shear)

# Add output to the PDE problem:
output_name_1 = 'compliance'
output_form_1 = compliance(state_function.sub(0), input_function)
output_name_2 = 'volume'
output_form_2 = volume(input_function)


fea.add_input(input_name, input_function)
fea.add_state(name=state_name,
                function=state_function,
                residual_form=residual_form,
                arguments=[input_name])
fea.add_output(name=output_name_1,
                type='scalar',
                form=output_form_1,
                arguments=[state_name,input_name])
fea.add_output(name=output_name_2,
                type='scalar',
                form=output_form_2,
                arguments=[input_name])
############ Set the BCs for the Scordelis-Lo roof problem ###################
ubc = Function(state_function_space)
ubc.vector.set(0.0)

locate_BC1 = locate_dofs_geometrical((state_function_space.sub(0).sub(1), state_function_space.sub(0).sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[0], 25. ,atol=1e-6))
locate_BC2 = locate_dofs_geometrical((state_function_space.sub(0).sub(2), state_function_space.sub(0).sub(2).collapse()[0]),
                                    lambda x: np.isclose(x[0], 25. ,atol=1e-6))
locate_BC3 = locate_dofs_geometrical((state_function_space.sub(0).sub(1), state_function_space.sub(0).sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC4 = locate_dofs_geometrical((state_function_space.sub(1).sub(0), state_function_space.sub(1).sub(0).collapse()[0]),
                                    lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC5 = locate_dofs_geometrical((state_function_space.sub(1).sub(2), state_function_space.sub(1).sub(2).collapse()[0]),
                                    lambda x: np.isclose(x[1], 0. ,atol=1e-6))
locate_BC6 = locate_dofs_geometrical((state_function_space.sub(0).sub(0), state_function_space.sub(0).sub(0).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC7 = locate_dofs_geometrical((state_function_space.sub(1).sub(1), state_function_space.sub(1).sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC8 = locate_dofs_geometrical((state_function_space.sub(1).sub(2), state_function_space.sub(1).sub(2).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))

bcs = [dirichletbc(ubc, locate_BC1, state_function_space.sub(0).sub(1)),
       dirichletbc(ubc, locate_BC2, state_function_space.sub(0).sub(2)),
       dirichletbc(ubc, locate_BC3, state_function_space.sub(0).sub(1)),
       dirichletbc(ubc, locate_BC4, state_function_space.sub(1).sub(0)),
       dirichletbc(ubc, locate_BC5, state_function_space.sub(1).sub(2)),
       dirichletbc(ubc, locate_BC6, state_function_space.sub(0).sub(0)),
       dirichletbc(ubc, locate_BC7, state_function_space.sub(1).sub(1)),
       dirichletbc(ubc, locate_BC8, state_function_space.sub(1).sub(2))
       ]

############ Strongly enforced boundary conditions #############
# fea.add_strong_bc(ubc, locate_BC_list, state_function_space)
for i in range(len(bcs)):
    fea.bc.append(bcs[i])
########## Solve with Newton solver wrapper: ##########
# solveNonlinear(F, w, bcs)


'''
4. Set up the CSDL model
'''

fea_model = FEAModel(fea=fea)
np.random.seed(0)
fea_model.create_input("{}".format('thickness'),
                            shape=nel,
                            val=np.ones(nel) * 0.25)

fea_model.add_design_variable('thickness', upper=1.0, lower=1e-4)
fea_model.add_objective('compliance')
fea_model.add_constraint('volume', equals=200.)
sim = Simulator(fea_model)

########### Test the forward solve ##############

sim.run()

########### Generate the N2 diagram #############
#sim.visualize_implementation()


############ Check the derivatives #############
sim.check_partials(compact_print=True)
# sim.prob.check_totals(compact_print=True)

'''
5. Set up the optimization problem
'''
############# Run the optimization with pyOptSparse #############
import openmdao.api as om
###### Driver = SNOPT #########
driver = om.pyOptSparseDriver()
driver.options['optimizer']='SNOPT'
driver.opt_settings['Verify level'] = 0

driver.opt_settings['Major iterations limit'] = 100000
driver.opt_settings['Minor iterations limit'] = 100000
driver.opt_settings['Iterations limit'] = 100000000
driver.opt_settings['Major step limit'] = 2.0

driver.opt_settings['Major feasibility tolerance'] = 1e-6
driver.opt_settings['Major optimality tolerance'] = 1e-8
driver.options['print_results'] = False

sim.prob.driver = driver
sim.prob.setup()

# from timeit import default_timer
# start = default_timer()

# sim.prob.run_driver()

# stop = default_timer()
# print('Optimization runtime:', str(stop-start), 'seconds')

print("Compliance value: ", sim['compliance'])

########## Output: ##############

uZ = computeNodalDisp(state_function.sub(0))[2]
# Comparing the results to the numerical solution
print("Scordelis-Lo roof theory tip deflection: v_tip = -0.3024")
print("Tip deflection:", min(uZ))
print("  Number of elements = "+str(mesh.topology.index_map(mesh.topology.dim).size_local))
print("  Number of vertices = "+str(mesh.topology.index_map(0).size_local))

########## Visualization: ##############

u_mid, _ = state_function.split()
with XDMFFile(MPI.COMM_WORLD, "solutions/u_mid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)
