
from fe_csdl_opt.fea.fea_dolfinx import *
from fe_csdl_opt.csdl_opt.fea_model import FEAModel
from fe_csdl_opt.csdl_opt.state_model import StateModel
from fe_csdl_opt.csdl_opt.output_model import OutputModel
import numpy as np
import csdl
from csdl_om import Simulator
from matplotlib import pyplot as plt
import argparse

from motor_pde import pdeRes

'''
1. Define the mesh
'''

# data_path = "motor_data/motor_mesh_1_new/"
data_path = "motor_data/motor_mesh_1_old/"
mesh_file = data_path + "motor_mesh_1"
mesh, boundaries_mf, subdomains_mf, association_table = import_mesh(
    prefix=mesh_file,
    dim=2,
    subdomains=True
)
dx = Measure('dx', domain=mesh, subdomain_data=subdomains_mf)
dS = Measure('dS', domain=mesh, subdomain_data=boundaries_mf)
# mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)
winding_id = [42,]
magnet_id = [29,]
steel_id = [1,2,3]
winding_range = range(41,76+1)

# Subdomains for calculating power losses
ec_loss_subdomain = [1,2,] # rotor and stator core
hysteresis_loss_subdomain = [1,2,]
pm_loss_subdomain = range(29, 40+1)
'''
2. Set up the PDE problem
'''
# PROBLEM SPECIFIC PARAMETERS
Hc = 838.e3  # 838 kA/m
p = 12
s = 3 * p
vacuum_perm = 4e-7 * np.pi
angle = 0.
iq = 282.2  / 0.00016231
##################### mesh motion subproblem ######################
# fea_mm = FEA(mesh)

# states for mesh motion subproblem
state_name_mm = 'uhat'
state_function_space_mm = VectorFunctionSpace(mesh, ('CG', 1))
state_function_mm = Function(state_function_space_mm)
state_function_mm.vector.set(0.0)
v_mm = TestFunction(state_function_space_mm)

##################### electomagnetic subproblem ######################
fea_em = FEA(mesh)
# Add input to the PDE problem: the inputs as the previous states

# Add state to the PDE problem:
# states for electromagnetic equation
state_name_em = 'u'
state_function_space_em = FunctionSpace(mesh, ('CG', 1))
state_function_em = Function(state_function_space_em)
v_em = TestFunction(state_function_space_em)

residual_form = pdeRes(state_function_em,v_em,state_function_mm,iq,dx,p,s,Hc,vacuum_perm,angle)

# # Add output to the PDE problem:
# output_name = 'output'
# output_form = outputForm(state_function_em, input_function, u_ex)


'''
3. Define the boundary conditions
'''

############ Strongly enforced boundary conditions (mesh_old)#############
ubc = Function(state_function_space_em)
ubc.vector.set(0.0)
locate_BC1 = locate_dofs_geometrical((state_function_space_em, state_function_space_em),
                            lambda x: np.isclose(x[0]**2+x[1]**2, 0.0144 ,atol=1e-6))
locate_BC_list = [locate_BC1,]
fea_em.add_strong_bc(ubc, locate_BC_list, state_function_space_em)

# ############ Strongly enforced boundary conditions (mesh_new)#############
# ubc = Function(state_function_space_em)
# ubc.vector.set(0.0)
# locate_BC1 = locate_dofs_geometrical((state_function_space_em, state_function_space_em),
#                             lambda x: np.isclose(x[0]**2+x[1]**2, 0.0144 ,atol=1e-6))
# locate_BC2 = locate_dofs_geometrical((state_function_space_em, state_function_space_em),
#                             lambda x: np.isclose(x[0]**2+x[1]**2, 0.0036 ,atol=1e-6))
# locate_BC_list = [locate_BC1, locate_BC2,]
# locate_BC_list = [locate_BC1,]
# fea_em.add_strong_bc(ubc, locate_BC_list, state_function_space_em)

############ Weakly enforced boundary conditions #############

##############################################################


fea_em.add_input(state_name_mm, state_function_mm)
fea_em.add_state(name=state_name_em,
                function=state_function_em,
                residual_form=residual_form,
                arguments=[state_name_mm])
# fea_em.add_output(name=output_name,
#                 type='scalar',
#                 form=output_form,
#                 arguments=[input_name,state_name])



'''
4. Set up the CSDL model
'''
fea_em.PDE_SOLVER = 'SNES'
fea_em.REPORT = True

fea_model = FEAModel(fea=fea_em)
fea_model.create_input("{}".format(state_name_mm),
                            shape=fea_em.inputs_dict[state_name_mm]['shape'],
                            val=np.random.random(fea_em.inputs_dict[state_name_mm]['shape']) * 0.0)

# fea_model.add_design_variable(input_name)
# fea_model.add_objective(output_name)

sim = Simulator(fea_model)

########### Test the forward solve ##############
#sim[input_name] = getFuncArray(f_ex)

sim.run()
############# Check the derivatives #############
#sim.check_partials(compact_print=True)
#sim.prob.check_totals(compact_print=True)  

# '''
# 5. Set up the optimization problem
# '''
# ############## Run the optimization with pyOptSparse #############
# import openmdao.api as om
# ####### Driver = SNOPT #########
# driver = om.pyOptSparseDriver()
# driver.options['optimizer']='SNOPT'

# driver.opt_settings['Major feasibility tolerance'] = 1e-12
# driver.opt_settings['Major optimality tolerance'] = 1e-14
# driver.options['print_results'] = False

# sim.prob.driver = driver
# sim.prob.setup()
# sim.prob.run_driver()


# print("Objective value: ", sim[output_name])
# print("="*40)
# control_error = errorNorm(f_ex, input_function)
# print("Error in controls:", control_error)
# state_error = errorNorm(u_ex, state_function_em)
# print("Error in states:", state_error)
# print("="*40)

with XDMFFile(MPI.COMM_WORLD, "solutions/state_"+state_name_em+".xdmf", "w") as xdmf:
    xdmf.write_mesh(fea_em.mesh)
    fea_em.states_dict[state_name_em]['function'].name = state_name_em
    xdmf.write_function(fea_em.states_dict[state_name_em]['function'])
with XDMFFile(MPI.COMM_WORLD, "solutions/input_"+state_name_mm+".xdmf", "w") as xdmf:
    xdmf.write_mesh(fea_em.mesh)
    fea_em.inputs_dict[state_name_mm]['function'].name = state_name_mm
    xdmf.write_function(fea_em.inputs_dict[state_name_mm]['function'])
    
    
    


