
from requests import post
from fe_csdl_opt.fea.fea_dolfinx import *
from fe_csdl_opt.csdl_opt.fea_model import FEAModel
from fe_csdl_opt.csdl_opt.state_model import StateModel
from fe_csdl_opt.csdl_opt.output_model import OutputModel
import numpy as np
import csdl
from csdl_om import Simulator
from matplotlib import pyplot as plt
import argparse

from motor_pde import pdeResEM
from power_loss_model import LossSumModel, PowerLossModel

I = Identity(2)

def pdeResMM(uhat, duhat, g=None,
            nitsche=False, sym=False, overpenalty=False, ds_=ds):
    """
    Formulation of mesh motion as a hyperelastic problem
    """
    # Residual for mesh, which satisfies a fictitious elastic problem:
    def _F(u):
        return grad(u)+I
    def _sigma(u):
        F = _F(u)
        E = 0.5*(F.T*F-I)
        m_jac_stiff_pow = 3
        # Artificially stiffen the mesh where it is getting crushed:
        K = 1/pow(det(F),m_jac_stiff_pow)
        mu = 1/pow(det(F),m_jac_stiff_pow)
        S = K*tr(E)*I + 2.0*mu*(E - tr(E)*I/3.0)
        return S
    def P(u):
        return _F(u)*_sigma(u)

    F_m = _F(uhat)
    S_m = _sigma(uhat)
    P_m = P(uhat)
    dS_m = _sigma(duhat)
    res_m = (inner(P_m,grad(duhat)))*dx


    if nitsche is True:
        beta = 50/pow(det(F_m),3)
        sgn = 1.0
        if sym is not True:
            sgn = -1.0
        n = FacetNormal(mesh)
        h_E = CellDiameter(mesh)
        f0 = -div(P(g))
        res_m += -dot(f0, duhat)*dx
        nitsche_1 = - inner(dot(P_m,n),duhat)
        nitsches_term_1 = nitsche_1("+")*ds_ + nitsche_1("-")*ds_
        dP = derivative(P_m, uhat, duhat)
        nitsche_2 = sgn * inner(dP*n,uhat-g)
        nitsches_term_2 = nitsche_2("+")*ds_ + nitsche_2("-")*ds_
        penalty = beta/h_E*inner(duhat,uhat-g)
        penalty_term = penalty("+")*ds_ + penalty("-")*ds_
        res_m += nitsches_term_1
        res_m += nitsches_term_2
        if sym is True or overpenalty is True:
            res_m += penalty_term
    return res_m

def getBCDerivatives(uhat, bc_indices):
    """
    Compute the derivatives of the PDE residual of the mesh motion
    subproblem wrt the BCs, which is a fixed sparse matrix with "-1"s
    on the entries corresponding to the edge indices.
    """
    total_dofs = len(uhat.x.array)
    total_dofs_bc = len(bc_indices)
    row_ind = bc_indices
    col_ind = bc_indices
    data = -1.0*np.ones(total_dofs_bc)
    M = csr_matrix((data, (row_ind, col_ind)),
                    shape=(total_dofs, total_dofs))
    M_petsc = PETSc.Mat().createAIJ(size=M.shape,csr=(M.indptr,M.indices,M.data))
    return M_petsc

def B_power_form(A_z, uhat, n, dx, subdomains):
    """
    Return the ufl form of `B**n*dx(subdomains)`
    """
    gradA_z = gradx(A_z,uhat)
    B_power_form = 0.
    B_magnitude = sqrt(gradA_z[0]**2+gradA_z[1]**2)
    for subdomain_id in subdomains:
        B_power_form += pow(B_magnitude, n)*J(uhat)*dx(subdomain_id)
    return B_power_form

def area_form(uhat, dx, subdomains):
    """
    Return the ufl form of `uhat*dx(subdomains)`
    """
    if type(subdomains) == int:
        subdomain_group = [subdomains]
    else:
        subdomain_group = subdomains
    area = 0
    for subdomain_id in subdomain_group:
        area += J(uhat)*dx(subdomain_id)
    return area

def B(A_z, uhat):
    gradA_z = gradx(A_z,uhat)
    B_form = as_vector((gradA_z[1], -gradA_z[0]))
    # dB_dAz = derivative(B_form, state_function_em)

    VB = VectorFunctionSpace(mesh,('DG',0))
    B = Function(VB)
    project(B_form,B)
    return B

def advance(func_old,func,i,increment_deltas):
    func_old.vector[:] = func.vector
    func_old.vector[edge_indices.astype(np.int32)] = (i+1)*increment_deltas[edge_indices.astype(np.int32)]

def solveIncremental(res,func,bc,report):
    func_old = input_function_mm
    # Get the relative movements from the previous step
    relative_edge_deltas = func_old.vector[:] - func.vector[:]
    STEPS, increment_deltas = getDisplacementSteps(func_old,
                                                relative_edge_deltas,
                                                mesh)
    # print("Nonzero edge movements:",increment_deltas[np.nonzero(increment_deltas)])
    # newton_solver = NewtonSolver(res, func, bc, rel_tol=1e-6, report=report)

    snes_solver = SNESSolver(res, func, bc, rel_tol=1e-6, report=report)
    # Incrementally set the BCs to increase to `edge_deltas`
    print(80*"=")
    print(' FEA: total steps for mesh motion:', STEPS)
    print(80*"=")
    for i in range(STEPS):
        if report == True:
            print(80*"=")
            print("  FEA: Step "+str(i+1)+" of mesh movement")
            print(80*"=")
        advance(func_old,func,i,increment_deltas)
        # func_old.vector[:] = func.vector
        # # func_old.vector[edge_indices] = 0.0
        # func_old.vector[np.nonzero(relative_edge_deltas)] = (i+1)*increment_deltas[np.nonzero(relative_edge_deltas)]
        # print(assemble_vector(form(res)))
        # newton_solver.solve(func)
        snes_solver.solve(None, func.vector)
        print(func_old.x.array[np.nonzero(relative_edge_deltas)][:10])
        print(func.x.array[np.nonzero(relative_edge_deltas)][:10])
    # Ru: A temporary correction for the mesh movement solution to make the inner boundary
    # curves not moving
    advance(func,func,i,increment_deltas)
    if report == True:
        print(80*"=")
        print(' FEA: L2 error of the mesh motion on the edges:',
                    np.linalg.norm(func.vector[np.nonzero(relative_edge_deltas)]
                             - relative_edge_deltas[np.nonzero(relative_edge_deltas)]))
        print(80*"=")
'''
1. Define the mesh
'''

mesh_name = "motor_mesh_test_1"
data_path = "motor_data_latest_coarse/"

# mesh_name = "motor_mesh_coarse_1"
# data_path = "motor_data_old/"
mesh_file = data_path + mesh_name
mesh, boundaries_mf, subdomains_mf, association_table = import_mesh(
    prefix=mesh_file,
    dim=2,
    subdomains=True
)

'''
The boundary movement data
'''
f = open(data_path+'init_edge_coords_coarse_1.txt', 'r+')
init_edge_coords = np.fromstring(f.read(), dtype=float, sep=' ')
f.close()

f = open(data_path+'edge_coord_deltas_coarse_1.txt', 'r+')
edge_deltas = np.fromstring(f.read(), dtype=float, sep=' ')
f.close()


dx = Measure('dx', domain=mesh, subdomain_data=subdomains_mf)
dS = Measure('dS', domain=mesh, subdomain_data=boundaries_mf)
# mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)
winding_id = [15,]
magnet_id = [3,]
steel_id = [1,2,51]
winding_range = range(15,50+1)

# Subdomains for calculating power losses
ec_loss_subdomain = [1,2,] # rotor and stator core
hysteresis_loss_subdomain = [1,2,]
pm_loss_subdomain = range(3, 14+1)

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
# iq = 282.2  / 1.0
##################### mesh motion subproblem ######################
fea_mm = FEA(mesh)

fea_mm.PDE_SOLVER = 'SNES'
fea_mm.REPORT = True
fea_mm.record = True


# inputs for mesh motion subproblem
input_name_mm = 'uhat_bc'
input_function_space_mm = VectorFunctionSpace(mesh, ('CG', 1))
input_function_mm = Function(input_function_space_mm)

edge_indices = locateDOFs(init_edge_coords,input_function_space_mm)
fea_mm.custom_solve = solveIncremental
input_function_mm.vector.set(0.0)
for i in range(len(edge_deltas)):
    input_function_mm.vector[edge_indices[i]] = 0.1*edge_deltas[i]
input_array = input_function_mm.x.array

# states for mesh motion subproblem
state_name_mm = 'uhat'
state_function_space_mm = VectorFunctionSpace(mesh, ('CG', 1))
state_function_mm = Function(state_function_space_mm)
state_function_mm.vector.set(0.0)
v_mm = TestFunction(state_function_space_mm)

# Add output to the PDE problem:
output_name_mm_1 = 'winding_area'
output_form_mm_1 = area_form(state_function_mm, dx, winding_id)
output_name_mm_2 = 'magnet_area'
output_form_mm_2 = area_form(state_function_mm, dx, magnet_id)
output_name_mm_3 = 'steel_area'
output_form_mm_3 = area_form(state_function_mm, dx, steel_id)

# ############ Strongly enforced boundary conditions #############
# ubc_mm = Function(state_function_space_mm)
# ubc_mm.vector.set(0.0)
# ######## new mesh ############
# locate_BC1_mm = locate_dofs_geometrical((state_function_space_mm, state_function_space_mm),
#                             lambda x: np.isclose(x[0]**2+x[1]**2, 0.0144 ,atol=1e-6))
# locate_BC2_mm = locate_dofs_geometrical((state_function_space_mm, state_function_space_mm),
#                             lambda x: np.isclose(x[0]**2+x[1]**2, 0.0036 ,atol=1e-6))

# locate_BC_list_mm = [locate_BC1_mm, locate_BC2_mm,]


# fea_mm.add_strong_bc(ubc_mm, locate_BC_list_mm, state_function_space_mm)


# ############ Strongly enforced boundary conditions (mesh_new)#############=

# residual_form_mm = pdeResMM(state_function_mm, v_mm)
# ubc_mm = Function(state_function_space_mm)
# ubc_mm.vector[:] = input_function_mm.vector
# winding_cells = subdomains_mf.find(winding_id[0])
# boundary_facets = boundaries_mf.find(1000)
# locate_BC1_mm = locate_dofs_topological(input_function_space_mm, mesh.topology.dim-1, boundary_facets)
# locate_BC_list_mm = [locate_BC1_mm,]
# fea_mm.add_strong_bc(ubc_mm, locate_BC_list_mm)


# # TODO: move the incremental solver outside of the FEA class,
# # and make it user-defined instead.
# fea_mm.ubc = ubc_mm
# #########################################################
# dR_duhat_bc = getBCDerivatives(state_function_mm, edge_indices)
# fea_mm.add_input(name=input_name_mm,
#                 function=input_function_mm)
# fea_mm.add_state(name=state_name_mm,
#                 function=state_function_mm,
#                 residual_form=residual_form_mm,
#                 dR_df_list=[dR_duhat_bc],
#                 arguments=[input_name_mm])



############ Weakly enforced boundary conditions #############

fea_mm.ubc = input_function_mm
residual_form_mm = pdeResMM(state_function_mm, v_mm, g=input_function_mm,
                                nitsche=True, sym=True, overpenalty=False,ds_=dS(1000))
fea_mm.add_input(name=input_name_mm,
                function=input_function_mm)
fea_mm.add_state(name=state_name_mm,
                function=state_function_mm,
                residual_form=residual_form_mm,
                arguments=[input_name_mm])


fea_mm.add_output(name=output_name_mm_1,
                type='scalar',
                form=output_form_mm_1,
                arguments=[state_name_mm])
fea_mm.add_output(name=output_name_mm_2,
                type='scalar',
                form=output_form_mm_2,
                arguments=[state_name_mm])
fea_mm.add_output(name=output_name_mm_3,
                type='scalar',
                form=output_form_mm_3,
                arguments=[state_name_mm])
#############################################################

##################### electomagnetic subproblem ######################
fea_em = FEA(mesh)

fea_em.PDE_SOLVER = 'SNES'
fea_em.REPORT = True

# Add input to the PDE problem: the inputs as the previous states

# Add state to the PDE problem:
# states for electromagnetic equation: magnetic potential vector
state_name_em = 'A_z'
state_function_space_em = FunctionSpace(mesh, ('CG', 1))
state_function_em = Function(state_function_space_em)
v_em = TestFunction(state_function_space_em)

residual_form_em = pdeResEM(state_function_em,v_em,state_function_mm,iq,dx,p,s,Hc,vacuum_perm,angle)



# Add output to the PDE problem:
output_name_1 = 'B_influence_eddy_current'
exponent_1 = 2
subdomains_1 = [1,2]
output_form_1 = B_power_form(state_function_em, state_function_mm,
                            exponent_1, dx, subdomains_1)

output_name_2 = 'B_influence_hysteresis'
exponent_2 = 1.76835 # Material parameter for Hiperco 50
subdomains_2 = [1,2]
output_form_2 = B_power_form(state_function_em, state_function_mm,
                            exponent_2, dx, subdomains_2)


'''
3. Define the boundary conditions
'''

############ Strongly enforced boundary conditions #############
ubc_em = Function(state_function_space_em)
ubc_em.vector.set(0.0)
######## new mesh ############
locate_BC1_em = locate_dofs_geometrical((state_function_space_em, state_function_space_em),
                            lambda x: np.isclose(x[0]**2+x[1]**2, 0.0144 ,atol=1e-6))
locate_BC2_em = locate_dofs_geometrical((state_function_space_em, state_function_space_em),
                            lambda x: np.isclose(x[0]**2+x[1]**2, 0.0036 ,atol=1e-6))

locate_BC_list_em = [locate_BC1_em, locate_BC2_em,]

# ######### old mesh ############
# locate_BC1_em = locate_dofs_geometrical((state_function_space_em, state_function_space_em),
#                             lambda x: np.isclose(x[0]**2+x[1]**2, 0.0144 ,atol=1e-6))

# locate_BC_list_em = [locate_BC1_em, ]

fea_em.add_strong_bc(ubc_em, locate_BC_list_em, state_function_space_em)




fea_em.add_input(name=state_name_mm,
                function=state_function_mm)
fea_em.add_state(name=state_name_em,
                function=state_function_em,
                residual_form=residual_form_em,
                arguments=[state_name_mm])
fea_em.add_output(name=output_name_1,
                type='scalar',
                form=output_form_1,
                arguments=[state_name_em,state_name_mm])
fea_em.add_output(name=output_name_2,
                type='scalar',
                form=output_form_2,
                arguments=[state_name_em,state_name_mm])



'''
4. Set up the CSDL model
'''
fea_model = FEAModel(fea=[fea_mm,fea_em])
# fea_model = FEAModel(fea=[fea_mm])

# Case-to-case postprocessor model
model = csdl.Model()
power_loss_model = PowerLossModel()
loss_sum_model = LossSumModel()
model.add(fea_model, name='fea_model', promotes=['*'])
model.add(power_loss_model, name='power_loss_model', promotes=['*'])
model.add(loss_sum_model, name='loss_sum_model', promotes=['*'])
model.create_input("{}".format(input_name_mm),
                            shape=fea_mm.inputs_dict[input_name_mm]['shape'],
                            val=0.0)
model.add_design_variable(input_name_mm)
model.add_objective('loss_sum')

sim = Simulator(model)

########### Generate the N2 diagram #############
# sim.visualize_implementation()


# ########### Test the forward solve ##############
sim[input_name_mm] = input_array
sim.run()

magnetic_flux_density = B(state_function_em, state_function_mm)

print("Actual B_influence_eddy_current", sim[output_name_1])
print("Actual B_influence_hysteresis", sim[output_name_2])
print("Winding area", sim[output_name_mm_1])
print("Magnet area", sim[output_name_mm_2])
print("Steel area", sim[output_name_mm_3])
print("Eddy current loss", sim['eddy_current_loss'])
print("Hysteresis loss", sim['hysteresis_loss'])
print("Loss sum", sim['loss_sum'])

############# Check the derivatives #############
# sim.check_partials(compact_print=True)
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


with XDMFFile(MPI.COMM_WORLD, "solutions/input_"+input_name_mm+".xdmf", "w") as xdmf:
    xdmf.write_mesh(fea_mm.mesh)
    fea_mm.inputs_dict[input_name_mm]['function'].name = input_name_mm
    xdmf.write_function(fea_mm.inputs_dict[input_name_mm]['function'])
with XDMFFile(MPI.COMM_WORLD, "solutions/state_"+state_name_mm+".xdmf", "w") as xdmf:
    xdmf.write_mesh(fea_mm.mesh)
    fea_mm.states_dict[state_name_mm]['function'].name = state_name_mm
    xdmf.write_function(fea_mm.states_dict[state_name_mm]['function'])

move(fea_mm.mesh, state_function_mm)
with XDMFFile(MPI.COMM_WORLD, "solutions/state_"+state_name_em+".xdmf", "w") as xdmf:
    xdmf.write_mesh(fea_em.mesh)
    fea_em.states_dict[state_name_em]['function'].name = state_name_em
    xdmf.write_function(fea_em.states_dict[state_name_em]['function'])

with XDMFFile(MPI.COMM_WORLD, "solutions/magnetic_flux_density.xdmf", "w") as xdmf:
    xdmf.write_mesh(fea_em.mesh)
    magnetic_flux_density.name = "B"
    xdmf.write_function(magnetic_flux_density)
