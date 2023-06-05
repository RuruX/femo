"""
Aeroelasticity analysis on an eVTOL wing model by VPM + FEniCSx
https://hangar.openvsp.org/vspfiles/522
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Nsteps',dest='Nsteps',default='5',
                    help='Number of time steps')
parser.add_argument('--add-rotors',dest='add_rotors',default=True,
                    action=argparse.BooleanOptionalAction,
                    help='Simulation with and without rotors')
args = parser.parse_args()
Nsteps = int(args.Nsteps)
add_rotors = args.add_rotors
# print("1. Add_rotors = ", add_rotors, type(add_rotors))
flowpy_path = './restart/'

if add_rotors:
    print("Reading restart file with rotors ...")
    f_restart_file_name = 'a6_dt0.001_n20_nr20_rotorstrue_pps5_pfield.300.h5'
else:
    print("Reading restart file without rotors ...")
    f_restart_file_name = 'a6_dt0.001_n20_nr10_rotorsfalse_pps5_pfield.200.h5'


from femo.fea.fea_dolfinx import *
from femo.csdl_opt.fea_model import FEAModel
from femo.csdl_opt.state_model import StateModel
from femo.csdl_opt.output_model import OutputModel
import numpy as np
import csdl
from csdl import Model
from csdl_om import Simulator as om_simulator
from python_csdl_backend import Simulator as py_simulator
from matplotlib import pyplot as plt
import argparse
from mpi4py import MPI
from shell_analysis_fenicsx import *
from shell_analysis_fenicsx import solveNonlinear as solveShell

from FSI_coupling.VLM_sim_handling import *
from FSI_coupling.shellmodule_utils import *
from FSI_coupling.NodalMapping import *
from FSI_coupling.VPM_sim_handling import *

import cProfile, pstats, io
from timeit import default_timer

def profile(filename=None, comm=MPI.COMM_WORLD):
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

##########################################################################
######################## Structural inputs ###############################
##########################################################################
tri_mesh = [
            "eVTOL_wing_half_tri_77020_103680.xdmf", # error
            "eVTOL_wing_half_tri_81475_109456.xdmf", # error
            "eVTOL_wing_half_tri_107695_136686.xdmf",
            "eVTOL_wing_half_tri_135957_170304.xdmf"] # error

test = 2
s_mesh_file_name = tri_mesh[test]
mesh_path = "./evtol_wing/"
solid_mesh_file = mesh_path + s_mesh_file_name

with XDMFFile(MPI.COMM_WORLD, solid_mesh_file, "r") as xdmf:
       mesh = xdmf.read_mesh(name="Grid")
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

# define structural properties
E = 6.8E10 # unit: Pa (N/m^2)
nu = 0.35
h = 3E-3 # overall thickness (unit: m)
rho_struct = 2710. # unit: kg/m^3

element_type = "CG2CG1" # with quad/tri elements

element = ShellElement(
                mesh,
                element_type,
                inplane_deg=3,
                shear_deg=3
                )
dx_inplane, dx_shear = element.dx_inplane, element.dx_shear


def pdeRes(h,w,E,f,CLT,dx_inplane,dx_shear,rho,uddot,thetaddot):
    w_temp = Function(w.function_space)
    dw = TestFunction(w.function_space)
    elastic_model = DynamicElasticModel(mesh, w_temp, CLT)
    elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane, dx_shear)
    ALPHA = 1
    dWint = elastic_model.weakFormResidual(ALPHA, elastic_energy, w_temp, dw, f)
    dWint_mid = ufl.replace(dWint, {w_temp: w_mid})
    # Inertial contribution to the residual:
    dWmass = elastic_model.inertialResidual(rho,h,uddot,thetaddot)
    F = dWmass + dWint_mid
    return F

def compliance(u_mid,h):
    h_mesh = CellDiameter(mesh)
    alpha = 1e-1
    dX = ufl.Measure('dx', domain=mesh, metadata={"quadrature_degree":0})
    return 0.5*dot(u_mid,u_mid)*dX \
            + 0.5*alpha*dot(grad(h), grad(h))*(h_mesh**2)*dX

def volume(h):
    return h*dx

def extractTipDisp(u):
    x_tip = [0.307515,5.31806,0.541493]
    cell_tip = 115587
    return u.eval(x_tip, cell_tip)[-1]


# path for saving the results
path = "test_"+str(Nsteps)

###########################################################################
########################## Set up time domain #############################
###########################################################################

dt_shell = 0.01
inner_steps = 10
dt_vpm = dt_shell/inner_steps # consistent time step size
T_shell = Nsteps * dt_shell
T = T_shell + 3*dt_vpm # 3 more VPM iterations for initialization


###########################################################################
######################## Aerodynamic inputs ###############################
###########################################################################

# define VPM input parameters
V_inf = 50.  # freestream velocity magnitude in m/s
AoA = 6.  # Angle of Attack in degrees
# AoA = 0.  # Angle of Attack in degrees
AoA_rad = np.deg2rad(AoA)  # Angle of Attack converted to radians
rho = 1.225  # International Standard Atmosphere air density at sea level in kg/m^3

################### Import Aerodynamic mesh ###################
print("Constructing aerodynamic mesh and mesh mappings...")
# Import a preconstructed VLM mesh
# Ru: need to add a restart file to achieve steady state solution of the wing


VPM_sim = VPM_sim_handling(V_inf, AoA_rad, duration=T, dt=dt_vpm,
                            restart_file=flowpy_path+f_restart_file_name,
                            run_name = path+"_vpm",
                            add_rotors=add_rotors)

# VPM_sim = VPM_sim_handling(V_inf, AoA_rad, duration=T, dt=dt_vpm,
#                             restart_file=None, add_rotors=True)
VPM_mesh_baseline = translate_mesh(VPM_sim.mesh,
                        np.array([VPM_sim.mesh[:, :, 0].min() \
                                - mesh.geometry.x[:, 0].min(), 0., 0.]))
VPM_mesh_starboard = VPM_mesh_baseline[int(np.floor(VPM_mesh_baseline.shape[0]/2)):, :, :]
VPM_mesh_list_baseline = construct_vpm_geometry_input_list_format(
                                        VPM_mesh_baseline)

########## Define force functions and aero-elastic coupling object ############
coupling_obj = FEniCSx_vortexmethod_coupling(mesh, VPM_mesh_starboard)
f_dist_solid = Function(coupling_obj.solid_aero_force_space)
f_nodal_solid = Function(coupling_obj.solid_aero_force_space)

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
##################################
u, theta = split(state_function)
dw = TestFunction(state_function_space)
du_mid,dtheta = split(dw)

# Quantities from the previous time step
w_old = Function(state_function_space)
u_old, theta_old = split(w_old)
wdot_old = Function(state_function_space)
udot_old, thetadot_old = split(wdot_old)

# Set up the time integration scheme
def implicitMidpointRule(u, u_old, udot_old, dt):
    u_mid = Constant(mesh, 0.5)*(u_old+u)
    udot = Constant(mesh, 2/dt)*u - Constant(mesh, 2/dt)*u_old - udot_old
    uddot = (udot - udot_old)/dt
    return u_mid, udot, uddot
def wdot_vector(w, w_old, wdot_old, dt):
    return 2/dt*w.vector - 2/dt*w_old.vector - wdot_old.vector

u_mid, udot, uddot = implicitMidpointRule(u, u_old, udot_old, dt_shell)
theta_mid, thetadot, thetaddot = implicitMidpointRule(theta, theta_old, thetadot_old, dt_shell)
w_mid = Constant(mesh, 0.5)*(w_old+state_function)
##################################
material_model = MaterialModel(E=E,nu=nu,h=input_function) # Simple isotropic material

def assembleStrainEnergy(w):
    elastic_model = ElasticModel(mesh, w, material_model.CLT)
    elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane, dx_shear)
    return assemble_scalar(form(elastic_energy))

residual_form = pdeRes(input_function,state_function,
                        E,f_dist_solid,material_model.CLT,dx_inplane,dx_shear,
                        rho_struct, uddot, thetaddot)

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

############ Set the BCs for the airplane model ###################

locate_BC1 = locate_dofs_geometrical((state_function_space.sub(0), state_function_space.sub(0).collapse()[0]),
                                    lambda x: np.less(x[1], 0.55))
locate_BC2 = locate_dofs_geometrical((state_function_space.sub(1), state_function_space.sub(1).collapse()[0]),
                                    lambda x: np.less(x[1], 0.55))
ubc = Function(state_function_space)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)
############ Strongly enforced boundary conditions #############
fea.add_strong_bc(ubc, [locate_BC1], state_function_space.sub(0))
fea.add_strong_bc(ubc, [locate_BC2], state_function_space.sub(1))

xdmf_file = XDMFFile(comm, path+"/u_mid.xdmf", "w")
xdmf_file.write_mesh(mesh)
xdmf_file_aero_f = XDMFFile(comm, path+"/aero_F.xdmf", "w")
xdmf_file_aero_f.write_mesh(mesh)
xdmf_file_aero_f_nodal = XDMFFile(comm, path+"/aero_F_nodal.xdmf", "w")
xdmf_file_aero_f_nodal.write_mesh(mesh)
state_function.sub(0).name = 'u_mid'
state_function.sub(1).name = 'theta'
uZ_tip_record = np.zeros(Nsteps)
strain_energy_record = np.zeros(Nsteps)
x_tip = [0.307515,5.31806,0.541493]
cell_tip = 115587

########### Test the forward solve ##############
# @profile(filename="profile_out_"+str(Nsteps))
# def main():
def solveDynamicAeroelasticity(res,func,bc,report=False):
################# Dynamic aerostructural coupling solve ####################\
    t = 0.0
    ndt = 0 # nondimentional time
    # do 3 steps to initialize data outputs
    for k in range(3):
        ndt += 1
        VPM_sim.step_vpm_sim()
    ########## Start of iteration loop for coupled solution procedure ##########
    # declare initial values for variables that will be updated during each iteration
    VPM_mesh_displaced_previous = deepcopy(VPM_mesh_starboard)
    VPM_mesh_list_step = compute_differences_between_VPM_input_lists(VPM_mesh_list_baseline, VPM_mesh_list_baseline)
    VPM_mesh_list_previous = deepcopy(VPM_mesh_list_baseline)

    for i in range(0, Nsteps):
        t += dt_shell
        iter = i
        print("------- Time step "+str(i+1)+"/"+str(Nsteps)
                +" , t = "+str(t)+", nondimentional time = "+str(ndt)+" -------")

        # Solve the nonlinear problem for this time step and put the solution
        # (in homogeneous coordinates) in y_hom.

        print("Running VPM sim...")
        ########## Update VPM mesh with deformation and run VPM sim: ##########
        VPM_sim.apply_displacement_step_to_vpm(VPM_mesh_list_step)

        for j in range(inner_steps):
            ndt += 1
            VPM_sim.step_vpm_sim()

        # extract panel forces from VLM simulation
        panel_forces = VPM_sim.output_force_vectors()
        panel_forces_starboard = panel_forces[int(np.floor(panel_forces.shape[0]/2)):]

        print("Total starboard aero force components: {}".format(list(np.sum(panel_forces_starboard, axis=0))))


        ########## Project VLM panel forces to solid CG1 space: ##########

        # compute and set distributed vlm load
        F_dist_solid = coupling_obj.compute_dist_solid_force_from_vlm(panel_forces_starboard)
        f_dist_solid.vector.setArray(F_dist_solid)

        print("Total aero force projected to solid: {}".format([assemble_scalar(form(f_dist_solid[i]*dx)) for i in range(3)]))

        ########## Solve with Newton solver wrapper: ##########
        print("Running solid shell sim...")
        # the PDE residual will be automaticly updated with the updated f_dist_solid
        # solveShell(residual_form,state_function,fea.bc,log=True)
        solveShell(res,func,bc,log=True)
        ########## Update displacements: ##########
        print("Updating displacements...")
        # compute VLM mesh displacement from the solid solution vector
        VPM_disp_vec = coupling_obj.compute_vlm_displacement_from_solid(func)

        # add current displacement to previous VPM mesh
        VPM_mesh_displaced = np.add(reshape_3D_array_to_2D(VPM_mesh_displaced_previous), VPM_disp_vec)

        # create mirrored displaced mesh and use that to create the VPM geometry input list
        VPM_mesh_displaced_mirrored = mirror_mesh_around_y_axis(reshape_2D_array_to_3D(VPM_mesh_displaced, VPM_mesh_starboard.shape))
        VPM_mesh_list_displaced = construct_vpm_geometry_input_list_format(VPM_mesh_displaced_mirrored)

        # compute the difference of the VPM geometry input list w.r.t. the previous geometry
        VPM_mesh_list_step = compute_differences_between_VPM_input_lists(VPM_mesh_list_previous, VPM_mesh_list_displaced)

        # update variables for next loop
        VPM_mesh_displaced_previous = reshape_2D_array_to_3D(VPM_mesh_displaced, VPM_mesh_starboard.shape)
        VPM_mesh_list_previous = deepcopy(VPM_mesh_list_displaced)

        print("Tip deflection:", extractTipDisp(func.sub(0)))
        # compute the aeroelastic work in the aerodynamic sim
        W_a = np.sum(np.diag(VPM_disp_vec.T@coupling_obj.P_map@panel_forces_starboard))

        # compute the aeroelastic work in the solid sim
        extracted_solid_disp = coupling_obj.extract_cg2_displacement_vector(func)
        cg1_disp = coupling_obj.Q_map@extracted_solid_disp
        W_s = np.reshape(cg1_disp, (cg1_disp.shape[0]*cg1_disp.shape[1]), order='C')@coupling_obj.Mat_f_sp@f_dist_solid.vector.getArray()

        # report the aeroelastic work in the fluid and solid sims
        print("Work in aerodynamic sim: {}".format(W_a))
        print("Work in solid sim: {}".format(W_s))

        # map VLM panel forces to solid nodal forces
        F_nodal_solid = coupling_obj.compute_nodal_solid_force_from_vlm(panel_forces_starboard)
        # set nodal pressure load (just for plotting purposes)
        f_nodal_solid.vector.setArray(F_nodal_solid)

        # Advance to the next time step
        # ** since u_dot, theta_dot are not functions, we cannot directly
        # ** interpolate them onto wdot_old.
        wdot_old.vector[:] = wdot_vector(func,w_old,wdot_old, dt_shell)
        w_old.interpolate(func)
        uZ_tip_record[i] = extractTipDisp(func.sub(0))
        strain_energy_record[i] = assembleStrainEnergy(func)
        print("Tip displacement: {}".format(uZ_tip_record[i]))
        print("Strain energy: {}".format(strain_energy_record[i]))
        # Save solution to XDMF format
        xdmf_file.write_function(func.sub(0), ndt)
        xdmf_file_aero_f.write_function(f_dist_solid, ndt)
        xdmf_file_aero_f_nodal.write_function(f_nodal_solid, ndt)

    ndt += 1
    VPM_sim.step_vpm_sim()
    print("Saving vpmdata history.h5...")
    fp.write_history_h5(path+"_vpm_history.h5", VPM_sim.vpmdata)


# cProfile.run('main()', "profile_out_"+str(Nsteps))

fea.custom_solve = solveDynamicAeroelasticity


'''
4. Set up the CSDL model
'''


fea.PDE_SOLVER = 'Newton'
# fea.REPORT = True
fea_model = FEAModel(fea=[fea])
fea_model.create_input("{}".format(input_name),
                            shape=fea.inputs_dict[input_name]['shape'],
                            val=h*np.ones(fea.inputs_dict[input_name]['shape']))

# fea_model.add_design_variable(input_name)
# fea_model.add_objective(output_name)

sim = py_simulator(fea_model)
# sim = om_simulator(fea_model)

# ########### Test the forward solve ##############
# @profile(filename="profile_out_"+str(Nsteps))
# def main(sim):
#     sim.run()
# cProfile.run('main(sim)', "profile_out_"+str(Nsteps))

sim.run()
########## Output: ##############
dofs = len(state_function.vector.getArray())
uZ = computeNodalDisp(state_function.sub(0))[2]
print("-"*50)
print("-"*50)
print("-"*8, s_mesh_file_name, "-"*9)
print("-"*50)
# print("Tip deflection:", max(uZ))
print("Tip deflection:", uZ_tip_record[-1])
print("  Number of elements = "+str(nel))
print("  Number of vertices = "+str(nn))
print("  Number of total dofs = ", dofs)
print("-"*50)
cl_history = VPM_sim.vpmdata.data_history["cl"]
cd_history = VPM_sim.vpmdata.data_history["cd"]
CL_history = VPM_sim.vpmdata.data_history["CL"]
CD_history = VPM_sim.vpmdata.data_history["CD"]
np.savetxt(path+'/cl_'+str(Nsteps)+'.out', cl_history, delimiter=',')
np.savetxt(path+'/cd_'+str(Nsteps)+'.out', cd_history, delimiter=',')
np.savetxt(path+'/CL_'+str(Nsteps)+'.out', CL_history, delimiter=',')
np.savetxt(path+'/CD_'+str(Nsteps)+'.out', CD_history, delimiter=',')

np.savetxt(path+'/tip_disp_'+str(Nsteps)+'.out', uZ_tip_record, delimiter=',')
np.savetxt(path+'/strain_energy_'+str(Nsteps)+'.out', strain_energy_record, delimiter=',')
