"""
Structural analysis of the pegasus wing
with the Reissner--Mindlin shell model

-----------------------------------------------------------
Test the integration of m3l and shell model
-----------------------------------------------------------
"""
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from VAST.core.generate_mappings_m3l import VASTNodalForces
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component
from caddee import GEOMETRY_FILES_FOLDER

import numpy as np
from mpi4py import MPI
import dolfinx
from femo.fea.utils_dolfinx import *
import caddee.api as cd
import csdl
from python_csdl_backend import Simulator
import shell_module as rmshell
from shell_pde import ShellPDE

from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import m3l
import lsdo_geo as lg
import array_mapper as am
import meshio
import pickle

# CADDEE geometry initialization
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
file_name = 'pegasus.stp'


spatial_rep = sys_rep.spatial_representation
spatial_rep.import_file(file_name='./pegasus_wing/'+file_name)
spatial_rep.refit_geometry(file_name='./pegasus_wing/'+file_name)

# Create Components
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing,']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)

# make thickness function and function space
wing_spaces = {}
wing_t_spaces = {}
coefficients = {}
thickness_coefficients = {}
t = 0.01 # starting wing thickness control point values

for name in wing_primitive_names:
    primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
    space = lg.BSplineSpace(name=primitive.name,
                            order=(primitive.order_u, primitive.order_v),
                            control_points_shape=primitive.shape,
                            knots=(primitive.knots_u, primitive.knots_v))
    space_t = lg.BSplineSpace(name=primitive.name,
                            order=(primitive.order_u, primitive.order_v),
                            control_points_shape=(primitive.control_points.shape[0], primitive.control_points.shape[1], 1),
                            knots=(primitive.knots_u, primitive.knots_v))
    wing_spaces[name] = space
    wing_t_spaces[name] = space_t
    coefficients[name] = m3l.Variable(name = name.replace(' ', '_').replace(',', '') + '_geo_coefficients', shape = primitive.control_points.shape, value = primitive.control_points)
    thickness_coefficients[name] = m3l.Variable(name = name.replace(' ', '_').replace(',', '') + '_t_coefficients', shape = (primitive.control_points.shape[0], primitive.control_points.shape[1], 1), value = t * np.ones((primitive.control_points.shape[0], primitive.control_points.shape[1], 1)))

wing_space_m3l = m3l.IndexedFunctionSpace(name='wing_space', spaces=wing_spaces)
wing_t_space_m3l = m3l.IndexedFunctionSpace(name='wing_space', spaces=wing_t_spaces)
wing_geo = m3l.IndexedFunction('wing_geo', space=wing_space_m3l, coefficients=coefficients)
wing_thickness = m3l.IndexedFunction('wing_thickness', space = wing_t_space_m3l, coefficients=thickness_coefficients)
# exit()

############################### Generate projection ##############################
# # import mesh
#
# filename = "./pegasus_wing_old/pegasus_6257_quad_SI.xdmf"
# mesh = meshio.read(filename)
# nodes = mesh.points
# nodes += np.tile(np.array([0.,0.,0.5]), (mesh.points.shape[0],1))
# nodes_parametric = []
#
# targets = spatial_rep.get_primitives(wing_primitive_names)
# num_targets = len(targets.keys())
# print(targets.keys())
# projected_points_on_each_target = []
# target_names = []
# # Project all points onto each target
# for target_name in targets.keys():
#     target = targets[target_name]
#     target_projected_points = target.project(points=nodes, properties=['geometry', 'parametric_coordinates'])
#             # properties are not passed in here because we NEED geometry
#     projected_points_on_each_target.append(target_projected_points)
#     target_names.append(target_name)
# num_targets = len(target_names)
# projected_points_on_each_target_numpy = np.zeros(tuple((num_targets,)) + nodes.shape)
#
# for i in range(num_targets):
#         projected_points_on_each_target_numpy[i] = projected_points_on_each_target[i]['geometry'].value
#
# distances = np.linalg.norm(projected_points_on_each_target_numpy - nodes, axis=-1)   # Computes norm across spatial axis
# closest_surfaces_indices = np.argmin(distances, axis=0) # Take argmin across surfaces
# flattened_surface_indices = closest_surfaces_indices.flatten()
#
# for i in range(nodes.shape[0]):
#      target_index = flattened_surface_indices[i]
#      receiving_target_name = target_names[target_index]
#      receiving_target = targets[receiving_target_name]
#      node_parametric_coordinates = projected_points_on_each_target[target_index]['parametric_coordinates']
#      nodes_parametric.append((receiving_target_name, node_parametric_coordinates))
#
# with open('data.pickle', 'wb') as f:
#     pickle.dump(nodes_parametric, f)
#
# exit()
########################################################################################

with open('data.pickle', 'rb') as f:
     nodes_parametric = pickle.load(f)

for i in range(len(nodes_parametric)):
    u_coord = nodes_parametric[i][1][0][i]
    v_coord = nodes_parametric[i][1][1][i]
    coord = np.array([u_coord, v_coord])
    nodes_parametric[i] = (nodes_parametric[i][0], np.reshape(coord, (1,2)))
thickness_nodes = wing_thickness.evaluate(nodes_parametric)

sys_rep.add_component(wing)

# filename = "./pegasus_wing/pegasus_wing.xdmf"
filename = "./pegasus_wing_old/pegasus_6257_quad_SI.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    fenics_mesh = xdmf.read_mesh(name="Grid")
nel = fenics_mesh.topology.index_map(fenics_mesh.topology.dim).size_local
nn = fenics_mesh.topology.index_map(0).size_local

shell_pde = ShellPDE(fenics_mesh)

# Aluminum 7050
E = 6.9E10 # unit: Pa (N/m^2)
nu = 0.327
h = 3E-3 # overall thickness (unit: m)
rho = 2700
f_d = -rho*h*9.81
y_bc = 0.0
semispan = 12.2157

G = E/2/(1+nu)

#### Getting facets of the LEFT and the RIGHT edge  ####
DOLFIN_EPS = 3E-16
def ClampedBoundary(x):
    return np.less_equal(x[1], y_bc)
def RightChar(x):
    return np.greater(x[0], semispan)
fdim = fenics_mesh.topology.dim - 1

ds_1 = createCustomMeasure(fenics_mesh, fdim, ClampedBoundary, measure='ds', tag=100)
dS_1 = createCustomMeasure(fenics_mesh, fdim, ClampedBoundary, measure='dS', tag=100)
dx_2 = createCustomMeasure(fenics_mesh, fdim+1, RightChar, measure='dx', tag=10)

g = Function(shell_pde.W)
with g.vector.localForm() as uloc:
     uloc.set(0.)

###################  m3l ########################

# create the shell dictionaries:
shells = {}
shells['wing_shell'] = {'E': E, 'nu': nu, 'rho': rho,# material properties
                        'dss': ds_1(100), # custom integrator: ds measure
                        'dSS': dS_1(100), # custom integrator: dS measure
                        'dxx': dx_2(10),  # custom integrator: dx measure
                        'g': g}
# Meshes definitions
# Wing VLM Mesh
# [RU] simulating semispan only
num_spanwise_vlm = 11
num_chordwise_vlm = 5
leading_edge = wing.project(np.linspace(np.array([7.5, 0., 2.5]),
                            np.array([7.5, 13.5, 2.5]), num_spanwise_vlm),
                            direction=np.array([0., 0., -1.]))  # returns MappedArray
trailing_edge = wing.project(np.linspace(np.array([13., 0., 2.5]),
                            np.array([13., 13.5, 2.5]), num_spanwise_vlm),
                             direction=np.array([0., 0., -1.]))

chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
wing_upper_surface_wireframe = wing.project(
                            chord_surface.value + np.array([0., 0., 1.5]),
                            direction=np.array([0., 0., -1.]),
                            grid_search_n=25)
wing_lower_surface_wireframe = wing.project(
                            chord_surface.value - np.array([0., 0., 1.5]),
                            direction=np.array([0., 0., 1.]),
                            grid_search_n=25)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe,
                                    wing_lower_surface_wireframe, 1) # this linspace will return average when n=1
# wing_camber_surface = wing_camber_surface.reshape(
#                                     (num_chordwise_vlm, num_spanwise_vlm, 3))
wing_vlm_mesh_name = 'wing_vlm_mesh'
sys_rep.add_output(wing_vlm_mesh_name, wing_camber_surface)
oml_mesh = am.vstack((wing_upper_surface_wireframe, wing_lower_surface_wireframe))
wing_oml_mesh_name = 'wing_oml_mesh'
sys_rep.add_output(wing_oml_mesh_name, oml_mesh)

sys_rep.add_output(name='chord_distribution',
                                    quantity=am.norm(leading_edge-trailing_edge))

# Wing shell Mesh
z_offset = 0.5
wing_shell_mesh = am.MappedArray(input=fenics_mesh.geometry.x + \
                                        np.array([0.,0.,z_offset])).reshape((-1,3))
shell_mesh = rmshell.LinearShellMesh(
                    meshes=dict(
                    wing_shell_mesh=wing_shell_mesh,
                    ))

# [RX] would lead to shape error from CADDEE system_representation_output
# sys_rep.add_output('wing_shell_mesh', wing_shell_mesh)

# design scenario
design_scenario = cd.DesignScenario(name='recon_mission')
# design_scenario.equations_of_motion_csdl = cd.EulerFlatEarth6DoFGenRef

# aircraft condition
# ha_cruise = cd.AircraftCondition(name='high_altitude_cruise',stability_flag=False,dynamic_flag=False,)
ha_cruise = cd.CruiseCondition(name="cruise_1")
ha_cruise.atmosphere_model = cd.SimpleAtmosphereModel()
ha_cruise.set_module_input('mach_number', 0.17, dv_flag=True, lower=0.1, upper=0.3, scaler=1)
ha_cruise.set_module_input(name='range', val=40000)
# ha_cruise.set_module_input('time', 3600)
ha_cruise.set_module_input('roll_angle', 0)
ha_cruise.set_module_input('pitch_angle', np.deg2rad(0))
ha_cruise.set_module_input('yaw_angle', 0)
ha_cruise.set_module_input('flight_path_angle', np.deg2rad(0))
ha_cruise.set_module_input('wind_angle', 0)
ha_cruise.set_module_input('observer_location', np.array([0, 0, 1000]))
ha_cruise.set_module_input('altitude', 15240)

ac_states = ha_cruise.evaluate_ac_states()

cruise_model = m3l.Model()
cruise_model.register_output(ac_states)

### Start defining computational graph ###
vlm_model = VASTFluidSover(
    surface_names=[
        wing_vlm_mesh_name,
    ],
    surface_shapes=[
        (1, ) + wing_camber_surface.evaluate().shape[1:],
    ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake'),
    mesh_unit='m',
    cl0=[0.43, 0]
)
# aero forces and moments
vlm_panel_forces, vlm_force, vlm_moment  = vlm_model.evaluate(ac_states=ac_states)
cruise_model.register_output(vlm_force)
cruise_model.register_output(vlm_moment)

vlm_force_mapping_model = VASTNodalForces(
    surface_names=[
        wing_vlm_mesh_name,
    ],
    surface_shapes=[
        (1, ) + wing_camber_surface.evaluate().shape[1:],
    ],
    initial_meshes=[
        wing_camber_surface,
        ]
)

oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=vlm_panel_forces, nodal_force_meshes=[oml_mesh, oml_mesh])
wing_forces = oml_forces[0]

shell_force_map_model = rmshell.RMShellForces(component=wing,
                                                mesh=shell_mesh,
                                                pde=shell_pde,
                                                shells=shells)
cruise_structural_wing_mesh_forces = shell_force_map_model.evaluate(
                        nodal_forces=wing_forces,
                        nodal_forces_mesh=oml_mesh)

shell_displacements_model = rmshell.RMShell(component=wing,
                                            mesh=shell_mesh,
                                            pde=shell_pde,
                                            shells=shells)

cruise_structural_wing_mesh_displacements, cruise_structural_wing_mesh_rotations, wing_mass = \
                                shell_displacements_model.evaluate(
                                    forces=cruise_structural_wing_mesh_forces,
                                    thicknesses=thickness_nodes)

cruise_model.register_output(cruise_structural_wing_mesh_displacements)
cruise_model.register_output(wing_mass)

# Add cruise m3l model to cruise condition
ha_cruise.add_m3l_model('cruise_model', cruise_model)
# Add design condition to design scenario
design_scenario.add_design_condition(ha_cruise)
# endregion
system_model.add_design_scenario(design_scenario=design_scenario)

testing_csdl_model = caddee.assemble_csdl()


# Wing_0_16
# Wing_0_17
# Wing_0_18
# Wing_0_19
# Wing_0_20
# Wing_0_21
# Wing_0_22 # not found
# Wing_0_23 # not found
# Wing_1_24
# Wing_1_25
# Wing_1_26 # not found
# Wing_1_27
# Wing_1_28 # not found
# Wing_1_29 # not found
# Wing_1_30 # not found
# Wing_1_31 # not found
right_wing_id = [16, 17, 18, 19, 20, 21, 24, 25, 27]
# h_init = 0.001*np.arange(len(right_wing_id))
h_init = 0.01*np.ones(len(right_wing_id))
# h_init[0] = 0.01
i = 0
shape = (625, 1)
for id in right_wing_id:
    surface_id = id
    if surface_id <= 23:
        surface_name = 'Wing_0_'+str(surface_id)
    else:
        surface_name = 'Wing_1_'+str(surface_id)
    h_i = testing_csdl_model.create_input('wing_thickness_'+str(surface_id), val=h_init[i])
    testing_csdl_model.register_output('wing_thickness_surface_'+str(surface_id), csdl.expand(h_i, shape))
    testing_csdl_model.connect('wing_thickness_surface_'+str(surface_id),
                                'system_model.recon_mission.cruise_1.cruise_1.wing_thickness_evaluation.'+\
                                surface_name+'_t_coefficients')
                                # thickness_coefficients[name].name)
    i += 1
#################### end of m3l ########################

sim = Simulator(testing_csdl_model, analytics=True)
sim.run()


# Comparing the solution to the Kirchhoff analytical solution
f_shell = sim['system_model.recon_mission.cruise_1.cruise_1.wing_rm_shell_force_mapping.wing_shell_forces']
f_vlm = sim['system_model.recon_mission.cruise_1.cruise_1.wing_vlm_mesh_vlm_force_mapping_model.wing_vlm_mesh_oml_forces'].reshape((-1,3))
u_shell = sim['system_model.recon_mission.cruise_1.cruise_1.wing_rm_shell_model.rm_shell.disp_extraction_model.wing_shell_displacement']
# u_nodal = sim['wing_rm_shell_displacement_map.wing_shell_nodal_displacement']
uZ = u_shell[:,2]
# uZ_nodal = u_nodal[:,2]

'''
vlm forces: -1879.3736799156954 0.0 -19286.701054888195
shell forces: -1879.3736799156907 0.0 -19286.70105488822
Wing tip deflection (on struture): 0.013594238254103594
Wing total mass (kg): 91.70632079110337
'''


wing_mass = sim['system_model.recon_mission.cruise_1.cruise_1.wing_rm_shell_model.rm_shell.mass_model.mass']
wing_elastic_energy = sim['system_model.recon_mission.cruise_1.cruise_1.wing_rm_shell_model.rm_shell.elastic_energy_model.elastic_energy']
wing_aggregated_stress = sim['system_model.recon_mission.cruise_1.cruise_1.wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress']
wing_von_Mises_stress = sim['system_model.recon_mission.cruise_1.cruise_1.wing_rm_shell_model.rm_shell.von_Mises_stress_model.von_Mises_stress']
########## Output: ##########
print("vlm forces:", sum(f_vlm[:,0]),sum(f_vlm[:,1]),sum(f_vlm[:,2]))
print("shell forces:", sum(f_shell[:,0]),sum(f_shell[:,1]),sum(f_shell[:,2]))
print("Wing tip deflection (on struture):",max(abs(uZ)))
print("Wing total mass (kg):", wing_mass)
print("Wing aggregated von Mises stress (Pa):", wing_aggregated_stress)
print("Wing maximum von Mises stress (Pa):", max(wing_von_Mises_stress))
print("  Number of elements = "+str(nel))
print("  Number of vertices = "+str(nn))


########## sensitivity analysis of the aggregated stress wrt thickness ############
# import timeit
# start = timeit.default_timer()
# aggregated_stress_name = 'system_model.recon_mission.cruise_1.cruise_1.wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress'
# thickness_name = 'system_model.recon_mission.cruise_1.cruise_1.wing_rm_shell_model.rm_shell.solid_model.wing_shell_thicknesses'
# derivative_dict = sim.compute_totals(of=[aggregated_stress_name],
#                                     wrt=[thickness_name])
#
# stop = timeit.default_timer()
# print('time for compute_totals:', stop-start)
# dCdT = derivative_dict[(aggregated_stress_name, thickness_name)]
# dCdT_function = Function(shell_pde.VT)
# dCdT_function.vector.setArray(dCdT)
# path = 'records'
# if MPI.COMM_WORLD.Get_rank() == 0:
#     with XDMFFile(MPI.COMM_WORLD, path+"/gradient_dCdT.xdmf", "w") as xdmf:
#         xdmf.write_mesh(fenics_mesh)
#         xdmf.write_function(dCdT_function)

########## Visualization: ##############
# import vedo
#
# plotter = vedo.Plotter()
# wing_shell_mesh_plot = vedo.Points(wing_shell_mesh.value.reshape((-1,3)))
# wing_oml_plot = vedo.Points(cruise_wing_structural_nodal_displacements_mesh.value.reshape((-1,3)))
# plotter.show([wing_shell_mesh_plot, wing_oml_plot], interactive=True, axes=1)    # Plotting point cloud
#
#
# plotter = vedo.Plotter()
# deformed_wing_shell_mesh_plot = vedo.Points(u_shell+wing_shell_mesh.value.reshape((-1,3)))
# deformed_wing_oml = u_nodal+cruise_wing_structural_nodal_displacements_mesh.value.reshape((-1,3))
# deformed_wing_oml_plot = vedo.Points(deformed_wing_oml)
# plotter.show([deformed_wing_shell_mesh_plot, deformed_wing_oml_plot],
#                 interactive=True, axes=1)    # Plotting point cloud
#
# spatial_rep.plot_meshes([deformed_wing_oml.reshape(cruise_wing_structural_nodal_displacements_mesh.shape)],
#                         mesh_plot_types=['mesh'], primitives=['none'])  # Plotting "framework" solution (using vedo for fitting)
