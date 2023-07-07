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

# CADDEE geometry initialization
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
file_name = 'pegasus.stp'


spatial_rep = sys_rep.spatial_representation
# spatial_rep.import_file(file_name=STP_FILES_FOLDER / file_name)
# spatial_rep.refit_geometry(file_name=STP_FILES_FOLDER / file_name)
spatial_rep.import_file(file_name='./pegasus_wing/'+file_name)
spatial_rep.refit_geometry(file_name='./pegasus_wing/'+file_name)

# Create Components
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing,']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)

tail_primitive_names = list(spatial_rep.get_primitives(search_names=['HT']).keys())
horizontal_stabilizer = LiftingSurface(name='tail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)

fuselage_primitive_names = list(spatial_rep.get_primitives(search_names=['Fuselage']))
fuselage = Component(name='fuselage', spatial_representation=spatial_rep, primitive_names=fuselage_primitive_names)

sys_rep.add_component(wing)
sys_rep.add_component(horizontal_stabilizer)
sys_rep.add_component(fuselage)


# (interactive visualization not available if using docker container)
# from shell_analysis_fenicsx.pyvista_plotter import plotter_3d


# filename = "./pegasus_wing/pegasus_wing.xdmf"
filename = "./pegasus_wing_old/pegasus_6257_quad_SI.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    fenics_mesh = xdmf.read_mesh(name="Grid")
nel = fenics_mesh.topology.index_map(fenics_mesh.topology.dim).size_local
nn = fenics_mesh.topology.index_map(0).size_local

shell_pde = ShellPDE(fenics_mesh)

# print(shell_mesh_coord)

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


# num_spanwise_vlm = 22
# num_chordwise_vlm = 5
# leading_edge = wing.project(am.linspace(am.array([7.5, -13.5, 2.5]),
#                             am.array([7.5, 13.5, 2.5]), num_spanwise_vlm),
#                             direction=am.array([0., 0., -1.]))  # returns MappedArray
# trailing_edge = wing.project(np.linspace(np.array([13., -13.5, 2.5]),
#                             np.array([13., 13.5, 2.5]), num_spanwise_vlm),
#                              direction=np.array([0., 0., -1.]))   # returns MappedArray
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
ha_cruise.set_module_input('time', 3600)
ha_cruise.set_module_input('roll_angle', 0)
ha_cruise.set_module_input('pitch_angle', np.deg2rad(0))
ha_cruise.set_module_input('yaw_angle', 0)
ha_cruise.set_module_input('flight_path_angle', np.deg2rad(0))
ha_cruise.set_module_input('wind_angle', 0)
ha_cruise.set_module_input('observer_location', np.array([0, 0, 1000]))
ha_cruise.set_module_input('altitude', 15240)

cruise_wing_structural_nodal_displacements_mesh = am.vstack((wing_upper_surface_wireframe,
                                                            wing_lower_surface_wireframe))
cruise_wing_aero_nodal_displacements_mesh = cruise_wing_structural_nodal_displacements_mesh
cruise_wing_structural_nodal_force_mesh = cruise_wing_structural_nodal_displacements_mesh
cruise_wing_aero_nodal_force_mesh = cruise_wing_structural_nodal_displacements_mesh

order_u = 3
num_control_points_u = 12
# knots_u_beginning = np.zeros((order_u-1,))
# knots_u_middle = np.linspace(0., 1., num_control_points_u+2)
# knots_u_end = np.ones((order_u-1,))
# knots_u = np.hstack((knots_u_beginning, knots_u_middle, knots_u_end))
order_v = 1
# knots_v = np.array([0., 0.5, 1.])

# [RU] may not need this step for the shell model since it's already on a surface mesh
dummy_b_spline_space = lg.BSplineSpace(name='dummy_b_spline_space',
                                    order=(order_u,order_v), control_points_shape=((num_control_points_u,1)))
dummy_function_space = lg.BSplineSetSpace(name='dummy_space',
                                    spaces={'dummy_b_spline_space': dummy_b_spline_space})

cruise_wing_pressure_coefficients = m3l.Variable(
                                        name='cruise_wing_pressure_coefficients',
                                        shape=(num_control_points_u,3)
                                        )
cruise_wing_pressure = m3l.Function(
                                    name='cruise_wing_pressure',
                                    space=dummy_function_space,
                                    coefficients=cruise_wing_pressure_coefficients
                                    )

cruise_wing_displacement_coefficients = m3l.Variable(
                                name='cruise_wing_displacement_coefficients',
                                shape=(num_control_points_u,3)
                                )
cruise_wing_displacement = m3l.Function(name='cruise_wing_displacement',
                                        space=dummy_function_space,
                                        coefficients=cruise_wing_displacement_coefficients)

### Start defining computational graph ###

cruise_structural_wing_nodal_forces = cruise_wing_pressure(
                                mesh=cruise_wing_structural_nodal_force_mesh)

shell_force_map_model = rmshell.RMShellForces(component=wing,
                                                mesh=shell_mesh,
                                                pde=shell_pde,
                                                shells=shells)
cruise_structural_wing_mesh_forces = shell_force_map_model.evaluate(
                        nodal_forces=cruise_structural_wing_nodal_forces,
                        nodal_forces_mesh=cruise_wing_structural_nodal_force_mesh)

shell_displacements_model = rmshell.RMShell(component=wing,
                                            mesh=shell_mesh,
                                            pde=shell_pde,
                                            shells=shells)
# shell_displacements_model.set_module_input('wing_beamt_cap_in', val=0.005, dv_flag=True, lower=0.001, upper=0.02, scaler=1E3)
# shell_displacements_model.set_module_input('wing_beamt_web_in', val=0.005, dv_flag=True, lower=0.001, upper=0.02, scaler=1E3)

cruise_structural_wing_mesh_displacements, cruise_structural_wing_mesh_rotations, wing_mass = \
                                shell_displacements_model.evaluate(
                                    forces=cruise_structural_wing_mesh_forces)

shell_displacement_map_model = rmshell.RMShellNodalDisplacements(
                                            component=wing,
                                            mesh=shell_mesh,
                                            pde=shell_pde,
                                            shells=shells)
cruise_structural_wing_nodal_displacements = shell_displacement_map_model.evaluate(
            shell_displacements=cruise_structural_wing_mesh_displacements,
            nodal_displacements_mesh=cruise_wing_structural_nodal_displacements_mesh)

cruise_model = m3l.Model()
cruise_model.register_output(cruise_structural_wing_nodal_displacements)
cruise_model.register_output(wing_mass)

# Add cruise m3l model to cruise condition
ha_cruise.add_m3l_model('cruise_model', cruise_model)
# Add design condition to design scenario
design_scenario.add_design_condition(ha_cruise)
# endregion
system_model.add_design_scenario(design_scenario=design_scenario)

testing_csdl_model = caddee.assemble_csdl()
testing_csdl_model.create_input('wing_shell_mesh', wing_shell_mesh.value.reshape((-1,3)))
force_vector = np.zeros((num_control_points_u,3))
force_vector[:,2] = 50000
cruise_wing_forces = testing_csdl_model.create_input(
                                'cruise_wing_pressure_input', val=force_vector)
testing_csdl_model.connect('cruise_wing_pressure_input',
                            'cruise_wing_pressure_evaluation.'+\
                            cruise_wing_pressure_coefficients.name)

#################### end of m3l ########################

sim = Simulator(testing_csdl_model, analytics=True)
sim.run()


# Comparing the solution to the Kirchhoff analytical solution
u_shell = sim['rm_shell_model.wing_shell_displacement']
u_nodal = sim['rm_shell_displacement_map.wing_shell_nodal_displacement']
uZ = u_shell[:,2]
uZ_nodal = u_nodal[:,2]
########## Output: ##########
print("Wing tip deflection (on struture):",max(uZ))
print("Wing tip deflection (on oml):",max(uZ_nodal))
print("  Number of elements = "+str(nel))
print("  Number of vertices = "+str(nn))

########## Visualization: ##############
import vedo

plotter = vedo.Plotter()
wing_shell_mesh_plot = vedo.Points(wing_shell_mesh.value.reshape((-1,3)))
wing_oml_plot = vedo.Points(cruise_wing_structural_nodal_displacements_mesh.value.reshape((-1,3)))
plotter.show([wing_shell_mesh_plot, wing_oml_plot], interactive=True, axes=1)    # Plotting point cloud


plotter = vedo.Plotter()
deformed_wing_shell_mesh_plot = vedo.Points(u_shell+wing_shell_mesh.value.reshape((-1,3)))
deformed_wing_oml = u_nodal+cruise_wing_structural_nodal_displacements_mesh.value.reshape((-1,3))
deformed_wing_oml_plot = vedo.Points(deformed_wing_oml)
plotter.show([deformed_wing_shell_mesh_plot, deformed_wing_oml_plot],
                interactive=True, axes=1)    # Plotting point cloud

spatial_rep.plot_meshes([deformed_wing_oml.reshape(cruise_wing_structural_nodal_displacements_mesh.shape)],
                        mesh_plot_types=['mesh'], primitives=['none'])  # Plotting "framework" solution (using vedo for fitting)
