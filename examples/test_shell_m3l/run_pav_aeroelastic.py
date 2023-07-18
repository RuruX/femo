"""
Structural analysis of the PAV wing
with the Reissner--Mindlin shell model

-----------------------------------------------------------
Test the integration of m3l and shell model
-----------------------------------------------------------
"""
# from VAST.core.vast_solver import VASTFluidSover
# from VAST.core.fluid_problem import FluidProblem
# from VAST.core.generate_mappings_m3l import VASTNodalForces
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component
import caddee.core.primitives.bsplines.bspline_functions as bsf
from caddee.core.caddee_core.system_representation.system_primitive.system_primitive import SystemPrimitive
from caddee.core.caddee_core.system_representation.spatial_representation import SpatialRepresentation
from caddee.core.caddee_core.system_representation.utils.mesh_utils import import_mesh

from caddee import GEOMETRY_FILES_FOLDER

import numpy as np
# from mpi4py import MPI
import caddee.api as cd
import csdl
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import m3l
import lsdo_geo as lg
import array_mapper as am
import pickle
import pathlib

import meshio
fenics = False
if fenics:
    import dolfinx
    from femo.fea.utils_dolfinx import *
    import shell_module as rmshell
    from shell_pde import ShellPDE


# CADDEE geometry initialization
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
spatial_rep = sys_rep.spatial_representation


## Generate geometry

# import initial geomrty
file_name = '/pav_wing/pav_wing.stp'
cfile = str(pathlib.Path(__file__).parent.resolve())
spatial_rep.import_file(file_name=cfile+file_name)
spatial_rep.refit_geometry(file_name=cfile+file_name)

# Manual surface identification
if False:
    for key in spatial_rep.primitives.keys():
        surfaces = list(spatial_rep.primitives.keys())
        surfaces.remove(key)
        print(key)
        spatial_rep.plot(primitives=surfaces)

# make wing components
left_wing_names = []
left_wing_top_names = []
left_wing_bottom_names = []
for i in range(14,22):
    surf_name = 'FrontWing, 1, ' + str(i)
    left_wing_names.append(surf_name)
    if i%2 == 0:
        left_wing_bottom_names.append(surf_name)
    else:
        left_wing_top_names.append(surf_name)

wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=list(spatial_rep.primitives.keys()))
wing_left = LiftingSurface(name='wing_left', spatial_representation=spatial_rep, primitive_names=left_wing_names)
wing_left_top = LiftingSurface(name='wing_left_top', spatial_representation=spatial_rep, primitive_names=left_wing_top_names)
wing_left_bottom = LiftingSurface(name='wing_left_bottom', spatial_representation=spatial_rep, primitive_names=left_wing_bottom_names)

sys_rep.add_component(wing)

structural_left_wing_names = left_wing_names.copy()

# projections for internal structure
do_plots = False
num_pts = 10
spar_rib_spacing_ratio = 3
num_rib_pts = 20

# Important points from openVSP
root_te = np.array([17.030, 0., 2.365])
root_le = np.array([10.278, 0, 2.719])
tip_te = np.array([13.276, -17.596, 2.5])
tip_le = np.array([10.261, -17.596, 2.5])

root_25 = (3*root_le+root_te)/4
root_75 = (root_le+3*root_te)/4
tip_25 = (3*tip_le+tip_te)/4
tip_75 = (tip_le+3*tip_te)/4

avg_spar_spacing = (np.linalg.norm(root_25-root_75)+np.linalg.norm(tip_25-tip_75))/2
half_span = root_le[1] - tip_le[1]
num_ribs = int(spar_rib_spacing_ratio*half_span/avg_spar_spacing)+1

f_spar_projection_points = np.linspace(root_25, tip_25, num_ribs)
r_spar_projection_points = np.linspace(root_75, tip_75, num_ribs)

rib_projection_points = np.linspace(f_spar_projection_points, r_spar_projection_points, num_rib_pts)

f_spar_top = wing_left_top.project(f_spar_projection_points, plot=do_plots)
f_spar_bottom = wing_left_bottom.project(f_spar_projection_points, plot=do_plots)

r_spar_top = wing_left_top.project(r_spar_projection_points, plot=do_plots)
r_spar_bottom = wing_left_bottom.project(r_spar_projection_points, plot=do_plots)

ribs_top = wing_left_top.project(rib_projection_points, direction=[0.,0.,1.], plot=do_plots, grid_search_n=100)
ribs_bottom = wing_left_bottom.project(rib_projection_points, direction=[0.,0.,1.], plot=do_plots, grid_search_n=100)


# make monolithic spars - makes gaps in internal structure, not good
if False:
    n_cp_spar = (num_ribs,2)
    order_spar = (2,)

    f_spar_points = np.zeros((num_ribs, 2, 3))
    f_spar_points[:,0,:] = f_spar_top.value
    f_spar_points[:,1,:] = f_spar_bottom.value
    f_spar_bspline = bsf.fit_bspline(f_spar_points, num_control_points=n_cp_spar, order = order_spar)
    f_spar = SystemPrimitive('front_spar', f_spar_bspline)
    spatial_rep.primitives[f_spar.name] = f_spar

    r_spar_points = np.zeros((num_ribs, 2, 3))
    r_spar_points[:,0,:] = r_spar_top.value
    r_spar_points[:,1,:] = r_spar_bottom.value
    r_spar_bspline = bsf.fit_bspline(r_spar_points, num_control_points=n_cp_spar, order = order_spar)
    r_spar = SystemPrimitive('rear_spar', r_spar_bspline)
    spatial_rep.primitives[r_spar.name] = r_spar

# make multi-patch spars - for coherence
n_cp = (2,2)
order = (2,)

for i in range(num_ribs-1):
    f_spar_points = np.zeros((2,2,3))
    f_spar_points[0,:,:] = ribs_top.value[0,(i,i+1),:]
    f_spar_points[1,:,:] = ribs_bottom.value[0,(i,i+1),:]
    f_spar_bspline = bsf.fit_bspline(f_spar_points, num_control_points=n_cp, order=order)
    f_spar = SystemPrimitive('f_spar_' + str(i), f_spar_bspline)
    spatial_rep.primitives[f_spar.name] = f_spar
    structural_left_wing_names.append(f_spar.name)

    r_spar_points = np.zeros((2,2,3))
    r_spar_points[0,:,:] = ribs_top.value[-1,(i,i+1),:]
    r_spar_points[1,:,:] = ribs_bottom.value[-1,(i,i+1),:]
    r_spar_bspline = bsf.fit_bspline(r_spar_points, num_control_points=n_cp, order=order)
    r_spar = SystemPrimitive('r_spar_' + str(i), r_spar_bspline)
    spatial_rep.primitives[r_spar.name] = r_spar
    structural_left_wing_names.append(r_spar.name)


# make ribs
n_cp_rib = (num_rib_pts,2)
order_rib = (2,)

for i in range(num_ribs):
    rib_points = np.zeros((num_rib_pts, 2, 3))
    rib_points[:,0,:] = ribs_top.value[:,i,:]
    rib_points[:,1,:] = ribs_bottom.value[:,i,:]
    rib_bspline = bsf.fit_bspline(rib_points, num_control_points=n_cp_rib, order = order_rib)
    rib = SystemPrimitive('rib_' + str(i), rib_bspline)
    spatial_rep.primitives[rib.name] = rib
    structural_left_wing_names.append(rib.name)

spatial_rep.assemble()

if do_plots:
    spatial_rep.plot(plot_types=['wireframe'])


# structural wing component:
wing_left_structural = LiftingSurface(name='wing_left_structural', 
                                      spatial_representation=spatial_rep, 
                                      primitive_names = structural_left_wing_names)
sys_rep.add_component(wing_left_structural)

## make additional geometry for meshing
write_geometry = False

mesh_spatial_rep = SpatialRepresentation()
# mesh_spatial_rep.primitives[f_spar.name] = f_spar
# mesh_spatial_rep.primitives[r_spar.name] = r_spar

# add ribs
for i in range(num_ribs):
    name = 'rib_' + str(i)
    mesh_spatial_rep.primitives[name] = spatial_rep.primitives[name]

# add spars
for i in range(num_ribs-1):
    name = 'f_spar_' + str(i)
    mesh_spatial_rep.primitives[name] = spatial_rep.primitives[name]
    name = 'r_spar_' + str(i)
    mesh_spatial_rep.primitives[name] = spatial_rep.primitives[name]

# make surface panels
n_cp = (num_rib_pts,2)
order = (2,)

for i in range(num_ribs-1):
    t_panel_points = ribs_top.value[:,(i,i+1),:]
    t_panel_bspline = bsf.fit_bspline(t_panel_points, num_control_points=n_cp, order=order)
    t_panel = SystemPrimitive('t_panel_' + str(i), t_panel_bspline)
    mesh_spatial_rep.primitives[t_panel.name] = t_panel

    b_panel_points = ribs_bottom.value[:,(i,i+1),:]
    b_panel_bspline = bsf.fit_bspline(b_panel_points, num_control_points=n_cp, order=order)
    b_panel = SystemPrimitive('b_panel_' + str(i), b_panel_bspline)
    mesh_spatial_rep.primitives[b_panel.name] = b_panel

mesh_spatial_rep.assemble()
if do_plots:
    mesh_spatial_rep.plot(plot_types=['wireframe'])

if write_geometry:
    mesh_spatial_rep.write_iges(cfile + '/pav_wing/pav_wing_structue.iges')
## At this point, use gmsh to generate a mesh using ^this^ .iges file

#### Meshing for shell solver

# This block imports the raw gmsh output, cleans it up, and exports the new mesh.
# This takes a ~15 minutes the first time
process_gmsh = False
run_reprojection = False

if process_gmsh or run_reprojection:
    file = '/pav_wing/pav_gmsh_2880.msh'
    nodes, connectivity = import_mesh(cfile + file, 
                                    spatial_rep,
                                    component = wing_left_structural,
                                    targets = list(wing_left_structural.get_primitives().values()),
                                    rescale=1e-3,
                                    remove_dupes=True, 
                                    optimize_projection=False, 
                                    tol=1e-8, 
                                    plot=do_plots,
                                    grid_search_n=100)

    cells = [("quad",connectivity)]
    mesh = meshio.Mesh(nodes.value, cells)
    meshio.write(cfile + '/pav_wing/pav_wing_caddee_mesh_' + str(nodes.shape[0]) + '_quad.msh', mesh, file_format='gmsh')

## make thickness function and function space
wing_spaces = {}
wing_t_spaces = {}
coefficients = {}
thickness_coefficients = {}
t = 0.005 # starting wing thickness control point values
for name in structural_left_wing_names:
    primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
    name = name.replace(' ', '_').replace(',','')
    space = lg.BSplineSpace(name=name,
                            order=(primitive.order_u, primitive.order_v),
                            control_points_shape=primitive.shape,
                            knots=(primitive.knots_u, primitive.knots_v))
    # print((primitive.control_points.shape[0], primitive.control_points.shape[1], 1))
    # print(name + '_t_coefficients')
    space_t = lg.BSplineSpace(name=name,
                            order=(primitive.order_u, primitive.order_v),
                            control_points_shape=(primitive.control_points.shape[0], primitive.control_points.shape[1], 1),
                            knots=(primitive.knots_u, primitive.knots_v))
    wing_spaces[name] = space
    wing_t_spaces[name] = space_t
    coefficients[name] = m3l.Variable(name = name + '_geo_coefficients', shape = primitive.control_points.shape, value = primitive.control_points)
    thickness_coefficients[name] = m3l.Variable(name = name + '_t_coefficients', shape = (primitive.control_points.shape[0], primitive.control_points.shape[1], 1), value = t * np.ones((primitive.control_points.shape[0], primitive.control_points.shape[1], 1)))

wing_space_m3l = m3l.IndexedFunctionSpace(name='wing_space', spaces=wing_spaces)
wing_t_space_m3l = m3l.IndexedFunctionSpace(name='wing_space', spaces=wing_t_spaces)
wing_geo = m3l.IndexedFunction('wing_geo', space=wing_space_m3l, coefficients=coefficients)
wing_thickness = m3l.IndexedFunction('wing_thickness', space = wing_t_space_m3l, coefficients=thickness_coefficients)

# make thickness mesh as m3l variable
# this will take ~4 minutes

if run_reprojection:
    file_name = '/pav_wing/pav_wing_mesh_data.pickle'

    nodes_parametric = []

    targets = spatial_rep.get_primitives(structural_left_wing_names)
    num_targets = len(targets.keys())

    projected_points_on_each_target = []
    target_names = []
    # Project all points onto each target           
    for target_name in targets.keys():
        target = targets[target_name]
        target_projected_points = target.project(points=nodes, properties=['geometry', 'parametric_coordinates'])
                # properties are not passed in here because we NEED geometry
        projected_points_on_each_target.append(target_projected_points)
        target_names.append(target_name)
    num_targets = len(target_names)
    distances = np.zeros(tuple((num_targets,)) + (nodes.shape[0],))
    for i in range(num_targets):
            distances[i,:] = np.linalg.norm(projected_points_on_each_target[i]['geometry'].value - nodes.value, axis=-1)
    closest_surfaces_indices = np.argmin(distances, axis=0) # Take argmin across surfaces
    flattened_surface_indices = closest_surfaces_indices.flatten()
    for i in range(nodes.shape[0]):
        target_index = flattened_surface_indices[i]
        receiving_target_name = target_names[target_index]
        receiving_target = targets[receiving_target_name]
        u_coord = projected_points_on_each_target[target_index]['parametric_coordinates'][0][i]
        v_coord = projected_points_on_each_target[target_index]['parametric_coordinates'][1][i]
        node_parametric_coordinates = np.array([u_coord, v_coord])
        nodes_parametric.append((receiving_target_name, node_parametric_coordinates))
    with open(cfile + file_name, 'wb') as f:
        pickle.dump(nodes_parametric, f)

with open(cfile + '/pav_wing/pav_wing_mesh_data.pickle', 'rb') as f:
    nodes_parametric = pickle.load(f)

for i in range(len(nodes_parametric)):
    nodes_parametric[i] = (nodes_parametric[i][0].replace(' ', '_').replace(',',''), nodes_parametric[i][1])


thickness_nodes = wing_thickness.evaluate(nodes_parametric)

exit()

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
    mesh_unit='ft',
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

system_model.add_design_scenario(design_scenario=design_scenario)

testing_csdl_model = caddee.assemble_csdl()

h_init = 0.01
i = 0
for name in wing_primitive_names:
    primitive = spatial_rep.get_primitives([name])[name].geometry_primitive

    surface_id = i
    shape = (primitive.control_points.shape[0], primitive.control_points.shape[1], 1)
    testing_csdl_model.create_input('wing_thickness'+str(surface_id), val=h_init*np.ones(shape))
    # testing_csdl_model.connect('wing_thickness'+str(surface_id),
    #                             'wing_thickness_evaluation.'+\
    #                             thickness_coefficients[name].name)
    i += 1

#################### end of m3l ########################

sim = Simulator(testing_csdl_model, analytics=True)
sim.run()


# Comparing the solution to the Kirchhoff analytical solution
f_shell = sim['system_model.recon_mission.cruise_1.cruise_1.wing_rm_shell_force_mapping.wing_shell_forces']
u_shell = sim['system_model.recon_mission.cruise_1.cruise_1.wing_rm_shell_model.rm_shell.disp_extraction_model.wing_shell_displacement']
# u_nodal = sim['wing_rm_shell_displacement_map.wing_shell_nodal_displacement']
uZ = u_shell[:,2]
# uZ_nodal = u_nodal[:,2]
########## Output: ##########
print("Wing tip deflection (on struture):",max(uZ))
# print("Wing tip deflection (on oml):",max(uZ_nodal))
print("  Number of elements = "+str(nel))
print("  Number of vertices = "+str(nn))

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
