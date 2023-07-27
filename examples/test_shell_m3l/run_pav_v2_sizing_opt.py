"""
Structural sizing optimization of the PAV wing
Structure: the Reissner--Mindlin shell model
Aerodynamics: Vortex Lattice Method (rigid aerodynamic loads)

"""
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from VAST.core.generate_mappings_m3l import VASTNodalForces
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component
import caddee.core.primitives.bsplines.bspline_functions as bsf
from caddee.core.caddee_core.system_representation.system_primitive.system_primitive import SystemPrimitive
from caddee.core.caddee_core.system_representation.spatial_representation import SpatialRepresentation
from caddee.core.caddee_core.system_representation.utils.mesh_utils import import_mesh as caddee_import_mesh

from caddee import GEOMETRY_FILES_FOLDER

import numpy as np
from mpi4py import MPI
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
from caddee.utils.aircraft_models.pav.pav_weight import PavMassProperties
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem

import meshio
fenics = True
if fenics:
    import dolfinx
    from femo.fea.utils_dolfinx import *
    import shell_module as rmshell
    from shell_pde import ShellPDE


in2m = 0.0254
ft2m = 0.3048
lbs2kg = 0.453592
psf2pa = 50

debug_geom_flag = False
visualize_flag = False
do_plots = False
force_reprojection = False

# region Meshes
plots_flag = False

# CADDEE geometry initialization
caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)
spatial_rep = sys_rep.spatial_representation


## Generate geometry

# import initial geomrty
# file_name = '/pav_wing/pav.stp'
file_name = '/pav_wing/pav_SI.stp'
cfile = str(pathlib.Path(__file__).parent.resolve())
spatial_rep.import_file(file_name=cfile+file_name)
spatial_rep.refit_geometry(file_name=cfile+file_name)

# fix naming
primitives_new = {}
indicies_new = {}
for key, item in spatial_rep.primitives.items():
    item.name = item.name.replace(' ','_').replace(',','')
    primitives_new[key.replace(' ','_').replace(',','')] = item

for key, item in spatial_rep.primitive_indices.items():
    indicies_new[key.replace(' ','_').replace(',','')] = item

spatial_rep.primitives = primitives_new
spatial_rep.primitive_indices = indicies_new

wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())

# Manual surface identification
if False:
    for key in wing_primitive_names:
        surfaces = wing_primitive_names
        surfaces.remove(key)
        print(key)
        spatial_rep.plot(primitives=surfaces)

# make wing components
left_wing_names = []
left_wing_top_names = []
left_wing_bottom_names = []
left_wing_te_top_names = []
left_wing_te_bottom_names = []
for i in range(22+168,37+168):
    surf_name = 'Wing_1_' + str(i)
    left_wing_names.append(surf_name)
    if i%4 == 2:
        left_wing_te_bottom_names.append(surf_name)
    elif i%4 == 3:
        left_wing_bottom_names.append(surf_name)
    elif i%4 == 0:
        left_wing_top_names.append(surf_name)
    else:
        left_wing_te_top_names.append(surf_name)
#wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=list(spatial_rep.primitives.keys()))
wing_left = LiftingSurface(name='wing_left', spatial_representation=spatial_rep, primitive_names=left_wing_names)
wing_left_top = LiftingSurface(name='wing_left_top', spatial_representation=spatial_rep, primitive_names=left_wing_top_names)
wing_left_bottom = LiftingSurface(name='wing_left_bottom', spatial_representation=spatial_rep, primitive_names=left_wing_bottom_names)

# structural_left_wing_names = left_wing_names.copy()
structural_left_wing_names = []


# projections for internal structure
num_pts = 10
spar_rib_spacing_ratio = 3
num_rib_pts = 20

# Important points from openVSP
root_te = np.array([15.170, 0., 1.961]) * ft2m
root_le = np.array([8.800, 0, 1.989]) * ft2m
tip_te = np.array([11.300, -14.000, 1.978]) * ft2m
tip_le = np.array([8.796, -14.000, 1.989]) * ft2m



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

f_spar_top = wing_left_top.project(f_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)
f_spar_bottom = wing_left_bottom.project(f_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)

r_spar_top = wing_left_top.project(r_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)
r_spar_bottom = wing_left_bottom.project(r_spar_projection_points, plot=do_plots, force_reprojection=force_reprojection)

ribs_top = wing_left_top.project(rib_projection_points, direction=[0.,0.,1.], plot=do_plots, grid_search_n=100, force_reprojection=force_reprojection)
ribs_bottom = wing_left_bottom.project(rib_projection_points, direction=[0.,0.,1.], plot=do_plots, grid_search_n=100, force_reprojection=force_reprojection)


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

# make surface panels
n_cp = (num_rib_pts,2)
order = (2,)

surface_dict = {}
for i in range(num_ribs-1):
    t_panel_points = ribs_top.value[:,(i,i+1),:]
    t_panel_bspline = bsf.fit_bspline(t_panel_points, num_control_points=n_cp, order=order)
    t_panel = SystemPrimitive('t_panel_' + str(i), t_panel_bspline)
    surface_dict[t_panel.name] = t_panel
    structural_left_wing_names.append(t_panel.name)

    b_panel_points = ribs_bottom.value[:,(i,i+1),:]
    b_panel_bspline = bsf.fit_bspline(b_panel_points, num_control_points=n_cp, order=order)
    b_panel = SystemPrimitive('b_panel_' + str(i), b_panel_bspline)
    surface_dict[b_panel.name] = b_panel
    structural_left_wing_names.append(b_panel.name)

surface_dict.update(spatial_rep.primitives)
spatial_rep.primitives = surface_dict

spatial_rep.assemble()

# Wing
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
wing = LiftingSurface(name='Wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)
sys_rep.add_component(wing)



if False:
    spatial_rep.plot(plot_types=['wireframe'])


# structural wing component:
wing_left_structural = LiftingSurface(name='wing_left_structural',
                                      spatial_representation=spatial_rep,
                                      primitive_names = structural_left_wing_names)
sys_rep.add_component(wing_left_structural)

### Non-wing stuff



# left wing only
num_wing_vlm = 21
num_chordwise_vlm = 5
point10 = root_le
point11 = root_te
point20 = tip_le
point21 = tip_te

leading_edge_points = np.linspace(point10, point20, num_wing_vlm)
trailing_edge_points = np.linspace(point11, point21, num_wing_vlm)

leading_edge = wing.project(leading_edge_points, direction=np.array([-1., 0., 0.]), plot=plots_flag, force_reprojection=force_reprojection)
trailing_edge = wing.project(trailing_edge_points, direction=np.array([-1., 0., 0.]), plot=plots_flag, force_reprojection=force_reprojection)

# Chord Surface
wing_chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
if plots_flag:
    spatial_rep.plot_meshes([wing_chord_surface])

# Upper and lower surface
wing_upper_surface_wireframe = wing.project(wing_chord_surface.value + np.array([0., 0., 0.5*ft2m]),
                                            direction=np.array([0., 0., -1.]), grid_search_n=25,
                                            plot=plots_flag, max_iterations=200, force_reprojection=force_reprojection)
wing_lower_surface_wireframe = wing.project(wing_chord_surface.value - np.array([0., 0., 0.5*ft2m]),
                                            direction=np.array([0., 0., 1.]), grid_search_n=25,
                                            plot=plots_flag, max_iterations=200, force_reprojection=force_reprojection)

# Chamber surface
left_wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
left_wing_vlm_mesh_name = 'left_wing_vlm_mesh'
sys_rep.add_output(left_wing_vlm_mesh_name, left_wing_camber_surface)

# OML mesh
left_wing_oml_mesh = am.vstack((wing_upper_surface_wireframe, wing_lower_surface_wireframe))
left_wing_oml_mesh_name = 'left_wing_oml_mesh'
sys_rep.add_output(left_wing_oml_mesh_name, left_wing_oml_mesh)

sys_rep.add_output(name='left_wing_chord_distribution',
                                    quantity=am.norm(leading_edge-trailing_edge))
# endregion



#############################################
filename = "./pav_wing/pav_wing_v2_caddee_mesh_SI_6307_quad.xdmf"
# filename = "./pav_wing/pav_wing_v2_caddee_mesh_SI_2303_quad.xdmf"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    fenics_mesh = xdmf.read_mesh(name="Grid")
nel = fenics_mesh.topology.index_map(fenics_mesh.topology.dim).size_local
nn = fenics_mesh.topology.index_map(0).size_local

nodes = fenics_mesh.geometry.x


## make thickness function and function space
wing_spaces = {}
wing_t_spaces = {}
coefficients = {}
thickness_coefficients = {}
t = 0.005 # starting wing thickness control point values
for name in structural_left_wing_names:
    primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
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


with open(cfile + '/pav_wing/pav_wing_v2_paneled_mesh_data_'+str(nodes.shape[0])+'.pickle', 'rb') as f:
    nodes_parametric = pickle.load(f)

for i in range(len(nodes_parametric)):
    # print(nodes_parametric[i][0].replace(' ', '_').replace(',',''))
    nodes_parametric[i] = (nodes_parametric[i][0].replace(' ', '_').replace(',',''), np.array([nodes_parametric[i][1]]))


thickness_nodes = wing_thickness.evaluate(nodes_parametric)

shell_pde = ShellPDE(fenics_mesh)



# Unstiffened Aluminum 2024 (T4)
# reference: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma2024t4
E = 73.1E9 # unit: Pa
nu = 0.33
h = 0.02*in2m # unit: m
rho = 2780 # unit: kg/m^3
f_d = -rho*h*9.81 # self-weight unit: N
tensile_yield_strength = 324E6 # unit: Pa
safety_factor = 1.5


y_bc = -1e-6
semispan = tip_te[1] + 0.001

G = E/2/(1+nu)

#### Getting facets of the LEFT and the RIGHT edge  ####
DOLFIN_EPS = 3E-16
def ClampedBoundary(x):
    return np.greater(x[1], y_bc)
def TipChar(x):
    return np.less(x[1], semispan)
fdim = fenics_mesh.topology.dim - 1

ds_1 = createCustomMeasure(fenics_mesh, fdim, ClampedBoundary, measure='ds', tag=100)
dS_1 = createCustomMeasure(fenics_mesh, fdim, ClampedBoundary, measure='dS', tag=100)
dx_2 = createCustomMeasure(fenics_mesh, fdim+1, TipChar, measure='dx', tag=10)

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


################# PAV  Wing #################

# Wing shell Mesh
z_offset = 0.0
wing_shell_mesh = am.MappedArray(input=fenics_mesh.geometry.x).reshape((-1,3))
shell_mesh = rmshell.LinearShellMesh(
                    meshes=dict(
                    wing_shell_mesh=wing_shell_mesh,
                    ))

# # design scenario
design_scenario_name = 'structural_sizing'
design_scenario = cd.DesignScenario(name=design_scenario_name)

# region Cruise condition
cruise_name = "cruise_3"
cruise_model = m3l.Model()
cruise_condition = cd.CruiseCondition(name=cruise_name)
cruise_condition.atmosphere_model = cd.SimpleAtmosphereModel()
cruise_condition.set_module_input(name='altitude', val=600*ft2m)
cruise_condition.set_module_input(name='mach_number', val=0.145972)  # 112 mph = 0.145972 Mach = 50m/s
cruise_condition.set_module_input(name='range', val=80467.2)  # 50 miles = 80467.2 m
# cruise_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0.50921594)) # 1g case
cruise_condition.set_module_input(name='pitch_angle', val=np.deg2rad(6)) # 3g case
cruise_condition.set_module_input(name='flight_path_angle', val=0)
cruise_condition.set_module_input(name='roll_angle', val=0)
cruise_condition.set_module_input(name='yaw_angle', val=0)
cruise_condition.set_module_input(name='wind_angle', val=0)
cruise_condition.set_module_input(name='observer_location', val=np.array([0, 0, 600*ft2m]))

cruise_ac_states = cruise_condition.evaluate_ac_states()
cruise_model.register_output(cruise_ac_states)
# endregion


# region Aerodynamics (left wing only)

left_wing_vlm_model = VASTFluidSover(
    surface_names=[
        left_wing_vlm_mesh_name,
    ],
    surface_shapes=[
        (1, ) + left_wing_camber_surface.evaluate().shape[1:],
        ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake'),
    mesh_unit='m',
    cl0=[0.3475, 0.0] # need to tune the coefficient
)
left_wing_vlm_panel_forces, left_wing_vlm_forces, left_wing_vlm_moments  = left_wing_vlm_model.evaluate(ac_states=cruise_ac_states)
cruise_model.register_output(left_wing_vlm_forces)
cruise_model.register_output(left_wing_vlm_moments)
# endregion




vlm_force_mapping_model = VASTNodalForces(
    surface_names=[
        left_wing_vlm_mesh_name,
    ],
    surface_shapes=[
        (1, ) + left_wing_camber_surface.evaluate().shape[1:],
    ],
    initial_meshes=[
        left_wing_camber_surface,
        ]
)

oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=left_wing_vlm_panel_forces, nodal_force_meshes=[left_wing_oml_mesh, left_wing_oml_mesh])
wing_forces = oml_forces[0]

shell_force_map_model = rmshell.RMShellForces(component=wing,
                                                mesh=shell_mesh,
                                                pde=shell_pde,
                                                shells=shells)
cruise_structural_wing_mesh_forces = shell_force_map_model.evaluate(
                        nodal_forces=wing_forces,
                        nodal_forces_mesh=left_wing_oml_mesh)

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
cruise_condition.add_m3l_model('cruise_model', cruise_model)
# Add design condition to design scenario
design_scenario.add_design_condition(cruise_condition)

system_model.add_design_scenario(design_scenario=design_scenario)

caddee_csdl_model = caddee.assemble_csdl()

system_model_name = 'system_model.'+design_scenario_name+'.'+cruise_name+'.'+cruise_name+'.'

caddee_csdl_model.add_constraint(system_model_name+'Wing_rm_shell_model.rm_shell.compliance_model.compliance',upper=2E-4,scaler=1E4)
caddee_csdl_model.add_constraint(system_model_name+'Wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress',upper=324E6/1.5,scaler=1E-8)
caddee_csdl_model.add_objective(system_model_name+'Wing_rm_shell_model.rm_shell.mass_model.mass', scaler=1e-1)

# Minimum thickness: 0.02 inch -> 0.000508 m

h_min = h

i = 0
# skin_shape = (625, 1)
spar_shape = (4, 1)
rib_shape = (40, 1)
skin_shape = rib_shape
shape = (4, 1)
valid_wing_surf = [23, 24, 27, 28, 31, 32, 35, 36]
# valid_structural_left_wing_names = []
# for name in structural_left_wing_names:
#     if "spar" in name:
#         valid_structural_left_wing_names.append(name)
#     elif "rib" in name:
#         valid_structural_left_wing_names.append(name)
#     elif "Wing" in name:
#         for id in valid_wing_surf:
#             if str(id+168) in name:
#                 valid_structural_left_wing_names.append(name)
# print("Full list of surface names for left wing:", structural_left_wing_names)
# print("Valid list of surface names for left wing:", svalid_structural_left_wing_names)

valid_structural_left_wing_names = structural_left_wing_names

################################################################
#### Full thicknesses: individual for spars, skins and ribs ####
################################################################
for name in valid_structural_left_wing_names:
    primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
    name = name.replace(' ', '_').replace(',','')
    surface_id = i
    if "spar" in name:
        shape = spar_shape
    elif "panel" in name:
        shape = skin_shape
    elif "rib" in name:
        shape = rib_shape

    h_init = caddee_csdl_model.create_input('wing_thickness_'+name, val=h_min)
    caddee_csdl_model.add_design_variable('wing_thickness_'+name, # 0.02 in
                                          lower=0.005 * in2m,
                                          upper=0.1 * in2m,
                                          scaler=1000,
                                          )
    caddee_csdl_model.register_output('wing_thickness_surface_'+name, csdl.expand(h_init, shape))
    caddee_csdl_model.connect('wing_thickness_surface_'+name,
                                system_model_name+'wing_thickness_evaluation.'+\
                                name+'_t_coefficients')
    i += 1



# region Optimization Setup


sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()

########################## Run optimization ##################################
# prob = CSDLProblem(problem_name='pav', simulator=sim)

# optimizer = SLSQP(prob, maxiter=50, ftol=1E-5)

# # from modopt.snopt_library import SNOPT
# # optimizer = SNOPT(prob,
# #                   Major_iterations = 100,
# #                   Major_optimality = 1e-5,
# #                   append2file=False)

# optimizer.solve()
# optimizer.print_results()


####### Aerodynamic output ##########
print("="*60)
print("="*20+'aerodynamic outputs'+"="*20)
print("="*60)
print('Pitch: ', np.rad2deg(
    sim[system_model_name+cruise_name+'_ac_states_operation.'+cruise_name+'_pitch_angle']))
print('C_L: ', sim[system_model_name+'left_wing_vlm_mesh_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.left_wing_vlm_mesh_C_L'])
print('Total lift: ', sim[system_model_name+'left_wing_vlm_mesh_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.total_lift'])

####### Structural output ##########
print("="*60)
print("="*20+'structure outputs'+"="*20)
print("="*60)
# Comparing the solution to the Kirchhoff analytical solution
f_shell = sim[system_model_name+'Wing_rm_shell_force_mapping.wing_shell_forces']
f_vlm = sim[system_model_name+'left_wing_vlm_mesh_vlm_force_mapping_model.left_wing_vlm_mesh_oml_forces'].reshape((-1,3))
u_shell = sim[system_model_name+'Wing_rm_shell_model.rm_shell.disp_extraction_model.wing_shell_displacement']
# u_nodal = sim['Wing_rm_shell_displacement_map.wing_shell_nodal_displacement']
uZ = u_shell[:,2]
# uZ_nodal = u_nodal[:,2]


wing_tip_compliance = sim[system_model_name+'Wing_rm_shell_model.rm_shell.compliance_model.compliance']
wing_mass = sim[system_model_name+'Wing_rm_shell_model.rm_shell.mass_model.mass']
wing_elastic_energy = sim[system_model_name+'Wing_rm_shell_model.rm_shell.elastic_energy_model.elastic_energy']
wing_aggregated_stress = sim[system_model_name+'Wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress']
wing_von_Mises_stress = sim[system_model_name+'Wing_rm_shell_model.rm_shell.von_Mises_stress_model.von_Mises_stress']
########## Output: ##########
# print("Spar, rib, skin thicknesses:", sim['h_spar'], sim['h_rib'], sim['h_skin'])

fz_func = Function(shell_pde.VT)
fz_func.x.array[:] = f_shell[:,-1]

fx_func = Function(shell_pde.VT)
fx_func.x.array[:] = f_shell[:,0]

fy_func = Function(shell_pde.VT)
fy_func.x.array[:] = f_shell[:,1]

dummy_func = Function(shell_pde.VT)
dummy_func.x.array[:] = 1.0
print("vlm forces:", sum(f_vlm[:,0]),sum(f_vlm[:,1]),sum(f_vlm[:,2]))
print("shell forces:", dolfinx.fem.assemble_scalar(form(fx_func*ufl.dx)),
                        dolfinx.fem.assemble_scalar(form(fy_func*ufl.dx)),
                        dolfinx.fem.assemble_scalar(form(fz_func*ufl.dx)))

print("Wing surface area:", dolfinx.fem.assemble_scalar(form(dummy_func*ufl.dx)))
print("Wing tip deflection (m):",max(abs(uZ)))
print("Wing tip compliance (= tip deflection^3/2 m^3):",wing_tip_compliance)
print("Wing total mass (kg):", wing_mass)
print("Wing aggregated von Mises stress (Pascal):", wing_aggregated_stress)
print("Wing maximum von Mises stress (Pascal):", max(wing_von_Mises_stress))
print("  Number of elements = "+str(nel))
print("  Number of vertices = "+str(nn))


# ######## Visualization: ##############
import vedo
# #
# plotter = vedo.Plotter()
# wing_shell_mesh_plot = vedo.Points(wing_shell_mesh.value.reshape((-1,3)))
# wing_oml_plot = vedo.Points(wing_oml_mesh.value.reshape((-1,3)))
# plotter.show([wing_shell_mesh_plot, wing_oml_plot], interactive=True, axes=1)    # Plotting point cloud
#
# plotter = vedo.Plotter()
# wing_oml_plot = vedo.Points(wing_oml_mesh.value.reshape((-1,3)))
# plotter.show([wing_oml_plot], interactive=True, axes=1)    # Plotting point cloud

# plotter = vedo.Plotter()
# wing_shell_mesh_plot = vedo.Points(wing_shell_mesh.value.reshape((-1,3)))
# wing_oml_plot = vedo.Points(left_wing_oml_mesh.value.reshape((-1,3)))
# plotter.show([wing_shell_mesh_plot, wing_oml_plot], interactive=True, axes=1)    # Plotting point cloud
