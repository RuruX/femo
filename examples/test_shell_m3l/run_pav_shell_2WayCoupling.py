## Caddee
from caddee.utils.aircraft_models.pav.pav_geom_mesh import PavGeomMesh
import caddee.api as cd

## Solvers
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from VAST.core.generate_mappings_m3l import VASTNodalForces, VASTNodelDisplacements
from VAST.core.vlm_llt.viscous_correction import ViscousCorrectionModel
# from lsdo_airfoil.core.pressure_profile import PressureProfile, NodalPressureProfile
import dolfinx
from femo.fea.utils_dolfinx import *
import shell_module as rmshell
from shell_pde import ShellPDE, NodalMap

# Other lsdo lab stuff
import csdl
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import m3l
from m3l.utils.utils import index_functions
import lsdo_geo as lg
import array_mapper as am
from m3l.core.function_spaces import IDWFunctionSpace


## Other stuff
import numpy as np
from scipy.sparse import csr_matrix, block_diag, dok_matrix, coo_matrix
from scipy.sparse import vstack as sparse_vstack
from mpi4py import MPI
import pickle
import pathlib
import sys
from copy import deepcopy

sys.setrecursionlimit(100000)


def construct_VLM_vertex_to_force_map(vertex_array_shape, F_chord_pos=0.25):
    node_map = np.zeros((vertex_array_shape[1]*vertex_array_shape[2], (vertex_array_shape[1]-1)*(vertex_array_shape[2]-1)))

    for j in range(vertex_array_shape[1]-1):
        for i in range(vertex_array_shape[2]-1):
            panel_num = j + i*(vertex_array_shape[1]-1)
            node_map[j + i*vertex_array_shape[1], panel_num] = 0.5*(1-F_chord_pos)
            node_map[j + 1 + i*vertex_array_shape[1], panel_num] = 0.5*(1-F_chord_pos)
            node_map[j + (i + 1)*vertex_array_shape[1], panel_num] = 0.5*F_chord_pos
            node_map[j + 1 + (i + 1)*vertex_array_shape[1], panel_num] = 0.5*F_chord_pos
    return node_map

debug_geom_flag = False
force_reprojection = False
visualize_flag = False
# Dashboard and xdmf recorder cannot be turned on at the same time
dashboard = False
xdmf_record = False

# flags that control whether couplings are handled in a conservative way
vlm_conservative = True
fenics_conservative = False

ft2m = 0.3048
in2m = 0.0254

# wing_cl0 = 0.3366
# pitch_angle_list = [-0.02403544, 6, 12.48100761]
# h_0 = 0.02*in2m

wing_cl0 = 0.3662
pitch_angle_list = [-0.38129494, 6, 12.11391141]
h_0 = 0.05*in2m
pitch_angle = np.deg2rad(pitch_angle_list[2])


caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()

# region Geometry and meshes
pav_geom_mesh = PavGeomMesh()
pav_geom_mesh.setup_geometry(
    include_wing_flag=True,
    include_htail_flag=False,
)
pav_geom_mesh.setup_internal_wingbox_geometry(debug_geom_flag=debug_geom_flag,
                                              force_reprojection=force_reprojection)
pav_geom_mesh.sys_rep.spatial_representation.assemble()
pav_geom_mesh.oml_mesh(include_wing_flag=True, 
                       grid_num_u=4, grid_num_v=3,
                       debug_geom_flag=debug_geom_flag, force_reprojection=force_reprojection)
pav_geom_mesh.vlm_meshes(include_wing_flag=True, num_wing_spanwise_vlm=51, num_wing_chordwise_vlm=9,
                         visualize_flag=visualize_flag, force_reprojection=force_reprojection)
pav_geom_mesh.setup_index_functions()

caddee.system_representation = sys_rep = pav_geom_mesh.sys_rep
caddee.system_parameterization = sys_param = pav_geom_mesh.sys_param
sys_param.setup()
spatial_rep = sys_rep.spatial_representation
# endregion

# we first determine the framework surfaces that contain both displacements and forces 
disp_input_surface_names = pav_geom_mesh.functions['wing_displacement_input'].space.spaces.keys()
force_surface_names = pav_geom_mesh.functions['wing_force'].space.spaces.keys()

framework_work_surface_names = []

for surf_name in disp_input_surface_names:
    if surf_name in force_surface_names:
        framework_work_surface_names += [surf_name]

disp_evaluationmaps = {}
disp_evaluationmaps_list = []

# loop over the surfaces that contain both forces and displacements and compute their invariant matrices
for surf_name in framework_work_surface_names:
    surf_primitive = spatial_rep.get_primitives([surf_name])[surf_name].geometry_primitive
    displacement_space = pav_geom_mesh.functions['wing_displacement_input'].space.spaces[surf_name]
    force_space = pav_geom_mesh.functions['wing_force'].space.spaces[surf_name]

    # NOTE: wing forces are currently represented by a set of vectors in these (parametric) points:
    force_parametricpoints = pav_geom_mesh.functions['wing_force'].space.spaces[surf_name].points
    # we sample the displacements in the same parametric points
    displacement_evaluationmap = displacement_space.compute_evaluation_map(force_parametricpoints)
    # NOTE: `displacement_evaluationmap` is effectively the invariant matrix of a single displacement-force component pair (in x, y or z) on surface `test_surf`
    disp_evaluationmaps[surf_name] = block_diag([displacement_evaluationmap]*3)  # repeat evaluation map 3 times to account for [x, y, z] in displacement vector
    disp_evaluationmaps_list += [displacement_evaluationmap]

# construct the total displacement evaluation map (i.e. invariant matrix) by stacking all surface contributions along the diagonal and repeating the result three times
framework_invariantmatrix_nonstacked = block_diag(disp_evaluationmaps_list)
framework_invariantmatrix = block_diag([framework_invariantmatrix_nonstacked]*3) 

# Construct VLM invariant matrix link nodal displacements and locations of force vectors
vlm_camber_mesh = pav_geom_mesh.mesh_data['vlm']['chamber_surface']['wing'].value
vlm_invariantmatrix_nonstacked = construct_VLM_vertex_to_force_map(vlm_camber_mesh.shape)
vlm_invariantmatrix = block_diag([vlm_invariantmatrix_nonstacked.T]*3)

# region FEniCS
#############################################
# filename = "./pav_wing/pav_wing_v2_caddee_mesh_SI_6307_quad.xdmf"
filename = "./test_shell_m3l/pav_wing/pav_wing_v2_caddee_mesh_SI_2303_quad.xdmf"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    fenics_mesh = xdmf.read_mesh(name="Grid")
nel = fenics_mesh.topology.index_map(fenics_mesh.topology.dim).size_local
nn = fenics_mesh.topology.index_map(0).size_local

nodes = fenics_mesh.geometry.x


with open('./test_shell_m3l/pav_wing/pav_wing_v2_paneled_mesh_data_'+str(nodes.shape[0])+'.pickle', 'rb') as f:
    nodes_parametric = pickle.load(f)

for i in range(len(nodes_parametric)):
    nodes_parametric[i] = (nodes_parametric[i][0].replace(' ', '_').replace(',',''), np.array([nodes_parametric[i][1]]))

wing_thickness = pav_geom_mesh.functions['wing_thickness']
thickness_nodes = wing_thickness.evaluate(nodes_parametric)

# `shell_pde` contains the nodal displacement map
shell_pde = ShellPDE(fenics_mesh)


# Unstiffened Aluminum 2024 (T4)
# reference: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma2024t4
E = 73.1E9 # unit: Pa
nu = 0.33
h = h_0 # unit: m
rho = 2780 # unit: kg/m^3
f_d = -rho*h*9.81 # self-weight unit: N
tensile_yield_strength = 324E6 # unit: Pa
safety_factor = 1.5


y_bc = -1e-6
semispan = pav_geom_mesh.geom_data['points']['wing']['l_tip_te'][1] + 0.001

G = E/2/(1+nu)

# Constructs FEniCS invariant matrix of force and displacement
# NOTE: DoFs seem to be ordered as [x_1, y_1, z_1, x_2, y_2, z_2, ...]
fenics_force_function = TestFunction(shell_pde.VF)
fenics_disp_function = TrialFunction(shell_pde.W.sub(0).collapse()[0])
fenics_invariantmatrix_petsc = assemble_matrix(form(inner(fenics_force_function, fenics_disp_function)*dx))
fenics_invariantmatrix_petsc.assemble()

fenics_invariantmatrix_csr = fenics_invariantmatrix_petsc.getValuesCSR()
fenics_invariantmatrix_sp = csr_matrix((fenics_invariantmatrix_csr[2], fenics_invariantmatrix_csr[1], fenics_invariantmatrix_csr[0]))
# multiply the invariant matrix with the displacement extraction operator (the matrix that only retains the displacements and leaves out the rotations)
disp_extraction_matrix_list = shell_pde.construct_disp_extraction_mats()
disp_extraction_matrix = sparse_vstack(disp_extraction_matrix_list)

# the matrix below is the FEniCS invariant matrix!
fenics_invariantmatrix = fenics_invariantmatrix_sp@disp_extraction_matrix

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
                        'g': g,
                        'record': xdmf_record}


################# PAV  Wing #################

# Wing shell Mesh
z_offset = 0.0
wing_shell_mesh = am.MappedArray(input=fenics_mesh.geometry.x).reshape((-1,3))
shell_mesh = rmshell.LinearShellMesh(
                    meshes=dict(
                    wing_shell_mesh=wing_shell_mesh,
                    ))


# endregion



# region Mission
design_scenario_name = 'structural_sizing'
design_scenario = cd.DesignScenario(name=design_scenario_name)
# endregion

# region Cruise condition
cruise_name = "cruise_3"
cruise_model = m3l.Model()
cruise_condition = cd.CruiseCondition(name=cruise_name)
cruise_condition.atmosphere_model = cd.SimpleAtmosphereModel()
cruise_condition.set_module_input(name='altitude', val=600 * ft2m)
cruise_condition.set_module_input(name='mach_number', val=0.145972)  # 112 mph = 0.145972 Mach
cruise_condition.set_module_input(name='range', val=80467.2)  # 50 miles = 80467.2 m
cruise_condition.set_module_input(name='pitch_angle', val=pitch_angle)
cruise_condition.set_module_input(name='flight_path_angle', val=0)
cruise_condition.set_module_input(name='roll_angle', val=0)
cruise_condition.set_module_input(name='yaw_angle', val=0)
cruise_condition.set_module_input(name='wind_angle', val=0)
cruise_condition.set_module_input(name='observer_location', val=np.array([0, 0, 600 * ft2m]))

cruise_ac_states = cruise_condition.evaluate_ac_states()
cruise_model.register_output(cruise_ac_states)
# endregion

## We define all solver-specific maps here so we can use their properties and functions to construct work-preserving projection maps
# map displacements from OML to VLM
vlm_disp_mapping_model = VASTNodelDisplacements(
    surface_names=[
        pav_geom_mesh.functions['wing_displacement_input'].name,
    ],
    surface_shapes=[
        (1,) + pav_geom_mesh.mesh_data['vlm']['chamber_surface']['wing'].evaluate().shape[1:],
        ],
    initial_meshes=[
        pav_geom_mesh.mesh_data['vlm']['chamber_surface']['wing'],
    ],
    output_names=[
        'wing_mesh_displacements'
    ]
)

# map forces from VLM to OML
vlm_force_mapping_model = VASTNodalForces(
    surface_names=[
        'wing',
    ],
    surface_shapes=[
        (1,) + pav_geom_mesh.mesh_data['vlm']['chamber_surface']['wing'].evaluate().shape[1:],
        ],
    initial_meshes=[
        pav_geom_mesh.mesh_data['vlm']['chamber_surface']['wing'],
    ]
)



wing_oml_wing_mesh = pav_geom_mesh.mesh_data['oml']['oml_geo_nodes']['wing']


# Map displacements from column 3 (framework) to 2
wing_displacement_input = pav_geom_mesh.functions['wing_displacement_input']
wing_force = pav_geom_mesh.functions['wing_force']
oml_para_nodes_wing = pav_geom_mesh.mesh_data['oml']['oml_para_nodes']['wing']

if vlm_conservative:
    disp_composition_mat = vlm_disp_mapping_model.disp_map(vlm_disp_mapping_model.parameters['initial_meshes'][0], oml=wing_oml_wing_mesh)
    # disp_composition_mat = block_diag([disp_composition_mat]*3)
    disp_composition_mat_test = deepcopy(disp_composition_mat)
    disp_composition_mat_test[disp_composition_mat_test <= 1e-4] = 0.

    # TODO: Check length of vlm_force_mapping_model.force_points and nodal_forces_meshes
    eval_pts_location = 0.25
    vlm_mesh = pav_geom_mesh.mesh_data['vlm']['chamber_surface']['wing'].value
    vlm_mesh_reshaped = vlm_mesh.reshape((1, vlm_mesh.shape[2], vlm_mesh.shape[1], 3))
    force_points_vlm = (
            (1 - eval_pts_location) * 0.5 * vlm_mesh_reshaped[:, 0:-1, 0:-1, :] +
            (1 - eval_pts_location) * 0.5 * vlm_mesh_reshaped[:, 0:-1, 1:, :] +
            eval_pts_location * 0.5 * vlm_mesh_reshaped[:, 1:, 0:-1, :] +
            eval_pts_location * 0.5 * vlm_mesh_reshaped[:, 1:, 1:, :])

    force_map_oml = vlm_force_mapping_model.disp_map(force_points_vlm.reshape((-1,3)),wing_oml_wing_mesh.value.reshape((-1,3)))
    # force_map_oml = block_diag([force_map_oml]*3)

    associated_coords = {}
    index = 0
    for item in oml_para_nodes_wing:
        key = item[0]
        value = item[1]
        if key not in associated_coords.keys():
            associated_coords[key] = [[index], value]
        else:
            associated_coords[key][0].append(index)
            associated_coords[key] = [associated_coords[key][0], np.vstack((associated_coords[key][1], value))]
        index += 1

    output_shape = (len(oml_para_nodes_wing), wing_force.coefficients[oml_para_nodes_wing[0][0]].shape[-1])

    fitting_matrix_list = []
    coefficient_idx_list = []
    fitting_matrix_key_list = []

    for key, value in associated_coords.items(): # in the future, use submodels from the function spaces?
        if hasattr(wing_force.space.spaces[key], 'compute_fitting_map'):
            fitting_matrix = wing_force.space.spaces[key].compute_fitting_map(value[1])
    
        fitting_matrix_list += [fitting_matrix]
        coefficient_idx_list += [value[0]]
        fitting_matrix_key_list += [key]
    
    # concatenate index list and construct fitting matrix list
    fitting_matrix_unordered = block_diag(fitting_matrix_list)
    coefficient_idxs = np.concatenate(coefficient_idx_list) 

    fitting_matrix_unordered = fitting_matrix_unordered.tocsr()
    # reorder the fitting matrix columns based on the entries of coefficient_idxs
    fitting_matrix = dok_matrix(fitting_matrix_unordered.shape)
    for i in range(coefficient_idxs.shape[0]):
        fitting_matrix[i, :] = fitting_matrix_unordered[coefficient_idxs[i], :]
    
    # force_map_oml_to_framework = block_diag([fitting_matrix]*3)

    disp_oml_nodes = wing_displacement_input.evaluate_conservative(fitting_matrix_key_list,
                                                                   num_oml_output_points=wing_oml_wing_mesh.value.shape[0],
                                                                   num_oml_dual_variable_points=wing_oml_wing_mesh.value.shape[0],
                                                                   composition_mat=disp_composition_mat,
                                                                   otherside_composed_mat=fitting_matrix@force_map_oml, 
                                                                   solver_invariant_mat=coo_matrix(vlm_invariantmatrix_nonstacked), 
                                                                   framework_invariant_mat=framework_invariantmatrix_nonstacked.T)  # use OML parametric nodes to evaluate wing displacement function
else:
    # Map displacements from column 3 to 2
    disp_oml_nodes = wing_displacement_input.evaluate(oml_para_nodes_wing)  # use OML parametric nodes to evaluate wing displacement function

    # Map displacements from column 2 to 1
    vlm_nodal_displacements = vlm_disp_mapping_model.evaluate(nodal_displacements=[disp_oml_nodes, ],
                                                nodal_displacements_mesh=[wing_oml_wing_mesh, ])

# region VLM Solver
# Construct VLM solver with CADDEE geometry
vlm_model = VASTFluidSover(
    surface_names=[
        'wing',
    ],
    surface_shapes=[
        (1,) + pav_geom_mesh.mesh_data['vlm']['chamber_surface']['wing'].evaluate().shape[1:],
        ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake'),
    mesh_unit='m',
    cl0=[wing_cl0, ],
    mesh = [
        pav_geom_mesh.mesh_data['vlm']['chamber_surface']['wing']
    ],
    displacement_names=[
        vlm_nodal_displacements[0].name
    ]
)
# Compute VLM solution (input: displacements, output: forces)
# NOTE: `wing_vlm_panel_forces` are the panel force vectors
# print("pre-VLM model evaluate")
wing_vlm_panel_forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=cruise_ac_states, displacements=vlm_nodal_displacements)
# print("post-VLM model evaluate")
# register the total VLM forces and moments as outputs for the M3L Model (column 5)
cruise_model.register_output(vlm_forces)
cruise_model.register_output(vlm_moments)

# construct operation to map from column 1 to column 2
oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=wing_vlm_panel_forces,
                                              nodal_force_meshes=[wing_oml_wing_mesh, ])
wing_forces = oml_forces[0]
# print("post-VLM force map evaluate")
# endregion

# region Strucutral Loads

# map forces from column 2 to column 3 (framework) 
wing_force.inverse_evaluate(oml_para_nodes_wing, wing_forces)
cruise_model.register_output(wing_force.coefficients)

left_wing_oml_para_coords = pav_geom_mesh.mesh_data['oml']['oml_para_nodes']['left_wing']
left_oml_geo_nodes = spatial_rep.evaluate_parametric(left_wing_oml_para_coords)

# map forces from column 3 to 4 (`evaluate` is used since we're mapping out from the framework representation)
left_wing_forces = wing_force.evaluate(left_wing_oml_para_coords)
wing_component = pav_geom_mesh.geom_data['components']['wing']

# Define force map that takes nodal forces and projects them to the shell mesh (column 4 to 5)
shell_force_map_model = rmshell.RMShellForces(component=wing_component,
                                                mesh=shell_mesh,
                                                pde=shell_pde,
                                                shells=shells)
cruise_structural_wing_mesh_forces = shell_force_map_model.evaluate(
                        nodal_forces=left_wing_forces,
                        nodal_forces_mesh=left_oml_geo_nodes)
# endregion

# region Structures

shell_displacements_model = rmshell.RMShell(component=wing_component,
                                            mesh=shell_mesh,
                                            pde=shell_pde,
                                            shells=shells)

cruise_structural_wing_mesh_displacements, _, cruise_structural_wing_mesh_stresses, wing_mass = \
                                shell_displacements_model.evaluate(
                                    forces=cruise_structural_wing_mesh_forces,
                                    thicknesses=thickness_nodes)
cruise_model.register_output(cruise_structural_wing_mesh_stresses)
cruise_model.register_output(cruise_structural_wing_mesh_displacements)
cruise_model.register_output(wing_mass)

# endregion

# print("post-RMshell evaluate")
# region Nodal Displacements

grid_num = 10
transfer_para_mesh = []
structural_left_wing_names = pav_geom_mesh.geom_data['primitive_names']['structural_left_wing_names']
# left_wing_skin_names = pav_geom_mesh.geom_data['primitive_names']['left_wing_bottom_names'] + pav_geom_mesh.geom_data['primitive_names']['left_wing_top_names']

# element_projection_names = structural_left_wing_names + left_wing_skin_names
# element_projection_names = pav_geom_mesh.geom_data['primitive_names']['left_wing'] #+ structural_left_wing_names except panels
element_projection_names = pav_geom_mesh.geom_data['primitive_names']['both_wings'] #+ structural_left_wing_names except panels

for name in element_projection_names:
    for u in np.linspace(0,1,grid_num):
        for v in np.linspace(0,1,grid_num):
            transfer_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

transfer_geo_nodes_ma = spatial_rep.evaluate_parametric(transfer_para_mesh)

# construct map from column 5 to 4
shell_nodal_displacements_model = rmshell.RMShellNodalDisplacements(component=wing_component,
                                                                    mesh=shell_mesh,
                                                                    pde=shell_pde,
                                                                    shells=shells)

nodal_displacements, tip_displacement = shell_nodal_displacements_model.evaluate(cruise_structural_wing_mesh_displacements, transfer_geo_nodes_ma)

# TODO: Figure out exactly where error is being thrown (around here somewhere)

# construct map from column 4 to 3 (the framework representation)

wing_displacement_output = pav_geom_mesh.functions['wing_displacement_output']

wing_displacement_output.inverse_evaluate(transfer_para_mesh, nodal_displacements)
cruise_model.register_output(wing_displacement_output.coefficients)

wing_stress = pav_geom_mesh.functions['wing_stress']
wing_stress.inverse_evaluate(nodes_parametric, cruise_structural_wing_mesh_stresses, regularization_coeff=1e-3)
cruise_model.register_output(wing_stress.coefficients)

cruise_model.register_output(tip_displacement)
cruise_model.register_output(nodal_displacements)

# endregion

# Add cruise m3l model to cruise condition
cruise_condition.add_m3l_model('cruise_model', cruise_model)
# Add design condition to design scenario
design_scenario.add_design_condition(cruise_condition)

system_model.add_design_scenario(design_scenario=design_scenario)

caddee_csdl_model = caddee.assemble_csdl()

system_model_name = 'system_model.'+design_scenario_name+'.'+cruise_name+'.'+cruise_name+'.'


caddee_csdl_model.add_constraint(system_model_name+'Wing_rm_shell_displacement_map.wing_shell_tip_displacement',upper=0.1,scaler=1E1)
caddee_csdl_model.add_constraint(system_model_name+'Wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress',upper=324E6/1.5,scaler=1E-8)
caddee_csdl_model.add_objective(system_model_name+'Wing_rm_shell_model.rm_shell.mass_model.mass', scaler=1e-1)

# Minimum thickness: 0.02 inch -> 0.000508 m

h_min = h

i = 0
shape = (9, 1)
valid_structural_left_wing_names = structural_left_wing_names

################################################################
#### Full thicknesses: individual for spars, skins and ribs ####
################################################################
for name in valid_structural_left_wing_names:
    primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
    name = name.replace(' ', '_').replace(',','')
    surface_id = i

    h_init = caddee_csdl_model.create_input('wing_thickness_dv_'+name, val=h_min)
    caddee_csdl_model.add_design_variable('wing_thickness_dv_'+name, # 0.02 in
                                          lower=0.005 * in2m,
                                          upper=0.1 * in2m,
                                          scaler=1000,
                                          )
    caddee_csdl_model.register_output('wing_thickness_surface_'+name, csdl.expand(h_init, shape))
    caddee_csdl_model.connect('wing_thickness_surface_'+name,
                                system_model_name+'wing_thickness_function_evaluation.'+\
                                name+'_wing_thickness_coefficients')
    i += 1

if dashboard:
    import lsdo_dash.api as ld
    index_functions_map = {}
    
    index_functions_map['wing_thickness'] = wing_thickness  
    index_functions_map['wing_force'] = wing_force
    index_functions_map['wing_displacement_input'] = wing_displacement_input
    index_functions_map['wing_displacement_output'] = wing_displacement_output
    index_functions_map['wing_stress'] = wing_stress

    rep = csdl.GraphRepresentation(caddee_csdl_model)

    # profiler.disable()
    # profiler.dump_stats('output')

    caddee_viz = ld.caddee_plotters.CaddeeViz(
        caddee = caddee,
        system_m3l_model = system_model,
        design_configuration_map={},
    )

if __name__ == '__main__':
    if dashboard:
        from dash_pav import TC2DB
        dashbuilder = TC2DB()
        sim = Simulator(rep, analytics=True, dashboard = dashbuilder)
    else:
        # print("pre-Simulator instantiation")
        sim = Simulator(caddee_csdl_model, analytics=True)
        # print("post-Simulator instantiation")

    # TODO: Wrap `sim.run` in while loop. Modify the caddee_csdl_model above during each iteration to set the framework displacement


    # displacement_arrays_old = []
    # displacement_arrays_new = []
    array_update_norms = np.zeros((len(wing_displacement_input.coefficients),))
    vlm_force_list = []
    max_disp_update_list = []
    vlm_work_list = []
    total_framework_work_disp_inputs_list = []
    fe_work_list = []
    total_framework_work_disp_outputs_list = []


    running = True

    # we loop over the force and displacement surfaces to determine their shared surfaces


    # initialize iteration loop
    iter_idx = 0
    while running:
        print("---"*10)
        print("Iteration {}".format(iter_idx))
        # run sim in current iteration

        # sim = Simulator(caddee_csdl_model, analytics=True)

        # set displacement inputs
        if iter_idx > 0:
            for i, key in enumerate(wing_displacement_output.coefficients):
                sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_evaluation.{}_wing_displacement_input_coefficients'.format(key)] = disp_output_list[i]

        sim.run()

        disp_input_list = []
        disp_output_list = []
        for i, key in enumerate(wing_displacement_output.coefficients):
            # query corresponding object in sim dict
            displacement_array_input = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_evaluation.{}_wing_displacement_input_coefficients'.format(key)]
            displacement_array_output = deepcopy(sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_output_function_inverse_evaluation.{}_wing_displacement_output_coefficients'.format(key)])
            array_update_norms[i] = np.linalg.norm(np.subtract(displacement_array_input, displacement_array_output))#/np.linalg.norm(displacement_array_output)

            # print("Surface {} displacement input array 2-norm: {}".format(key, np.linalg.norm(displacement_array_input)))
            # print("Surface {} displacement output array 2-norm: {}".format(key, np.linalg.norm(displacement_array_output)))

            disp_input_list += [displacement_array_input]
            disp_output_list += [displacement_array_output]

            # write output array back to input array for next iteration
            # sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_evaluation.{}_wing_displacement_input_coefficients'.format(key)] = displacement_array_output


        vlm_force_list += [np.sum(sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_total_forces'], axis=1)[0]]
        max_disp_update_list += [array_update_norms.max()]

        print("Max 2-norm update: {}".format(array_update_norms.max()))
        print("2-norm updates: {}".format(array_update_norms))

        if array_update_norms.max() < 1e-10 or iter_idx >= 10:
            running = False

        # Below we compute the aeroelastic work with the various displacement and force variables & the invariant matrices:
        # VLM work (F^T@mat@u)
        vlm_F = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_total_forces'][0, :, :]
        vlm_F_flat = vlm_F.flatten(order='F')
        vlm_u = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_model.vast.wing_mesh_displacements'][0, :, :, :]
        vlm_u_2d = np.reshape(vlm_u, (vlm_u.shape[0]*vlm_u.shape[1], vlm_u.shape[2]), order='C')
        vlm_u_flat = vlm_u_2d.flatten(order='F')
        vlm_work = vlm_F_flat@vlm_invariantmatrix@vlm_u_flat

        # Framework work with input displacements
        total_framework_work_disp_inputs = 0.
        for surf_name in framework_work_surface_names:
            surf_disp = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_evaluation.{}_wing_displacement_input_coefficients'.format(surf_name)]
            surf_disp_flat = surf_disp.flatten(order='F')
            surf_force = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_inverse_evaluation.{}_wing_force_coefficients'.format(surf_name)]
            surf_force_flat = surf_force.flatten(order='F')
            surf_invariantmatrix = disp_evaluationmaps[surf_name]
            surf_work = surf_force_flat@surf_invariantmatrix@surf_disp_flat

            total_framework_work_disp_inputs += surf_work

        # FEniCS work (F^T@mat@u):
        fe_F = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_force_mapping.wing_shell_forces']
        fe_F_flat = fe_F.flatten(order='C')
        fe_u = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_model.rm_shell.solid_model.disp_solid']

        # NOTE: We multiply with a factor of 2 here to account for the fact that only one of the wings is included in FEniCS
        fe_work = 2*fe_F_flat@fenics_invariantmatrix@fe_u

        # Framework work with output displacements
        total_framework_work_disp_outputs = 0.
        for surf_name in framework_work_surface_names:
            surf_disp = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_output_function_inverse_evaluation.{}_wing_displacement_output_coefficients'.format(surf_name)]
            surf_disp_flat = surf_disp.flatten(order='F')
            surf_force = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_inverse_evaluation.{}_wing_force_coefficients'.format(surf_name)]
            surf_force_flat = surf_force.flatten(order='F')
            surf_invariantmatrix = disp_evaluationmaps[surf_name]
            surf_work = surf_force_flat@surf_invariantmatrix@surf_disp_flat

            total_framework_work_disp_outputs += surf_work

        print("VLM work: {}".format(vlm_work))
        print("Displacement input framework work: {}".format(total_framework_work_disp_inputs))
        print("FEniCS work: {}".format(fe_work))
        print("Displacement output framework work: {}".format(total_framework_work_disp_outputs))
        vlm_work_list += [vlm_work]
        total_framework_work_disp_inputs_list += [total_framework_work_disp_inputs]
        fe_work_list += [fe_work]
        total_framework_work_disp_outputs_list += [total_framework_work_disp_outputs]

        iter_idx += 1

    print("VLM forces per iteration: {}".format(np.vstack(vlm_force_list)))
    print("Max array update norm per iteration: {}".format(max_disp_update_list))
    print("VLM work per iteration: {}".format(vlm_work_list))
    print("Displacement input framework work per iteration: {}".format(total_framework_work_disp_inputs_list))
    print("FEniCS work per iteration: {}".format(fe_work_list))
    print("Displacement output framework work per iteration: {}".format(total_framework_work_disp_outputs_list))



    ####### Aerodynamic output ##########
    print("="*60)
    print("="*20+'aerodynamic outputs'+"="*20)
    print("="*60)
    print('Pitch: ', np.rad2deg(
        sim[system_model_name+cruise_name+'_ac_states_operation.'+cruise_name+'_pitch_angle']))
    print('C_L: ', sim[system_model_name+'wing_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_C_L'])
    # print('Total lift: ', sim[system_model_name+'wing_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.total_lift'])

    ####### Structural output ##########
    print("="*60)
    print("="*20+'structure outputs'+"="*20)
    print("="*60)
    # Comparing the solution to the Kirchhoff analytical solution
    f_shell = sim[system_model_name+'Wing_rm_shell_force_mapping.wing_shell_forces']
    f_vlm = sim[system_model_name+'wing_vlm_nodal_forces_model.wing_oml_forces'].reshape((-1,3))
    u_shell = sim[system_model_name+'Wing_rm_shell_model.rm_shell.disp_extraction_model.wing_shell_displacement']
    u_nodal = sim[system_model_name+'Wing_rm_shell_displacement_map.wing_shell_nodal_displacement']
    u_tip = sim[system_model_name+'Wing_rm_shell_displacement_map.wing_shell_tip_displacement']
    uZ = u_shell[:,2]
    # uZ_nodal = u_nodal[:,2]


    wing_von_Mises_stress = sim[system_model_name+'Wing_rm_shell_model.rm_shell.von_Mises_stress_model.wing_shell_stress']
    wing_mass = sim[system_model_name+'Wing_rm_shell_model.rm_shell.mass_model.mass']
    wing_elastic_energy = sim[system_model_name+'Wing_rm_shell_model.rm_shell.elastic_energy_model.elastic_energy']
    wing_aggregated_stress = sim[system_model_name+'Wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress']
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
    # print("vlm panel force sums: {}".format(np.sum(wing_vlm_panel_forces[0].value, axis=0)))
    print("vlm forces:", sum(f_vlm[:,0]),sum(f_vlm[:,1]),sum(f_vlm[:,2]))
    print("shell forces:", dolfinx.fem.assemble_scalar(form(fx_func*ufl.dx)),
                            dolfinx.fem.assemble_scalar(form(fy_func*ufl.dx)),
                            dolfinx.fem.assemble_scalar(form(fz_func*ufl.dx)))

    print("Wing surface area:", dolfinx.fem.assemble_scalar(form(dummy_func*ufl.dx)))
    print("Wing tip deflection (m):",max(abs(uZ)))
    print("Wing tip deflection computed by CSDL (m):",np.max(u_nodal))
    print("Wing tip deflection computed by CSDL (m):",u_tip)
    print("Wing total mass (kg):", wing_mass)
    print("Wing aggregated von Mises stress (Pascal):", wing_aggregated_stress)
    print("Wing maximum von Mises stress (Pascal):", max(wing_von_Mises_stress))
    print("Wing maximum von Mises stress on OML (Pascal):", wing_aggregated_stress)
    print("  Number of elements = "+str(nel))
    print("  Number of vertices = "+str(nn))

    # -----------------------------------------------
    # Code to compute the total forces in columns 2, 3 and 4 (M3L-internal mappings)
    # first column 1:
    vlm_internal_total_forces = np.sum(sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_total_forces'], axis=1)[0]

    # then, column 2:
    col2_proj_total_forces = np.sum(sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_nodal_forces_model.wing_oml_forces'], axis=0)

    # column 3 (framework):    
    # loop over keys in `wing_force.coefficients` dict
    wing_0_force = np.zeros((3,))
    wing_1_force = np.zeros((3,))
    for key in wing_force.coefficients:
        # query corresponding object in sim dict
        force_arr = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_inverse_evaluation.{}_wing_force_coefficients'.format(key)]
        summed_forces = np.sum(force_arr, axis=0)
        # separate left and right wing forces
        if 'Wing_0' in key:
            wing_0_force += summed_forces
        elif 'Wing_1' in key:
            wing_1_force += summed_forces
    
    # column 4 (OML mesh of solid solver)
    col4_oml_forces = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_evaluation.evaluated_wing_force_function']
    col4_proj_total_forces = np.sum(sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_evaluation.evaluated_wing_force_function'], axis=0)

    # column 5 (solid solver)
    col5_shell_force_coefficients = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_force_mapping.wing_shell_forces']


    # NOTE: The objects in sim[''] can be found by looking for their corresponding variable names in SUMMARY_GRAPH.txt
    # NOTE: wing_1_force matches the forces applied to the shell model!


    # -----------------------------------------------
    # Code to compute the (tip) displacements in columns 3, 4 and 5 (M3L-internal mappings)

    # The solid solver computes the displacements on the shell mesh in CG2, 
    # and then maps them to CG1 with the mapping approach that is covered in our SciTech 2023 submission.
    # The nodal association of the CG1 space is then used to construct a NodalMap for projection to the nodal OML mesh of column 4
    # TODO: Combine the maps from CG2 to the nodal OML mesh if necessary (might not be the case) 

    # column 5 (CG1 nodal displacements)
    wing_cg1_disp = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_model.rm_shell.disp_extraction_model.wing_shell_displacement']
    # column 5 (total solution vector)
    sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_model.rm_shell.solid_model.disp_solid']


    # finding NaNs in column 5 -> 4 map
    # nan_idxs = np.argwhere(np.isnan(sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_displacement_map.wing_shell_displacements_to_nodal_displacements']))
    # unique_idxs = np.unique(nan_idxs[:, 0])  # these indices correspond to the OML mesh DoF indices

    # # locations of OML nodes: transfer_geo_nodes_ma
    # # locations of shell nodes: shell_mesh.parameters['meshes']['wing_shell_mesh'].value
    # # compute NodalMap
    # col5to4_NodalMap = NodalMap(shell_mesh.parameters['meshes']['wing_shell_mesh'].value, transfer_geo_nodes_ma.value, RBF_width_par=20.0,
    #                         column_scaling_vec=shell_pde.bf_sup_sizes)
    

    # column 4 (nodal OML mesh displacements)
    wing_oml_disp = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_displacement_map.wing_shell_nodal_displacement']

    # column 3 (framework, after inverse_evaluate)
    # loop over keys in `wing_displacement.coefficients` dict
    # disp_list = []
    # for key in wing_displacement_output.coefficients:
    #     # query corresponding object in sim dict
    #     disp_arr = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_output_function_evaluation.{}_wing_displacement_coefficients'.format(key)]
    #     disp_list += [disp_arr]



    # -------------------------------
    # We verify whether the various maps that are used to couple VLM to the framework match with the ones that are computed manually for the work-conserving map
    disp_oml_to_vlm_map = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_vlm_nodal_displacements_model.wing_displacement_input_function_displacements_map']
    force_vlm_to_oml_map = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_nodal_forces_model.wing_force_map']
    # force_oml_to_framework_map = sim['']
    # we need to patch together the concatenated `force_oml_to_framework_map` from the maps to the various surfaces
    oml_to_framework_map_per_surface = []
    for key, value in associated_coords.items():
        oml_to_framework_map_surface = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_inverse_evaluation.fitting_matrix_{}'.format(key)]
        oml_to_framework_map_per_surface += [oml_to_framework_map_surface]
    
    oml_to_framework_maps_combined = block_diag(oml_to_framework_map_per_surface)

    diff_disp_oml_to_vlm_maps = np.linalg.norm(np.subtract(disp_oml_to_vlm_map, disp_composition_mat))
    diff_force_vlm_to_oml_maps = np.linalg.norm(np.subtract(force_vlm_to_oml_map, force_map_oml))
    diff_force_oml_to_framework_maps = np.linalg.norm(np.subtract(oml_to_framework_maps_combined.toarray(), fitting_matrix.toarray()))