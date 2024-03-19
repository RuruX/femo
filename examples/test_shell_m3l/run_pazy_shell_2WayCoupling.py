## Caddee
from caddee.core.caddee_core.system_representation.utils.mesh_utils import import_mesh as caddee_import_mesh
from caddee.utils.aircraft_models.pazy.pazy_geom_mesh import PazyGeomMesh
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
from scipy.sparse import csr_matrix, block_diag, dok_matrix, coo_matrix, issparse
from scipy.sparse import vstack as sparse_vstack
from mpi4py import MPI
import pickle
import pathlib
import sys
import meshio
from copy import deepcopy

sys.setrecursionlimit(100000)

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:geometrical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
                           "name_to_read": [cell_data]})
    return out_mesh

def construct_VLM_vertex_to_force_map(vertex_array_shape, F_chord_pos=0.25):
    # we assume that `vertex_array_shape` is a 4D array; 
    # its second index corresponds to chordwise nodes, and its third index to spanwise nodes
    node_map = np.zeros((vertex_array_shape[1]*vertex_array_shape[2], (vertex_array_shape[1]-1)*(vertex_array_shape[2]-1)))

    # loop over chordwise panels
    for i in range(vertex_array_shape[1]-1):
        # loop over spanwise panels
        for j in range(vertex_array_shape[2]-1):
            # define the panel number through a lexicographic ordering by first traversing the spanwise panels and then the chordwise panels
            # so node_map[0, 0] corresponds to the LE most outboard node on the right wing,
            # and node_map [1, 0] corresponds to the second-most outboard node on the LE of the right wing

            # similarly, node_map[0, 1] corresponds to the effect of the right wing's most outboard LE node on the second-most outboard LE panel
            panel_num = j + i*(vertex_array_shape[2]-1)
            node_map[j + i*vertex_array_shape[2], panel_num] = 0.5*(1-F_chord_pos)
            node_map[j + 1 + i*vertex_array_shape[2], panel_num] = 0.5*(1-F_chord_pos)
            node_map[j + (i + 1)*vertex_array_shape[2], panel_num] = 0.5*F_chord_pos
            node_map[j + 1 + (i + 1)*vertex_array_shape[2], panel_num] = 0.5*F_chord_pos
    return node_map

debug_geom_flag = False
force_reprojection = False
visualize_flag = False
# NOTE: Dashboard and xdmf recorder cannot be turned on at the same time
dashboard = False
xdmf_record = True

# flags that control whether couplings are handled in a conservative way
vlm_conservative = False
fenics_conservative = False

ft2m = 0.3048
in2m = 0.0254

# wing_cl0 = 0.3366
# pitch_angle_list = [-0.02403544, 6, 12.48100761]
# h_0 = 0.02*in2m

wing_cl0 = 0.0
pitch_angle_list = [-0.38129494, 6, 12.11391141]
h_0 = 0.05*in2m
pitch_angle_deg = -5.
pitch_angle = np.deg2rad(pitch_angle_deg)  #np.deg2rad(pitch_angle_list[2])

caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()

# region Geometry and meshes
pav_geom_mesh = PazyGeomMesh()
pav_geom_mesh.setup_geometry(
    include_wing_flag=True,
    include_htail_flag=False,
)
pav_geom_mesh.setup_internal_wingbox_geometry(debug_geom_flag=debug_geom_flag,
                                              force_reprojection=force_reprojection)
pav_geom_mesh.sys_rep.spatial_representation.assemble()
pav_geom_mesh.oml_mesh(include_wing_flag=True, 
                       grid_num_u=10, grid_num_v=10,
                       debug_geom_flag=debug_geom_flag, force_reprojection=force_reprojection)
pav_geom_mesh.vlm_meshes(include_wing_flag=True, num_wing_spanwise_vlm=29, num_wing_chordwise_vlm=11,
                         visualize_flag=visualize_flag, force_reprojection=force_reprojection)
pav_geom_mesh.setup_index_functions()

caddee.system_representation = sys_rep = pav_geom_mesh.sys_rep
caddee.system_parameterization = sys_param = pav_geom_mesh.sys_param
sys_param.setup()
spatial_rep = sys_rep.spatial_representation
# endregion

# export spatial representation to IGES file
if False:
    # Some lines of code to remove 'PazyWingGeom_0_2' and 'PazyWingGeom_0_3' from `spatial_rep.primitives`, so they aren't written to the IGES file
    primitives_new = {}
    indicies_new = {}
    for key, item in spatial_rep.primitives.items():
        if 'PazyWingGeom' not in key:
            primitives_new[key] = item

    for key, item in spatial_rep.primitive_indices.items():
        if 'PazyWingGeom' not in key:
            indicies_new[key] = item

    spatial_rep.primitives = primitives_new
    spatial_rep.primitive_indices = indicies_new
    spatial_rep.write_iges('./pazy_wing/pazy_wing_structure.iges')

# convert gmsh mesh to caddee mesh
if False:
    file = './pazy_wing/pazy_wing_gmsh_SI_quad_2992.msh'
    nodes, connectivity = caddee_import_mesh(file,
                                    spatial_rep,
                                    remove_dupes=True,
                                    tol=1e-8,
                                    grid_search_n=100)

    cells = [("quad",connectivity)]
    mesh = meshio.Mesh(nodes.value, cells)
    meshio.write('./pazy_wing/pazy_wing_caddee_mesh_' + str(nodes.shape[0]) + '_quad.xdmf', mesh, file_format='xdmf')

if False:
    mesh_from_file = meshio.read('./pazy_wing/pazy_wing_caddee_mesh_3213_quad.msh')
    quad_mesh = create_mesh(mesh_from_file, "quad")
    meshio.write('./pazy_wing/pazy_wing_caddee_mesh_3213_quad.xdmf', quad_mesh)

wing_component = pav_geom_mesh.geom_data['components']['wing']

# Instead of using the ordering of `disp_input_surface_names` below we look at 
# the associated coords of pav_geom_mesh.mesh_data['oml']['oml_para_nodes']['wing']
indexed_mesh = pav_geom_mesh.mesh_data['oml']['oml_para_nodes']['right_wing']
associated_coords = {}
index = 0
for item in indexed_mesh:
    key = item[0]
    value = item[1]
    if key not in associated_coords.keys():
        associated_coords[key] = [[index], value]
    else:
        associated_coords[key][0].append(index)
        associated_coords[key] = [associated_coords[key][0], np.vstack((associated_coords[key][1], value))]
    index += 1

# we first determine the framework surfaces that contain both displacements and forces 
disp_input_surface_names = associated_coords.keys()
force_surface_names = pav_geom_mesh.functions['wing_force'].space.spaces.keys()

framework_work_surface_names = []

for surf_name in disp_input_surface_names:
    if surf_name in force_surface_names:
        framework_work_surface_names += [surf_name]

disp_evaluationmaps = {}
disp_evaluationmaps_list = []

# loop over the surfaces that contain both forces and displacements and compute their invariant matrices
for surf_name in framework_work_surface_names:
    displacement_space = pav_geom_mesh.functions['wing_displacement_input'].space.spaces[surf_name]
    force_space = pav_geom_mesh.functions['wing_force'].space.spaces[surf_name]

    # NOTE: wing forces are currently represented by a set of vectors in these (parametric) points:
    force_parametricpoints = force_space.points
    # we sample the displacements in the same parametric points
    displacement_evaluationmap = displacement_space.compute_evaluation_map(force_parametricpoints)
    # NOTE: `displacement_evaluationmap` is effectively the invariant matrix of a single displacement-force component pair (in x, y or z) on surface `test_surf`
    disp_evaluationmaps[surf_name] = [displacement_evaluationmap]
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
filename = "pazy_wing/pazy_wing_caddee_mesh_3165_quad.xdmf"
# filename = "pazy_wing/pazy_wing_caddee_mesh_12633_quad.xdmf"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    fenics_mesh = xdmf.read_mesh(name="Grid")
nel = fenics_mesh.topology.index_map(fenics_mesh.topology.dim).size_local
nn = fenics_mesh.topology.index_map(0).size_local

nodes = fenics_mesh.geometry.x


if False:
    file_name = './pazy_wing/pazy_wing_mesh_data_3165_quad.pickle'

    nodes_parametric = []

    targets = spatial_rep.get_primitives(pav_geom_mesh.geom_data['primitive_names']['structural_right_wing_names'])
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
            distances[i,:] = np.linalg.norm(projected_points_on_each_target[i]['geometry'].value - nodes, axis=-1)
            # distances[i,:] = np.linalg.norm(projected_points_on_each_target[i]['geometry'].value - nodes.value, axis=-1)
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
    with open(file_name, 'wb') as f:
        pickle.dump(nodes_parametric, f)

# `shell_pde` contains the nodal displacement map
shell_pde = ShellPDE(fenics_mesh)

# Aluminum 7075 T6
# reference: "Experimental Aeroelastic Benchmark of a Very Flexible Wing", Avin et al., AIAA Journal Vol. 60, No. 3, 03/2022
E_Al = 71E9 # unit: Pa
nu_Al = 0.33
rho_Al = 2795 # unit: kg/m^3
tensile_yield_strength_Al = 468E6 # unit: Pa
G_Al = E_Al/2/(1+nu_Al)

# Nylon 12 (3D-printed)
# reference: "Experimental Aeroelastic Benchmark of a Very Flexible Wing", Avin et al., AIAA Journal Vol. 60, No. 3, 03/2022
E_N = 1.7E9 # unit: Pa
nu_N = 0.394
rho_N = 930 # unit: kg/m^3
tensile_yield_strength_N = 48E6 # unit: Pa
# E_N = 71E9 # unit: Pa
# nu_N = 0.33

G_N = E_N/2/(1+nu_N)

# thickness data in m
f_spar_t = 6e-3
r_spar_t = 5.8e-3
rib_t = 5e-3
skin_panel_t = 2e-3
stiffener_t = 2.25e-3

# First we import the nodes of the FEA mesh so we can project quantities to the shell mesh
with open('./pazy_wing/pazy_wing_mesh_data_3165_quad.pickle', 'rb') as f:
# with open('./pazy_wing/pazy_wing_mesh_data_12658_quad.pickle', 'rb') as f:
    nodes_parametric = pickle.load(f)

for i in range(len(nodes_parametric)):
    nodes_parametric[i] = (nodes_parametric[i][0].replace(' ', '_').replace(',',''), np.array([nodes_parametric[i][1]]))

# we set the material parameters defined above
wing_thickness = pav_geom_mesh.functions['wing_thickness']
wing_E = pav_geom_mesh.functions['wing_E']
wing_G = pav_geom_mesh.functions['wing_G']
wing_nu = pav_geom_mesh.functions['wing_nu']

for key, t_item in wing_thickness.coefficients.items():
    if 'f_spar' in key:
        t_item.value = f_spar_t*np.ones(t_item.shape)
    elif 'r_spar' in key:
        t_item.value = r_spar_t*np.ones(t_item.shape)
    elif 'rib' in key:
        t_item.value = rib_t*np.ones(t_item.shape)
    elif 'stiffener' in key:
        t_item.value = stiffener_t*np.ones(t_item.shape)
    elif 'f_spar' in key:
        t_item.value = f_spar_t*np.ones(t_item.shape)
    elif 'panel' in key:
        t_item.value = skin_panel_t*np.ones(t_item.shape)

for key, _ in wing_E.coefficients.items(): # loop over the wing_E coefficient items (these are identical for all material parameter functions)
    E_item = wing_E.coefficients[key]
    G_item = wing_G.coefficients[key]
    nu_item = wing_nu.coefficients[key]
    if 'stiffener' in key: # or 'rib' in key:
        # assign Al 7075 parameters
        E_item.value = E_Al*np.ones(E_item.shape)
        G_item.value = G_Al*np.ones(G_item.shape)
        nu_item.value = nu_Al*np.ones(nu_item.shape)
    else:
        # assign Nylon 12 parameters
        E_item.value = E_N*np.ones(E_item.shape)
        G_item.value = G_N*np.ones(G_item.shape)
        nu_item.value = nu_N*np.ones(nu_item.shape)

thickness_nodes = wing_thickness.evaluate(nodes_parametric)
E_nodes = wing_E.evaluate(nodes_parametric)
G_nodes = wing_G.evaluate(nodes_parametric)
nu_nodes = wing_nu.evaluate(nodes_parametric)


y_bc = 1e-8
semispan = 0.55

# Constructs FEniCS invariant matrix of force and displacement
# NOTE: DoFs seem to be ordered as [x_1, y_1, z_1, x_2, y_2, z_2, ...]
fenics_force_function = TestFunction(shell_pde.VF)
fenics_force_trialfunction = TrialFunction(shell_pde.VF)
fenics_disp_function = TrialFunction(shell_pde.W.sub(0).collapse()[0])
fenics_invariantmatrix_petsc = assemble_matrix(form(inner(fenics_force_function, fenics_disp_function)*dx))
fenics_invariantmatrix_petsc.assemble()

fenics_invariantmatrix_csr = fenics_invariantmatrix_petsc.getValuesCSR()
fenics_invariantmatrix_sp = csr_matrix((fenics_invariantmatrix_csr[2], fenics_invariantmatrix_csr[1], fenics_invariantmatrix_csr[0]))
# multiply the invariant matrix with the displacement extraction operator (the matrix that only retains the displacements and leaves out the rotations)
disp_extraction_matrix_list = shell_pde.construct_disp_extraction_mats()
disp_cg2_cg1_interpolation_map = shell_pde.construct_CG2_CG1_interpolation_map()
# disp_extraction_matrix = sparse_vstack([disp_cg2_cg1_interpolation_map@disp_extraction_mat for disp_extraction_mat in disp_extraction_matrix_list])
disp_extraction_matrix = sparse_vstack(disp_extraction_matrix_list)
# cg2_cg1_interpolations_stacked = sparse_vstack([disp_cg2_cg1_interpolation_map]*3)

# NOTE: indices of displacement component i are in shell_pde.W.sub(0).collapse()[0].sub(i).collapse()[1]

# construct FEniCS invariant matrix corresponding to a single force and displacement component
fenics_force_function_component = TestFunction(shell_pde.VF.sub(0).collapse()[0])
fenics_force_trialfunction_component = TrialFunction(shell_pde.VF.sub(0).collapse()[0])
fenics_disp_function_component = TrialFunction(shell_pde.W.sub(0).collapse()[0].sub(0).collapse()[0])
fenics_invariantmatrix_petsc_component = assemble_matrix(form(inner(fenics_force_function_component, fenics_disp_function_component)*dx))
fenics_invariantmatrix_petsc_component.assemble()

fenics_invariantmatrix_component_csr = fenics_invariantmatrix_petsc_component.getValuesCSR()
fenics_invariantmatrix_component_sp = csr_matrix((fenics_invariantmatrix_component_csr[2], fenics_invariantmatrix_component_csr[1], fenics_invariantmatrix_component_csr[0]))

# the matrix below is the complete FEniCS invariant matrix (for all components)!
fenics_invariantmatrix = fenics_invariantmatrix_sp@disp_extraction_matrix

# the matrix below is the FEniCS invariant matrix of a single force and displacement component
fenics_invariantmatrix_component = fenics_invariantmatrix_component_sp@disp_extraction_matrix_list[0]

fenics_cg1_invariantmatrix_petsc_component = assemble_matrix(form(inner(fenics_force_function_component, fenics_force_trialfunction_component)*dx))
fenics_cg1_invariantmatrix_petsc_component.assemble()

fenics_cg1_invariantmatrix_component_csr = fenics_cg1_invariantmatrix_petsc_component.getValuesCSR()
fenics_cg1_invariantmatrix_component_sp = csr_matrix((fenics_cg1_invariantmatrix_component_csr[2], fenics_cg1_invariantmatrix_component_csr[1], fenics_cg1_invariantmatrix_component_csr[0]))

#### Getting facets of the LEFT and the RIGHT edge  ####
DOLFIN_EPS = 3E-16
def ClampedBoundary(x):
    return np.less(x[1], y_bc)
def TipChar(x):
    return np.greater(x[1], semispan-y_bc)
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
shells['wing_shell'] = {#'E': E, 'nu': nu, 
                        'rho': rho_Al,# material properties
                        # TODO: make `rho` a function on the shell mesh as well
                        'dss': ds_1(100), # custom integrator: ds measure
                        'dSS': dS_1(100), # custom integrator: dS measure
                        'dxx': dx_2(10),  # custom integrator: dx measure
                        'g': g,
                        'record': xdmf_record}


################# PAV  Wing #################

# Wing shell Mesh
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
cruise_condition.set_module_input(name='altitude', val=0 * ft2m)
cruise_condition.set_module_input(name='mach_number', val=(50./30.)*0.08746)  # 112 mph = 0.145972 Mach
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

# construct solid displacement map from column 5 to 4
shell_nodal_displacements_model = rmshell.RMShellNodalDisplacements(component=wing_component,
                                                                    mesh=shell_mesh,
                                                                    pde=shell_pde,
                                                                    shells=shells)

grid_num = 10
transfer_para_mesh = []
structural_right_wing_names = pav_geom_mesh.geom_data['primitive_names']['structural_right_wing_names']
# left_wing_skin_names = pav_geom_mesh.geom_data['primitive_names']['left_wing_bottom_names'] + pav_geom_mesh.geom_data['primitive_names']['left_wing_top_names']

element_projection_names = pav_geom_mesh.geom_data['primitive_names']['right_wing']

for name in element_projection_names:
    for u in np.linspace(0,1,grid_num):
        for v in np.linspace(0,1,grid_num):
            transfer_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

transfer_geo_nodes_ma = spatial_rep.evaluate_parametric(transfer_para_mesh)

wing_oml_wing_mesh = pav_geom_mesh.mesh_data['oml']['oml_geo_nodes']['right_wing']


# Map displacements from column 3 (framework) to 2
wing_displacement_input = pav_geom_mesh.functions['wing_displacement_input']
wing_displacement_output = pav_geom_mesh.functions['wing_displacement_output']
wing_force = pav_geom_mesh.functions['wing_force']
oml_para_nodes_wing = pav_geom_mesh.mesh_data['oml']['oml_para_nodes']['right_wing']

# we convert `oml_para_nodes_wing` to physical space to check whether we're ordering the displacement OML points consistently
# NOTE: It seems like `oml_geo_nodes_wing.value` is identical to `wing_oml_wing_mesh.value`, so the displacement OML points are ordered consistently!
oml_geo_nodes_wing = spatial_rep.evaluate_parametric(oml_para_nodes_wing)

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
# register the total VLM forces and moments as outputs for the M3L Model (column 1)
cruise_model.register_output(vlm_forces)
cruise_model.register_output(vlm_moments)

if vlm_conservative:
    # compute displacement map from OML to VLM solver
    # TODO: Figure out the ordering of the nodes in the array below
    disp_oml_to_solver_map = vlm_disp_mapping_model.disp_map(vlm_disp_mapping_model.parameters['initial_meshes'][0], oml=wing_oml_wing_mesh)

    # compute displacement map from framework to OML
    associated_coords = {}
    index = 0
    for item in oml_para_nodes_wing:
        key = item[0]
        value = item[1]
        if key not in associated_coords.keys() or key not in framework_work_surface_names:
            associated_coords[key] = [[index], value]
        else:
            associated_coords[key][0].append(index)
            associated_coords[key] = [associated_coords[key][0], np.vstack((associated_coords[key][1], value))]
        index += 1

    # output_name = f'evaluated_{wing_displacement_input.name}'
    output_shape = (len(oml_para_nodes_wing), wing_displacement_input.coefficients[oml_para_nodes_wing[0][0]].shape[-1])
    
    coefficients_dict = {} 
    for key, coefficients in wing_displacement_input.coefficients.items():
        num_coefficients = np.prod(coefficients.shape[:-1])
        coefficients_dict[key] = coefficients.value.reshape((-1, coefficients.shape[-1]))

    evaluation_matrix_list = []
    # NOTE: We use the ordering from `framework_work_surface_names` below to make the mapping matrix ordering consistent with the framework invariant matrix
    for ordered_key in framework_work_surface_names:
        value = associated_coords[ordered_key]
        evaluation_matrix = wing_displacement_input.space.spaces[ordered_key].compute_evaluation_map(value[1])

        evaluation_matrix_list += [coo_matrix(evaluation_matrix)]
        
    disp_framework_to_oml_map = block_diag(evaluation_matrix_list)
    # coefficients_mat = np.vstack(coefficients_list)
    # coefficients_list = coefficients_mat.reshape(order='F')

    # Compute the framework forces
    wing_force.inverse_evaluate_conservative(framework_work_surface_names, wing_vlm_panel_forces[0], 
                                     otherside_composed_mat=csr_matrix(disp_oml_to_solver_map)@disp_framework_to_oml_map,
                                     solver_invariant_mat=coo_matrix(vlm_invariantmatrix_nonstacked), 
                                     framework_invariant_mat=framework_invariantmatrix_nonstacked.T)
    cruise_model.register_output(wing_force.coefficients)

else:
    # construct operation to map from column 1 to column 2
    oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=wing_vlm_panel_forces,
                                                nodal_force_meshes=[wing_oml_wing_mesh, ])
    wing_forces = oml_forces[0]
    # print("post-VLM force map evaluate")
    # endregion

    # region Structural Loads

    # map forces from column 2 to column 3 (framework) 
    wing_force.inverse_evaluate(oml_para_nodes_wing, wing_forces)
    cruise_model.register_output(wing_force.coefficients)

# left_wing_oml_para_coords = pav_geom_mesh.mesh_data['oml']['oml_para_nodes']['left_wing']
right_wing_oml_para_coords = pav_geom_mesh.mesh_data['oml']['oml_para_nodes']['right_wing']
# left_oml_geo_nodes = pav_geom_mesh.mesh_data['oml']['oml_geo_nodes']['left_wing']  #spatial_rep.evaluate_parametric(left_wing_oml_para_coords)
right_oml_geo_nodes = pav_geom_mesh.mesh_data['oml']['oml_geo_nodes']['right_wing']

if fenics_conservative:
    # construct matrix that maps the CG1 shell displacements to the OML
    mesh_original = wing_shell_mesh.value.reshape((-1,3))

    # mesh_mirrored = deepcopy(mesh_original)
    # mesh_mirrored[:, 1] *= -1.  # mirror coordinates along the y-axis
    # we concatenate both meshes (original and mirrored) and compute the displacement map
    # mesh_concat = np.vstack([mesh_original, mesh_mirrored])
    displacement_map = shell_nodal_displacements_model.umap(mesh_original,
                    oml=transfer_geo_nodes_ma.value.reshape((-1,3)),
                    repeat_bf_sup_size_vector=False)

    # we create a matrix that repeats the shell displacement variables twice
    # the shape of the shell displacement variables coincides with the number of fenics mesh nodes
    # rep_mat = np.vstack([np.eye(fenics_mesh.geometry.x.shape[0])]*2)
    # we manually set the fifth entry of rep_mat to `-1`, since the y-displacement is mirrored
    # rep_mat[4, 4] = -1.
    
    # combine maps into map of CG1 displacements to OML displacements
    disp_solver_to_oml_map = displacement_map  # @rep_mat

    # we multiply `disp_solver_to_oml_map` with the CG2-CG1 displacement projection matrix (NOTE: Operation moved to invariant matrix)
    disp_solver_to_oml_map = disp_solver_to_oml_map  # @disp_cg2_cg1_interpolation_map

    # Next we construct the OML -> framework displacement map

    associated_coords = {}
    index = 0
    for item in transfer_para_mesh:
        key = item[0]
        value = item[1]
        if key not in associated_coords.keys():
            associated_coords[key] = [[index], value]
        else:
            associated_coords[key][0].append(index)
            associated_coords[key] = [associated_coords[key][0], np.vstack((associated_coords[key][1], value))]
        index += 1

    # function_values = csdl_model.register_module_input('function_values', shape=self.arguments['function_values'].shape)
    # function_values = csdl.reshape(function_values, output_shape)
    # csdl_model.register_module_output('test_function_values', function_values)

    # loop over the framework surfaces in the order in which they are used in the framework invariant matrix, as defined in `framework_work_surface_names`
    fitting_matrix_per_surface_list = []
    for key in framework_work_surface_names:
        value = associated_coords[key]
        if hasattr(wing_displacement_output.space.spaces[key], 'compute_fitting_map'):
            fitting_matrix = wing_displacement_output.space.spaces[key].compute_fitting_map(value[1])
        else:
            evaluation_matrix = wing_displacement_output.space.spaces[key].compute_evaluation_map(value[1])
            if issparse(evaluation_matrix):
                evaluation_matrix = evaluation_matrix.toarray()
            # if self.regularization_coeff is not None:
                # fitting_matrix = np.linalg.inv(evaluation_matrix.T@evaluation_matrix + self.regularization_coeff*np.eye(evaluation_matrix.shape[1]))@evaluation_matrix.T # tested with 1e-3
            # else:
            fitting_matrix = np.linalg.pinv(evaluation_matrix)

        fitting_matrix_per_surface_list += [fitting_matrix]

    solid_disp_oml_to_framework_map = block_diag(fitting_matrix_per_surface_list)
    composed_disp_maps = solid_disp_oml_to_framework_map@disp_solver_to_oml_map

    # left_wing_forces = wing_force.evaluate(left_wing_oml_para_coords)

    # Debugging TODO's:
    #                   x compare displacement mapping matrices from above with the actual matrices that are used
    #                   - do pen-and-paper derivation of work conservation for a blockwise-diagonal system of 2 blocks, one for each wing
    #                   - check whether defining a left-wing framework invariant matrix is useful (would add a projection step of mirroring the displacements) 
    cruise_structural_wing_mesh_forces = wing_force.evaluate_conservative(framework_work_surface_names, 
                                     csr_matrix(composed_disp_maps),
                                     csr_matrix(fenics_cg1_invariantmatrix_component_sp.T),  
                                     framework_invariantmatrix_nonstacked.T,
                                     'wing_shell_forces',
                                     fenics_mesh.geometry.x.shape)
    # cruise_model.register_output(cruise_structural_wing_mesh_forces)

else:
    # map forces from column 3 to 4 (`evaluate` is used since we're mapping out from the framework representation)
    left_wing_forces = wing_force.evaluate(right_wing_oml_para_coords)

    # Define force map that takes nodal forces and projects them to the shell mesh (column 4 to 5)
    shell_force_map_model = rmshell.RMShellForces(component=wing_component,
                                                    mesh=shell_mesh,
                                                    pde=shell_pde,
                                                    shells=shells)
    cruise_structural_wing_mesh_forces = shell_force_map_model.evaluate(
                            nodal_forces=left_wing_forces,
                            nodal_forces_mesh=right_oml_geo_nodes)
# endregion

# region Structures
# run shell simulation
shell_displacements_model = rmshell.RMShell(component=wing_component,
                                            mesh=shell_mesh,
                                            pde=shell_pde,
                                            shells=shells)

cruise_structural_wing_mesh_displacements, _, cruise_structural_wing_mesh_stresses, wing_mass = \
                                shell_displacements_model.evaluate(
                                    forces=cruise_structural_wing_mesh_forces,
                                    thicknesses=thickness_nodes, E_moduli=E_nodes, G_moduli=G_nodes, nus=nu_nodes)
cruise_model.register_output(cruise_structural_wing_mesh_stresses)
cruise_model.register_output(cruise_structural_wing_mesh_displacements)
cruise_model.register_output(wing_mass)

# endregion

# print("post-RMshell evaluate")
# region Nodal Displacements

# map displacements from column 5 to 4
nodal_displacements, tip_displacement = shell_nodal_displacements_model.evaluate(cruise_structural_wing_mesh_displacements, transfer_geo_nodes_ma)

# construct map from column 4 to 3 (the framework representation)
wing_displacement_output.inverse_evaluate(transfer_para_mesh, nodal_displacements)
cruise_model.register_output(wing_displacement_output.coefficients)

wing_stress = pav_geom_mesh.functions['wing_stress']
wing_stress.inverse_evaluate(nodes_parametric, cruise_structural_wing_mesh_stresses, regularization_coeff=1e-3)
cruise_model.register_output(wing_stress.coefficients)

cruise_model.register_output(tip_displacement)
cruise_model.register_output(nodal_displacements)

# lastly we reflect the left wing displacements to the right wing 
# in a separate framework function object -- this object is used in the two-way coupling
if  False:
    rightwing_disp_reflection_model = rmshell.OMLPointsFromReflection(source_indexed_physical_coordinates=left_oml_geo_nodes.value,
                                                                    target_indexed_physical_coordinates=right_oml_geo_nodes.value,
                                                                    RBF_eps=5.)
    rightwing_reflected_disp = rightwing_disp_reflection_model.evaluate(nodal_displacements)
    # we also map the right wing OML displacement to the framework
    wing_displacement_rightwing = pav_geom_mesh.functions['wing_displacement_output_rightwing']
    wing_displacement_rightwing.inverse_evaluate(right_wing_oml_para_coords, rightwing_reflected_disp)
    cruise_model.register_output(wing_displacement_rightwing.coefficients)

# endregion

# Add cruise m3l model to cruise condition
cruise_condition.add_m3l_model('cruise_model', cruise_model)
# Add design condition to design scenario
design_scenario.add_design_condition(cruise_condition)

system_model.add_design_scenario(design_scenario=design_scenario)

caddee_csdl_model = caddee.assemble_csdl()

system_model_name = 'system_model.'+design_scenario_name+'.'+cruise_name+'.'+cruise_name+'.'


# caddee_csdl_model.add_constraint(system_model_name+'Wing_rm_shell_displacement_map.wing_shell_tip_displacement',upper=0.1,scaler=1E1)
# caddee_csdl_model.add_constraint(system_model_name+'Wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress',upper=324E6/1.5,scaler=1E-8)
# caddee_csdl_model.add_objective(system_model_name+'Wing_rm_shell_model.rm_shell.mass_model.mass', scaler=1e-1)

# Minimum thickness: 0.02 inch -> 0.000508 m
# h_min = h

i = 0
shape = (9, 1)
valid_structural_left_wing_names = structural_right_wing_names

################################################################
#### Full thicknesses: individual for spars, skins and ribs ####
################################################################
# for name in valid_structural_left_wing_names:
#     primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
#     name = name.replace(' ', '_').replace(',','')
#     surface_id = i

#     h_init = caddee_csdl_model.create_input('wing_thickness_dv_'+name, val=h_min)
#     caddee_csdl_model.add_design_variable('wing_thickness_dv_'+name, # 0.02 in
#                                           lower=0.005 * in2m,
#                                           upper=0.1 * in2m,
#                                           scaler=1000,
#                                           )
#     caddee_csdl_model.register_output('wing_thickness_surface_'+name, csdl.expand(h_init, shape))
#     caddee_csdl_model.connect('wing_thickness_surface_'+name,
#                                 system_model_name+'wing_thickness_function_evaluation.'+\
#                                 name+'_wing_thickness_coefficients')
#     i += 1

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

    # array_leftwing_update_norms = np.zeros((len(wing_displacement_output.coefficients),))
    array_rightwing_update_norms = np.zeros((len(framework_work_surface_names),))
    vlm_force_list = []
    shell_force_list = []
    leftwing_force_list = []
    rightwing_force_list = []
    max_disp_update_list = []
    vlm_work_list = []
    total_framework_work_disp_inputs_list = []
    leftwing_framework_work_list = []
    rightwing_framework_work_list = []
    fe_work_list = []
    total_framework_work_disp_outputs_list = []

    fenics_wingtip_disp_list = []
    framework_wingtip_disp_list = []
    vlm_wingtip_disp_list = []

    framework_max_stress_list = []
    fenics_max_stress_list = []

    save_results_dict = {}

    running = True
    # initialize iteration loop
    iter_idx = 0
    while running:
        print("---"*10)
        print("Iteration {}".format(iter_idx))
        # set displacement inputs
        if iter_idx > 0:
            # first we copy over the wing displacements
            for i, key in enumerate(framework_work_surface_names):
                sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_evaluation.{}_wing_displacement_input_coefficients'.format(key)] = disp_rightwing_output_list[i]

        sim.run()

        # then we do the same thing for the right wing displacements
        disp_rightwing_input_list = []
        disp_rightwing_output_list = []
        rightwing_force_sums = np.zeros((3,))
        for i, key in enumerate(framework_work_surface_names):
            # query corresponding object in sim dict
            # NOTE: At the moment we only map the displacement outputs on PazyWingGeo_0_2 and PazyWingGeo_0_3 back to the input (since the internal structure inputs are unused at the moment)
            displacement_array_input = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_evaluation.{}_wing_displacement_input_coefficients'.format(key)]
            displacement_array_output = deepcopy(sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_output_function_inverse_evaluation.{}_wing_displacement_output_coefficients'.format(key)])
            array_rightwing_update_norms[i] = np.linalg.norm(np.subtract(displacement_array_input, displacement_array_output))#/np.linalg.norm(displacement_array_output)

            rightwing_force_sums += sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_inverse_evaluation.{}_wing_force_coefficients'.format(key)].sum(axis=0)

            disp_rightwing_input_list += [displacement_array_input]
            disp_rightwing_output_list += [displacement_array_output]


        vlm_force_list += [np.sum(sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_total_forces'], axis=1)[0]]
        max_disp_update_list += [array_rightwing_update_norms.max()]
        rightwing_force_list += [rightwing_force_sums]


        print("Max right wing 2-norm update: {}".format(array_rightwing_update_norms.max()))
        print("Right wing 2-norm updates: {}".format(array_rightwing_update_norms))

        if (array_rightwing_update_norms.max() < 1e-12) or iter_idx >= 10:
            running = False

        # ----------------------------------------- #
        # Below we compute the aeroelastic work with the various displacement and force variables & the invariant matrices:
        vlm_F = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_total_forces'][0, :, :]
        # the panel forces are just the forces coming from the panels, without force corrections
        # vlm_panel_forces = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.panel_forces'][0, :, :]
        # vlm_panel_forces[:, 0] *= -1.
        # vlm_panel_forces[:, 2] *= -1.
        # vlm_F_3d = vlm_F.reshape((6, 30, 3), order='C')
        # vlm_F_reshaped = vlm_F_3d.reshape((180, 3), order='F')

        vlm_u = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_model.vast.wing_mesh_displacements'][0, :, :, :]
        vlm_u_2d = np.reshape(vlm_u, (vlm_u.shape[0]*vlm_u.shape[1], vlm_u.shape[2]), order='C')
        
        # vlm_u_2d_x = vlm_u[:, :, 0].flatten(order='C')
        # vlm_u_2d_y = vlm_u[:, :, 1].flatten(order='C')
        # vlm_u_2d_z = vlm_u[:, :, 2].flatten(order='C')
        # vlm_u_2d_reshaped = np.vstack([vlm_u_2d_x, vlm_u_2d_y, vlm_u_2d_z]).T

        # vlm_u_flat = vlm_u_2d.flatten(order='F')
        vlm_work_tensor = vlm_F.T@vlm_invariantmatrix_nonstacked.T@vlm_u_2d
        vlm_work = np.diag(vlm_work_tensor).sum()

        # Framework work with input displacements
        total_framework_work_disp_inputs = 0.
        for surf_name in framework_work_surface_names:
            surf_disp = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_evaluation.{}_wing_displacement_input_coefficients'.format(surf_name)]
            # surf_disp_flat = surf_disp.flatten(order='F')
            surf_force = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_inverse_evaluation.{}_wing_force_coefficients'.format(surf_name)]
            # surf_force_flat = surf_force.flatten(order='F')
            surf_invariantmatrix = disp_evaluationmaps[surf_name][0]
            surf_work_mat = surf_force.T@surf_invariantmatrix@surf_disp
            surf_work = np.diag(surf_work_mat).sum()

            total_framework_work_disp_inputs += surf_work
            # if surf_name in framework_work_left_wing_surface_names:
            #     leftwing_framework_work_disp_inputs += surf_work
            # if surf_name in framework_work_right_wing_surface_names:
            #     rightwing_framework_work_disp_inputs += surf_work

        # FEniCS work (F^T@mat@u):
        if fenics_conservative:
            fe_F = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_evaluation.wing_shell_forces']
            # fe_F_flat = fe_F.flatten(order='C')
        else:
            fe_F = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_force_mapping.wing_shell_forces']

        fe_u = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_model.rm_shell.solid_model.disp_solid']

        extracted_solid_disp = [disp_extraction_matrix_list[i]@fe_u for i in range(3)]
        extracted_solid_disp = np.array(extracted_solid_disp)
        cg1_disp = disp_cg2_cg1_interpolation_map@extracted_solid_disp.T
        fe_work = np.sum(np.diag(cg1_disp.T@fenics_cg1_invariantmatrix_component_sp@fe_F))
        # else:
        #     fe_F = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_force_mapping.wing_shell_forces']
        #     fe_F_flat = fe_F.flatten(order='C')

        #     fe_u = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_model.rm_shell.solid_model.disp_solid']

        #     fe_work = fe_F_flat@fenics_invariantmatrix@fe_u

        # Framework work with output displacements
        total_framework_work_disp_outputs = 0.
        for surf_name in framework_work_surface_names:
            surf_disp = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_output_function_inverse_evaluation.{}_wing_displacement_output_coefficients'.format(surf_name)]
            # surf_disp_flat = surf_disp.flatten(order='F')
            surf_force = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_inverse_evaluation.{}_wing_force_coefficients'.format(surf_name)]
            # surf_force_flat = surf_force.flatten(order='F')
            surf_invariantmatrix = disp_evaluationmaps[surf_name][0]
            surf_work_mat = surf_force.T@surf_invariantmatrix@surf_disp
            surf_work = np.diag(surf_work_mat).sum()

            total_framework_work_disp_outputs += surf_work

        print("VLM work: {}".format(vlm_work))
        print("Displacement input total framework work: {}".format(total_framework_work_disp_inputs))
        print("FEniCS work: {}".format(fe_work))
        print("Displacement output total framework work: {}".format(total_framework_work_disp_outputs))
        vlm_work_list += [vlm_work]
        total_framework_work_disp_inputs_list += [total_framework_work_disp_inputs]
        fe_work_list += [fe_work]
        total_framework_work_disp_outputs_list += [total_framework_work_disp_outputs]

        u_shell = sim[system_model_name+'Wing_rm_shell_model.rm_shell.disp_extraction_model.wing_shell_displacement']
        uZ = u_shell[:,2]
        print("Wing tip deflection (m):",max(abs(uZ)))

        # compute fenics forces
        if fenics_conservative:
            f_shell = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_evaluation.wing_shell_forces']
        else:
            f_shell = sim[system_model_name+'Wing_rm_shell_force_mapping.wing_shell_forces']
        fz_func = Function(shell_pde.VT)
        fz_func.x.array[:] = f_shell[:,-1]

        fx_func = Function(shell_pde.VT)
        fx_func.x.array[:] = f_shell[:,0]

        fy_func = Function(shell_pde.VT)
        fy_func.x.array[:] = f_shell[:,1]

        shell_forces = (dolfinx.fem.assemble_scalar(form(fx_func*ufl.dx)), dolfinx.fem.assemble_scalar(form(fy_func*ufl.dx)), dolfinx.fem.assemble_scalar(form(fz_func*ufl.dx)))
        print("shell forces:", shell_forces)
        shell_force_list += [shell_forces]

        # compute the max Von Mises stresses in the framework and FEniCS
        # first the framework:
        max_von_mises_stress_framework = 0.
        for right_wing_struct_name in structural_right_wing_names:
            struct_surf_coefficients = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_stress_function_inverse_evaluation.{}_wing_stress_coefficients'.format(right_wing_struct_name)]
            max_von_mises_stress_framework = np.max([max_von_mises_stress_framework, struct_surf_coefficients.max()])            

        # then in FEniCS:
        max_von_mises_stress_fenics = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_model.rm_shell.von_Mises_stress_model.wing_shell_stress'].max()

        framework_max_stress_list += [max_von_mises_stress_framework]
        fenics_max_stress_list += [max_von_mises_stress_fenics]

        # compute the tip displacements
        u_shell = sim[system_model_name+'Wing_rm_shell_model.rm_shell.disp_extraction_model.wing_shell_displacement']
        uZ = u_shell[:,2]
        u_nodal = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_evaluation.evaluated_wing_displacement_input_function']
        fenics_wingtip_disp_list += [max(abs(uZ))]
        framework_wingtip_disp_list += [np.abs(u_nodal[:, 2]).max()]
        vlm_wingtip_disp_list += [np.abs(vlm_u[:, :, 2]).max()]

        iter_idx += 1


    # save vlm mesh and input/output fields
    # NOTE: All forces are in the body reference frame and need to be converted to the inertial frame to compute lift and drag
    save_results_dict['vlm_forcefields'] = vlm_F
    save_results_dict['vlm_displacementfields'] = vlm_u
    save_results_dict['vlm_undeformed_mesh'] = vlm_disp_mapping_model.parameters['initial_meshes'][0]
    # save various work lists
    save_results_dict['vlm_work_list'] = vlm_work_list
    save_results_dict['framework_total_work_disp_inputs_list'] = total_framework_work_disp_inputs_list
    # save_results_dict['framework_leftwing_work_list'] = leftwing_framework_work_list
    # save_results_dict['framework_rightwing_work_list'] = rightwing_framework_work_list
    save_results_dict['fenics_work_list'] = fe_work_list
    # save various force tuple lists
    save_results_dict['vlm_force_list'] = vlm_force_list
    # save_results_dict['framework_leftwing_force_list'] = leftwing_force_list
    save_results_dict['framework_rightwing_force_list'] = rightwing_force_list
    save_results_dict['fenics_force_list'] = shell_force_list
    # save tip displacement lists
    save_results_dict['vlm_wingtip_disp_list'] = vlm_wingtip_disp_list
    save_results_dict['framework_wingtip_disp_list'] = framework_wingtip_disp_list
    save_results_dict['fenics_wingtip_disp_list'] = fenics_wingtip_disp_list
    # save max stresses
    save_results_dict['fenics_max_stress_list'] = fenics_max_stress_list
    save_results_dict['framework_max_stress_list'] = framework_max_stress_list 

    # compute the dynamic pressure
    vel_mag = sim['system_model.structural_sizing.cruise_3.cruise_3.cruise_3_ac_states_operation.cruise_3_speed']
    air_density = sim['system_model.structural_sizing.cruise_3.cruise_3.cruise_3_ac_states_operation.atmosphere_model.cruise_3_density']
    dynamic_pressure = 0.5*air_density*(vel_mag**2)
    dynamic_pressure_int = np.rint(dynamic_pressure)

    # save save_results_dict
    np.save('records/pazy_postprocess_dict_{}deg_pitch_{}pa_dyn_pres.npy'.format(int(pitch_angle_deg), int(dynamic_pressure_int)), save_results_dict)


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
    if fenics_conservative:
        f_shell = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_evaluation.wing_shell_forces']
    else:
        f_shell = sim[system_model_name+'Wing_rm_shell_force_mapping.wing_shell_forces']
    f_vlm = sim[system_model_name+'wing_vlm_model.vast.VLMSolverModel.VLM_outputs.LiftDrag.wing_total_forces'].reshape((-1,3))
    u_shell = sim[system_model_name+'Wing_rm_shell_model.rm_shell.disp_extraction_model.wing_shell_displacement']
    u_nodal = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_evaluation.evaluated_wing_displacement_input_function']
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

    # print("Wing surface area:", dolfinx.fem.assemble_scalar(form(dummy_func*ufl.dx)))
    print("Wing tip deflection FEniCS (m):",max(abs(uZ)))
    print("Wing tip deflection framework (m):",np.abs(u_nodal[:, 2]).max())
    print("Wing tip deflection VLM (m):",np.abs(vlm_u[:, :, 2]).max())
    # print("Wing tip deflection computed by CSDL (m):",u_tip)
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
    print("VLM internal total forces: {}".format(vlm_internal_total_forces))
    # then, column 2:
    # this parameter will not exist if the conservative map is used
    # col2_proj_total_forces = np.sum(sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_nodal_forces_model.wing_oml_forces'], axis=0)

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
    print("Framework wing 0 total forces: {}".format(wing_0_force))
    print("Framework wing 1 total forces: {}".format(wing_1_force))
    print("Framework total aero forces: {}".format(np.add(wing_0_force, wing_1_force)))
    # column 4 (OML mesh of solid solver)
    # col4_oml_forces = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_evaluation.evaluated_wing_force_function']
    # col4_proj_total_forces = np.sum(sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_evaluation.evaluated_wing_force_function'], axis=0)

    # # column 5 (solid solver)
    # col5_shell_force_coefficients = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_force_mapping.wing_shell_forces']


    # NOTE: The objects in sim[''] can be found by looking for their corresponding variable names in SUMMARY_GRAPH.txt
    # NOTE: wing_1_force matches the forces applied to the shell model!


    # -----------------------------------------------
    # Code to compute the (tip) displacements in columns 3, 4 and 5 (M3L-internal mappings)

    # The solid solver computes the displacements on the shell mesh in CG2, 
    # and then maps them to CG1 with the mapping approach that is covered in our SciTech 2023 submission.
    # The nodal association of the CG1 space is then used to construct a NodalMap for projection to the nodal OML mesh of column 4
    # TODO: Combine the maps from CG2 to the nodal OML mesh if necessary (might not be the case) 

    # column 5 (CG1 nodal displacements)
    # wing_cg1_disp = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_model.rm_shell.disp_extraction_model.wing_shell_displacement']
    # column 5 (total solution vector)
    # sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_model.rm_shell.solid_model.disp_solid']

    # column 4 (nodal OML mesh displacements)
    # wing_oml_disp = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_displacement_map.wing_shell_nodal_displacement']

    # column 3 (framework, after inverse_evaluate)
    # loop over keys in `wing_displacement.coefficients` dict
    # disp_list = []
    # for key in wing_displacement_output.coefficients:
    #     # query corresponding object in sim dict
    #     disp_arr = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_output_function_evaluation.{}_wing_displacement_coefficients'.format(key)]
    #     disp_list += [disp_arr]

    # column 2 (nodal OML mesh displacements)
    # sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_evaluation.evaluated_wing_displacement_input_function']


    # -------------------------------
    # We verify whether the various maps that are used to couple VLM to the framework match with the ones that are computed manually for the work-conserving map
    # disp_oml_to_vlm_map = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_vlm_nodal_displacements_model.wing_displacement_input_function_displacements_map']
    # # force_vlm_to_oml_map = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_nodal_forces_model.wing_force_map']
    
    # # we need to patch together the concatenated `disp_framework_to_oml_map` from the maps of the various surfaces,
    # disp_framework_to_oml_map_per_surface = []
    # for ordered_key in framework_work_surface_names:
    #     evaluation_matrix = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_input_function_evaluation.evaluation_matrix_{}'.format(key)]

    #     disp_framework_to_oml_map_per_surface += [coo_matrix(evaluation_matrix)]
    
    # disp_framework_to_oml_maps_combined = block_diag(disp_framework_to_oml_map_per_surface)

    # # and we need to patch together the concatenated `force_oml_to_framework_map`
    # oml_to_framework_map_per_surface = []

    # associated_coords = {}
    # index = 0
    # for item in oml_para_nodes_wing:
    #     key = item[0]
    #     value = item[1]
    #     if key not in associated_coords.keys():
    #         associated_coords[key] = [[index], value]
    #     else:
    #         associated_coords[key][0].append(index)
    #         associated_coords[key] = [associated_coords[key][0], np.vstack((associated_coords[key][1], value))]
    #     index += 1

    # for key, value in associated_coords.items():
    #     oml_to_framework_map_surface = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_inverse_evaluation.fitting_matrix_{}'.format(key)]
    #     oml_to_framework_map_per_surface += [oml_to_framework_map_surface]

    # oml_to_framework_maps_combined = block_diag(oml_to_framework_map_per_surface)

    # # diff_disp_oml_to_vlm_maps = np.linalg.norm(np.subtract(disp_oml_to_vlm_map, disp_composition_mat))
    # # diff_force_vlm_to_oml_maps = np.linalg.norm(np.subtract(force_vlm_to_oml_map, force_map_oml))
    # # diff_force_oml_to_framework_maps = np.linalg.norm(np.subtract(oml_to_framework_maps_combined.toarray(), fitting_matrix.toarray()))

    # # the matrix below reflects the FEniCS displacements to the right wing
    # disp_fenics_to_oml_repeater_mat = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_displacement_map.wing_shell_displacement_repeater_mat']
    # disp_fenics_to_oml_mat_ref = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_displacement_map.wing_shell_displacements_to_nodal_displacements']

    # # construct
    # disp_oml_to_framework_map_per_surface = []
    # for ordered_key in framework_work_surface_names:
    #     evaluation_matrix = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_output_function_inverse_evaluation.fitting_matrix_{}'.format(key)]

    #     disp_oml_to_framework_map_per_surface += [coo_matrix(evaluation_matrix)]
    
    # disp_oml_to_framework_maps_combined = block_diag(disp_oml_to_framework_map_per_surface)
