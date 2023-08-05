"""
Structural analysis of the PAV wing
with the Reissner--Mindlin shell model

-----------------------------------------------------------
Test the integration of m3l and shell model
-----------------------------------------------------------
"""

## Caddee
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component
import caddee.core.primitives.bsplines.bspline_functions as bsf
from caddee.core.caddee_core.system_representation.system_primitive.system_primitive import SystemPrimitive
from caddee.core.caddee_core.system_representation.spatial_representation import SpatialRepresentation
from caddee.core.caddee_core.system_representation.utils.mesh_utils import import_mesh as caddee_import_mesh
from caddee import GEOMETRY_FILES_FOLDER
import caddee.api as cd
from caddee.utils.aircraft_models.pav.pav_weight import PavMassProperties

## Solvers
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from VAST.core.generate_mappings_m3l import VASTNodalForces
from VAST.core.vlm_llt.viscous_correction import ViscousCorrectionModel
from lsdo_airfoil.core.pressure_profile import PressureProfile, NodalPressureProfile, NodalForces
import dolfinx
from femo.fea.utils_dolfinx import *
import shell_module as rmshell
from shell_pde import ShellPDE

# Other lsdo lab stuff
import csdl
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import m3l
from m3l.utils.utils import index_functions
import lsdo_geo as lg
import array_mapper as am
from m3l.core.function_spaces import IDWFunctionSpace, IDWFunctionSpace2

## Other stuff
import numpy as np
from mpi4py import MPI
import pickle
import pathlib
import sys
import meshio


sys.setrecursionlimit(100000)

in2m = 0.0254
ft2m = 0.3048
lbs2kg = 0.453592
psf2pa = 50

wing_cl0 = 0.3662
pitch_angle_list = [-0.38129494, 6, 12.11391141]
h_0 = 0.05*in2m
pitch_angle = np.deg2rad(pitch_angle_list[2])

debug_geom_flag = False
visualize_flag = False
do_plots = False
force_reprojection = False


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
file_name = '/pav_wing/pav_new_SI.stp'
cfile = str(pathlib.Path(__file__).parent.resolve())
spatial_rep.import_file(file_name=cfile+file_name)
spatial_rep.refit_geometry(file_name=cfile+file_name)

# region Geometry processing/id

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
for i in range(22+172,37+172):
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

right_wing_names = []
right_wing_top_names = []
right_wing_bottom_names = []
right_wing_te_top_names = []
right_wing_te_bottom_names = []
for i in range(174,189):
    surf_name = 'Wing_0_' + str(i)
    right_wing_names.append(surf_name)
    if i%4 == 2:
        right_wing_te_bottom_names.append(surf_name)
    elif i%4 == 3:
        right_wing_bottom_names.append(surf_name)
    elif i%4 == 0:
        right_wing_top_names.append(surf_name)
    else:
        right_wing_te_top_names.append(surf_name)

# Important points from openVSP
root_te = np.array([15.170, 0., 1.961]) * ft2m
root_le = np.array([8.800, 0, 1.989]) * ft2m
l_tip_te = np.array([11.300, -14.000, 1.978]) * ft2m
l_tip_le = np.array([8.796, -14.000, 1.989]) * ft2m
r_tip_te = np.array([11.300, 14.000, 1.978]) * ft2m
r_tip_le = np.array([8.796, 14.000, 1.989]) * ft2m

# endregion

# region Components

#wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=list(spatial_rep.primitives.keys()))
wing_left = LiftingSurface(name='wing_left', spatial_representation=spatial_rep, primitive_names=left_wing_names)
wing_left_top = LiftingSurface(name='wing_left_top', spatial_representation=spatial_rep, primitive_names=left_wing_top_names)
wing_left_bottom = LiftingSurface(name='wing_left_bottom', spatial_representation=spatial_rep, primitive_names=left_wing_bottom_names)
wing_left_te = LiftingSurface(name='wing_te', spatial_representation=spatial_rep, primitive_names=left_wing_te_top_names+left_wing_te_bottom_names)

wing_oml = LiftingSurface(name='wing_oml', spatial_representation=spatial_rep, primitive_names=left_wing_names+right_wing_names)
wing_top = LiftingSurface(name='wing_top', spatial_representation=spatial_rep, primitive_names=left_wing_top_names+right_wing_top_names)
wing_bottom = LiftingSurface(name='wing_bottom', spatial_representation=spatial_rep, primitive_names=left_wing_bottom_names+right_wing_bottom_names)
wing_te = LiftingSurface(name='wing_te', spatial_representation=spatial_rep, primitive_names=left_wing_te_top_names+left_wing_te_bottom_names+right_wing_te_top_names+right_wing_te_bottom_names)

# endregion




# structural_left_wing_names = left_wing_names.copy()
structural_left_wing_names = []


# region Internal structure generation

# projections for internal structure
num_pts = 10
spar_rib_spacing_ratio = 3
num_rib_pts = 20

tip_te = l_tip_te
tip_le = l_tip_le

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
# endregion


# region VLM mesh

# full wing
num_wing_vlm = 41
num_chordwise_vlm = 2 # increase if not using ML
point10 = r_tip_le
point11 = r_tip_te
point20 = tip_le
point21 = tip_te

le_offset = [-10,0,0]
te_offset = [10,0,0]

leading_edge_points = np.linspace(point10, point20, num_wing_vlm)
trailing_edge_points = np.vstack((np.linspace(r_tip_te, root_te, int(num_wing_vlm/2)+1),np.linspace(root_te, l_tip_te, int(num_wing_vlm/2)+1)[1:,:]))

leading_edge = wing_oml.project(leading_edge_points + le_offset, direction=np.array([0., 0., -1]), plot=do_plots, force_reprojection=force_reprojection)
trailing_edge = wing_te.project(trailing_edge_points, direction=np.array([1., 0., 0.]), plot=do_plots, force_reprojection=force_reprojection)

# Chord Surface
wing_chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
if False:
    spatial_rep.plot_meshes([wing_chord_surface])

# Upper and lower surface
wing_upper_surface_wireframe = wing_oml.project(wing_chord_surface.value + np.array([0., 0., 0.5*ft2m]),
                                            direction=np.array([0., 0., -1.]), grid_search_n=25,
                                            plot=plots_flag, max_iterations=200, force_reprojection=force_reprojection)
wing_lower_surface_wireframe = wing_oml.project(wing_chord_surface.value - np.array([0., 0., 0.5*ft2m]),
                                            direction=np.array([0., 0., 1.]), grid_search_n=25,
                                            plot=plots_flag, max_iterations=200, force_reprojection=force_reprojection)

# Chamber surface
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)
wing_vlm_mesh_name = 'wing_vlm_mesh'
sys_rep.add_output(wing_vlm_mesh_name, wing_camber_surface)

# OML mesh
wing_oml_mesh = am.vstack((wing_upper_surface_wireframe, wing_lower_surface_wireframe))
wing_oml_mesh_name = 'wing_oml_mesh'
sys_rep.add_output(wing_oml_mesh_name, wing_oml_mesh)

sys_rep.add_output(name='wing_chord_distribution',
                                    quantity=am.norm(leading_edge-trailing_edge))

# better OML mesh
surfaces = left_wing_names + right_wing_names
oml_para_mesh = []
grid_num = 10
for name in surfaces:
    for u in np.linspace(0,1,grid_num):
        for v in np.linspace(0,1,grid_num):
            oml_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

oml_geo_mesh = spatial_rep.evaluate_parametric(oml_para_mesh)
# endregion

# region ML Mesh
do_ML = True
if do_ML:
    wing_trailing_edge = trailing_edge
    wing_leading_edge = leading_edge
    num_spanwise_vlm = num_wing_vlm
    num_spanwise_ml = num_spanwise_vlm - 1
    wing_surface_offset_ml = [10,0,0]
    wing_surface_offset_ml_2 = [10,0,0]
    grid_search_n = 100
    # OML mesh for ML pressures wing
    te_points = np.vstack((np.linspace(r_tip_te, root_te, int(num_spanwise_vlm/2)+1),np.linspace(root_te, l_tip_te, int(num_spanwise_vlm/2)+1)[1:,:]))
    wing_trailing_edge_ml_2 = wing_te.project(te_points, direction=np.array([1, 0., 0.]), plot=do_plots, force_reprojection=force_reprojection)  
    wing_leading_edge_ml_2 = wing_oml.project(wing_trailing_edge_ml_2.evaluate() - wing_surface_offset_ml_2, direction=np.array([0., 0., -1.]), grid_search_n=25, plot=do_plots, force_reprojection=force_reprojection)
    wing_chord_surface_ml_2 = am.linspace(wing_leading_edge_ml_2, wing_trailing_edge_ml_2, num_chordwise_vlm)

    # print(wing_trailing_edge.value)
    wing_trailing_edge_array = wing_trailing_edge.value
    wing_trailing_edge_array_ml = np.zeros((num_spanwise_ml, 3))
    for i in range(num_spanwise_vlm-1):
        x = wing_trailing_edge_array[i, 0] + (wing_trailing_edge_array[i+1, 0] - wing_trailing_edge_array[i, 0])/2
        y = wing_trailing_edge_array[i, 1] + (wing_trailing_edge_array[i+1, 1] - wing_trailing_edge_array[i, 1])/2
        z = wing_trailing_edge_array[i, 2] + (wing_trailing_edge_array[i+1, 2] - wing_trailing_edge_array[i, 2])/2
        wing_trailing_edge_array_ml[i, 0] = x
        wing_trailing_edge_array_ml[i, 1] = y
        wing_trailing_edge_array_ml[i, 2] = z

    # print(wing_trailing_edge_array_ml)

    wing_trailing_edge_ml = wing_te.project(wing_trailing_edge_array_ml, direction=np.array([1., 0., 0.]), plot=do_plots, force_reprojection=force_reprojection)
    wing_leading_edge_ml = wing_oml.project(wing_trailing_edge_ml.evaluate() - wing_surface_offset_ml, direction=np.array([0., 0., -1.]), grid_search_n=25, plot=do_plots, force_reprojection=force_reprojection)
    wing_chord_surface_ml = am.linspace(wing_leading_edge_ml, wing_trailing_edge_ml, num_chordwise_vlm)


    num_ml_points = 100
    chord_surface_ml = am.linspace(wing_leading_edge_ml, wing_trailing_edge_ml, num_ml_points)
    i_vec = np.arange(0, len(chord_surface_ml.value))
    x_range = np.linspace(0, 1, num_ml_points)

    x_interp_x = wing_chord_surface_ml.value[1,:, 0].reshape(num_spanwise_ml, 1) - ((wing_chord_surface_ml.value[1, :, 0] - wing_chord_surface_ml.value[0, :, 0]).reshape(num_spanwise_ml, 1) * np.cos(np.pi/(2 * len(x_range)) * i_vec).reshape(1,100))
    x_interp_y = wing_chord_surface_ml.value[1,:, 1].reshape(num_spanwise_ml, 1) - ((wing_chord_surface_ml.value[1, :, 1] - wing_chord_surface_ml.value[0, :, 1]).reshape(num_spanwise_ml, 1) * np.cos(np.pi/(2 * len(x_range)) * i_vec).reshape(1,100))
    x_interp_z = wing_chord_surface_ml.value[1,:, 2].reshape(num_spanwise_ml, 1) - ((wing_chord_surface_ml.value[1, :, 2] - wing_chord_surface_ml.value[0, :, 2]).reshape(num_spanwise_ml, 1) * np.cos(np.pi/(2 * len(x_range)) * i_vec).reshape(1,100))

    x_interp_x_2 = wing_chord_surface_ml_2.value[1,:, 0].reshape(num_spanwise_vlm, 1) - ((wing_chord_surface_ml_2.value[1, :, 0] - wing_chord_surface_ml_2.value[0, :, 0]).reshape(num_spanwise_vlm, 1) * np.cos(np.pi/(2 * len(x_range)) * i_vec).reshape(1,100))
    x_interp_y_2 = wing_chord_surface_ml_2.value[1,:, 1].reshape(num_spanwise_vlm, 1) - ((wing_chord_surface_ml_2.value[1, :, 1] - wing_chord_surface_ml_2.value[0, :, 1]).reshape(num_spanwise_vlm, 1) * np.cos(np.pi/(2 * len(x_range)) * i_vec).reshape(1,100))
    x_interp_z_2 = wing_chord_surface_ml_2.value[1,:, 2].reshape(num_spanwise_vlm, 1) - ((wing_chord_surface_ml_2.value[1, :, 2] - wing_chord_surface_ml_2.value[0, :, 2]).reshape(num_spanwise_vlm, 1) * np.cos(np.pi/(2 * len(x_range)) * i_vec).reshape(1,100))

    new_chord_surface = np.zeros((num_ml_points, num_spanwise_ml, 3))
    new_chord_surface[:, :, 0] = x_interp_x.T
    new_chord_surface[:, :, 1] = x_interp_y.T
    new_chord_surface[:, :, 2] = x_interp_z.T

    new_chord_surface_2 = np.zeros((num_ml_points, num_spanwise_vlm, 3))
    new_chord_surface_2[:, :, 0] = x_interp_x_2.T
    new_chord_surface_2[:, :, 1] = x_interp_y_2.T
    new_chord_surface_2[:, :, 2] = x_interp_z_2.T

    wing_upper_surface_ml_dict = wing_top.project(new_chord_surface, direction=np.array([0., 0., 1.]), grid_search_n=grid_search_n, plot=do_plots, properties=['geometry', 'parametric_coordinates'], max_iterations=100, force_reprojection=force_reprojection)
    wing_lower_surface_ml_dict = wing_bottom.project(new_chord_surface, direction=np.array([0., 0., -1.]), grid_search_n=grid_search_n, plot=do_plots, properties=['geometry', 'parametric_coordinates'], max_iterations=100, force_reprojection=force_reprojection)

    wing_upper_surface_ml = wing_upper_surface_ml_dict['geometry']
    wing_lower_surface_ml = wing_lower_surface_ml_dict['geometry']

    wing_upper_surface_ml_2 = wing_top.project(new_chord_surface_2, direction=np.array([0., 0., 1.]), grid_search_n=grid_search_n, plot=do_plots, max_iterations=100, force_reprojection=force_reprojection)
    wing_lower_surface_ml_2 = wing_bottom.project(new_chord_surface_2, direction=np.array([0., 0., -1.]), grid_search_n=grid_search_n, plot=do_plots, max_iterations=100, force_reprojection=force_reprojection)

    upper_surface_pts = wing_upper_surface_ml_2.value
    lower_surface_pts = wing_lower_surface_ml_2.value

    avg_pts_upper = np.zeros((num_ml_points+1, num_spanwise_vlm, 3))
    avg_pts_lower = np.zeros((num_ml_points+1, num_spanwise_vlm, 3))
    avg_pts_upper[0,:,:] = upper_surface_pts[0,:,:]
    avg_pts_lower[0,:,:] = lower_surface_pts[0,:,:]
    avg_pts_upper[-1,:,:] = upper_surface_pts[-1,:,:]
    avg_pts_lower[-1,:,:] = lower_surface_pts[-1,:,:]


    for j in range(num_ml_points-1):
        avg_pts_upper[j+1,:,:] = (upper_surface_pts[j,:,:] + upper_surface_pts[j+1,:,:])/2
        avg_pts_lower[j+1,:,:] = (lower_surface_pts[j,:,:] + lower_surface_pts[j+1,:,:])/2

    wing_upper_corner_pts = wing_oml.project(avg_pts_upper, grid_search_n=grid_search_n, plot=do_plots, max_iterations=100, force_reprojection=force_reprojection)
    wing_lower_corner_pts = wing_oml.project(avg_pts_lower, grid_search_n=grid_search_n, plot=do_plots, max_iterations=100, force_reprojection=force_reprojection) 
    


    # print(wing_lower_surface_ml_2.value.shape)
    # exit()
    # exit()
    # wing_upper_surface_np_array = wing_upper_surface_ml_2.value
    # for i in range(num_spanwise_vlm-1):
    #     for j in range(100):
    #         if j==0:
    #             dy = np.linalg.norm(wing_upper_surface_np_array[j, i+1, :] - wing_upper_surface_np_array[j, i, :])
    #             dx = np.linalg.norm(wing_upper_surface_np_array[j, i, :] +  (wing_upper_surface_np_array[j, i, :] + wing_upper_surface_np_array[j, i, :])/2)
    #             area = dy * dx

    # exit()
    wing_oml_mesh_name_ml = 'wing_oml_mesh_ML'
    wing_oml_mesh_ml = am.vstack((wing_upper_surface_ml, wing_lower_surface_ml))
    # exit()
# endregion


#############################################
# filename = "./pav_wing/pav_wing_v2_caddee_mesh_SI_6307_quad.xdmf"
filename = "./pav_wing/pav_wing_v2_caddee_mesh_SI_2303_quad.xdmf"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    fenics_mesh = xdmf.read_mesh(name="Grid")
nel = fenics_mesh.topology.index_map(fenics_mesh.topology.dim).size_local
nn = fenics_mesh.topology.index_map(0).size_local

nodes = fenics_mesh.geometry.x

# region Functions

# geometry funciton
coefficients = {}

geo_space = lg.BSplineSpace(name='geo_base_space',
                            order=(4,4),
                            control_points_shape=(25,25))
wing_oml_geo = index_functions(left_wing_names+right_wing_names, 'wing_oml_geo', geo_space, 3)

for name in left_wing_names+right_wing_names:
    primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
    coefficients[name] = m3l.Variable(name = name + '_geo_coefficients', shape = primitive.control_points.shape, value = primitive.control_points)

wing_oml_geo.coefficients = coefficients


# wing thickness function
order = 2
shape = 3
space_t = lg.BSplineSpace(name='thickness_base_space',
                        order=(order, order),
                        control_points_shape=(shape, shape))
wing_thickness = index_functions(structural_left_wing_names, 'wing_thickness', space_t, 1)

# wing displacement function
order = 3
shape = 5
space_u = lg.BSplineSpace(name='displacement_base_space',
                        order=(order, order),
                        control_points_shape=(shape, shape))
wing_displacement = index_functions(structural_left_wing_names, 'wing_displacement', space_u, 3)

# wing force function
num = 25
u, v = np.meshgrid(np.linspace(0,1,num), np.linspace(0,1,num))
u = np.array(u).flatten()
v = np.array(v).flatten()
points = np.vstack((u,v)).T
space_f = IDWFunctionSpace(name='force_base_space', points=points, order=1, coefficients_shape=(points.shape[0],))
wing_force = index_functions(left_wing_names+right_wing_names, 'wing_force', space_f, 3)

# wing cp function
knots = np.linspace(0,1,shape+order)
space_cp = lg.BSplineSpace(name='cp_base_space',
                        order=(2, 4),
                        control_points_shape=(2, 10))
wing_cp = index_functions(left_wing_names+right_wing_names, 'wing_cp', space_cp, 1)

# endregion
with open(cfile + '/pav_wing/pav_wing_v2_paneled_mesh_data_'+str(nodes.shape[0])+'.pickle', 'rb') as f:
    nodes_parametric = pickle.load(f)

for i in range(len(nodes_parametric)):
    nodes_parametric[i] = (nodes_parametric[i][0].replace(' ', '_').replace(',',''), np.array([nodes_parametric[i][1]]))

thickness_nodes = wing_thickness.evaluate(nodes_parametric)

shell_pde = ShellPDE(fenics_mesh)


# # Aluminum 7050
# nu = 0.327
# E = 6.9E10 # unit: Pa (N/m^2)
# h = 3E-3 # overall thickness (unit: m)
# rho = 2700
# f_d = -rho*h*9.81


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
cruise_condition.set_module_input(name='pitch_angle', val=np.deg2rad(6))
cruise_condition.set_module_input(name='flight_path_angle', val=0)
cruise_condition.set_module_input(name='roll_angle', val=0)
cruise_condition.set_module_input(name='yaw_angle', val=0)
cruise_condition.set_module_input(name='wind_angle', val=0)
cruise_condition.set_module_input(name='observer_location', val=np.array([0, 0, 600*ft2m]))

cruise_ac_states = cruise_condition.evaluate_ac_states()
cruise_model.register_output(cruise_ac_states)
# endregion


# region VLM

wing_vlm_model = VASTFluidSover(
    surface_names=[
        wing_vlm_mesh_name,
    ],
    surface_shapes=[
        (1, ) + wing_camber_surface.evaluate().shape[1:],
        ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake'),
    mesh_unit='m',
    # cl0=[0.0, 0.0] # need to tune the coefficient
    cl0=[0.3475, 0.0], # need to tune the coefficient
    ML=True
)
cl_distribution, re_spans, wing_vlm_panel_forces, panel_areas, evaluation_pt, wing_vlm_forces, wing_vlm_moments  = wing_vlm_model.evaluate(ac_states=cruise_ac_states)
cruise_model.register_output(wing_vlm_forces)
cruise_model.register_output(wing_vlm_moments)
# endregion


# region ML
if do_ML:
    ml_pressures = PressureProfile(
        airfoil_name='NASA_langley_ga_1',
        use_inverse_cl_map=True,
    )

    cp_upper, cp_lower, Cd = ml_pressures.evaluate(cl_distribution, re_spans) #, mach_number, reynolds_number)
    cruise_model.register_output(cp_upper)
    cruise_model.register_output(cp_lower)

    viscous_drag_correction = ViscousCorrectionModel(
        surface_names=[
            f'{wing_vlm_mesh_name}_cruise',
        ],
        surface_shapes=[
            (1, ) + wing_camber_surface.evaluate().shape[1:],
        ],
    )
    moment_point = None
    vlm_F, vlm_M = viscous_drag_correction.evaluate(ac_states=cruise_ac_states, forces=wing_vlm_panel_forces, cd_v=Cd, panel_area=panel_areas, moment_pt=moment_point, evaluation_pt=evaluation_pt, design_condition=cruise_condition)
    cruise_model.register_output(vlm_F)
    cruise_model.register_output(vlm_M)

    ml_pressures_oml_map = NodalPressureProfile(
        surface_names=[
            f'{wing_vlm_mesh_name}_cruise',
        ],
        surface_shapes=[
            wing_upper_surface_ml.value.shape,
        ]
    )

    cp_upper_oml, cp_lower_oml = ml_pressures_oml_map.evaluate(cp_upper, cp_lower, nodal_pressure_mesh=[])
    wing_oml_pressure_upper = cp_upper_oml[0]
    wing_oml_pressure_lower = cp_lower_oml[0]

    upper_normals_ml = wing_oml_geo.evaluate_normals(wing_upper_surface_ml_dict['parametric_coordinates'])
    lower_normals_ml = wing_oml_geo.evaluate_normals(wing_lower_surface_ml_dict['parametric_coordinates'])

    # print(wing_upper_surface_ml.shape)
    # print(wing_upper_surface_ml_2.shape)
    # exit()


    ml_nodal_force_map = NodalForces()
    ml_f_upper, ml_f_lower = ml_nodal_force_map.evaluate(vlm_F = vlm_F,
                                                         oml_pressures_upper=wing_oml_pressure_upper, 
                                                         oml_pressures_lower=wing_oml_pressure_lower, 
                                                         normals_upper=upper_normals_ml, 
                                                         normals_lower=lower_normals_ml, 
                                                         upper_ml_mesh=wing_upper_surface_ml,
                                                         lower_ml_mesh=wing_lower_surface_ml,
                                                         upper_ml_vlm_mesh=wing_upper_surface_ml_2,
                                                         lower_ml_vlm_mesh=wing_lower_surface_ml_2)

    cruise_model.register_output(ml_f_upper)
    cruise_model.register_output(ml_f_lower)

    # ml_parametric_nodes = wing_upper_surface_ml_dict['parametric_coordinates']+wing_lower_surface_ml_dict['parametric_coordinates']

    vstack = m3l.VStack()
    wing_oml_force = vstack.evaluate(ml_f_upper, ml_f_lower)
    
    vstack = m3l.VStack()
    wing_oml_pressure = vstack.evaluate(wing_oml_pressure_upper, wing_oml_pressure_lower)

    cruise_model.register_output(wing_oml_pressure_upper)
    cruise_model.register_output(wing_oml_pressure_lower)

    ml_nodes = wing_oml_mesh_ml.value.reshape((num_ml_points*2*num_spanwise_ml, 3), order='F')
    ml_nodes_correct = np.zeros(ml_nodes.shape)
    for i in range(num_spanwise_ml):
        ml_nodes_correct[i*100:i*100+100] = ml_nodes[i*200:i*200+100]
        ml_nodes_correct[i*100+100*num_spanwise_ml:i*100+100*num_spanwise_ml+100] = ml_nodes[i*200+100:i*200+200]
    ml_nodes = ml_nodes_correct

    ml_nodes_parametric = wing_oml.project(ml_nodes, properties=['parametric_coordinates'], force_reprojection=force_reprojection)

    valid_surfaces_ml = []
    for item in ml_nodes_parametric:
        if not item[0] in valid_surfaces_ml:
            valid_surfaces_ml.append(item[0])

    wing_force.inverse_evaluate(ml_nodes_parametric,wing_oml_force)
    cruise_model.register_output(wing_force.coefficients)

    wing_cp.inverse_evaluate(ml_nodes_parametric, wing_oml_pressure)
    cruise_model.register_output(wing_cp.coefficients)
# endregion



# region Force transfer

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

oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=wing_vlm_panel_forces, nodal_force_meshes=[oml_geo_mesh, wing_oml_mesh])
wing_forces = oml_forces[0]

# wing_force.inverse_evaluate(oml_para_mesh, wing_forces)
# cruise_model.register_output(wing_force.coefficients)

left_oml_para_coords = []
num_points = 10
for name in valid_surfaces_ml:
    if name in left_wing_names:
        for u in np.linspace(0,1,num_points):
            for v in np.linspace(0,1,num_points):
                left_oml_para_coords.append((name, np.array([u,v]).reshape((1,2))))

left_oml_geo_nodes = spatial_rep.evaluate_parametric(left_oml_para_coords)

left_wing_forces = wing_force.evaluate(left_oml_para_coords)

shell_force_map_model = rmshell.RMShellForces(component=wing,
                                                mesh=shell_mesh,
                                                pde=shell_pde,
                                                shells=shells)
cruise_structural_wing_mesh_forces = shell_force_map_model.evaluate(
                        nodal_forces=left_wing_forces,
                        nodal_forces_mesh=left_oml_geo_nodes)


# endregion

# region Structures

shell_displacements_model = rmshell.RMShell(component=wing,
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

# region Nodal Displacements

grid_num = 10
transfer_para_mesh = []
for name in structural_left_wing_names:
    for u in np.linspace(0,1,grid_num):
        for v in np.linspace(0,1,grid_num):
            transfer_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

transfer_geo_nodes_ma = spatial_rep.evaluate_parametric(transfer_para_mesh)


shell_nodal_displacements_model = rmshell.RMShellNodalDisplacements(component=wing,
                                                                    mesh=shell_mesh,
                                                                    pde=shell_pde,
                                                                    shells=shells)
nodal_displacements = shell_nodal_displacements_model.evaluate(cruise_structural_wing_mesh_displacements, transfer_geo_nodes_ma)
wing_displacement.inverse_evaluate(transfer_para_mesh, nodal_displacements)
cruise_model.register_output(wing_displacement.coefficients)

# endregion

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
shape = (9, 1)
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
    # if "spar" in name:
    #     shape = spar_shape
    # # elif "Wing" in name:
    # #     shape = skin_shape
    # elif "panel" in name:
    #     shape = skin_shape
    # elif "rib" in name:
    #     shape = rib_shape

    h_init = caddee_csdl_model.create_input('wing_thickness_'+name, val=h_min)
    # h_init = caddee_csdl_model.create_input('wing_thickness_'+name, val=h_min+i*0.0001)
    caddee_csdl_model.add_design_variable('wing_thickness_'+name, # 0.02 in
                                          lower=0.005 * in2m,
                                          upper=0.1 * in2m,
                                          scaler=1000,
                                          )
    caddee_csdl_model.register_output('wing_thickness_surface_'+name, csdl.expand(h_init, shape))
    caddee_csdl_model.connect('wing_thickness_surface_'+name,
                                system_model_name+'wing_thickness_function_evaluation.'+\
                                name+'_wing_thickness_coefficients')
    # system_model.structural_sizing.cruise_3.cruise_3.wing_thickness_function_evaluation.t_panel_17_wing_thickness_coefficients
    i += 1



################################################################
#### Simple thicknesses: constant for spars, skins and ribs ####
################################################################
# h_spar = caddee_csdl_model.create_input('h_spar', val=h_min)
# caddee_csdl_model.add_design_variable('h_spar',
#                                       lower=0.01 * in2m,
#                                       upper=0.1 * in2m,
#                                       scaler=1000,
#                                       )
# h_skin = caddee_csdl_model.create_input('h_skin', val=h_min)
# caddee_csdl_model.add_design_variable('h_skin',
#                                       lower=0.01 * in2m,
#                                       upper=0.1* in2m,
#                                       scaler=1000,
#                                       )
# h_rib = caddee_csdl_model.create_input('h_rib', val=h_min)
# caddee_csdl_model.add_design_variable('h_rib',
#                                       lower=0.01 * in2m,
#                                       upper=0.1 * in2m,
#                                       scaler=1000,
#                                       )
# for name in valid_structural_left_wing_names:
#     primitive = spatial_rep.get_primitives([name])[name].geometry_primitive
#     name = name.replace(' ', '_').replace(',','')
#     surface_id = i
#     if "spar" in name:
#         shape = spar_shape
#         h_init = h_spar
#     elif "Wing" in name:
#         shape = skin_shape
#         h_init = h_skin
#     elif "rib" in name:
#         shape = rib_shape
#         h_init = h_rib
#
#     # h_init = caddee_csdl_model.create_input('wing_thickness_'+name, val=h_min)
#     caddee_csdl_model.register_output('wing_thickness_surface_'+name, csdl.expand(h_init, shape))
#     caddee_csdl_model.connect('wing_thickness_surface_'+name,
#                                 system_model_name+'wing_thickness_evaluation.'+\
#                                 name+'_t_coefficients')
#     i += 1
#################### end of m3l ########################




sim = Simulator(caddee_csdl_model, analytics=True)
sim.run()

'''

print('!!!!!!!!!!!!!!!!!!!!!')
print(sim['system_model.structural_sizing.cruise_3.cruise_3.ml_nodal_force_evaluation.ml_f_upper'])
print(np.linalg.norm(sim['system_model.structural_sizing.cruise_3.cruise_3.wing_oml_geo_function_normal_evaluation.evaluated_normal_wing_oml_geo_function'], axis=1))


vlm_forces = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_mesh_vlm_nodal_forces_model.wing_vlm_mesh_oml_forces']
ml_u_forces = sim['system_model.structural_sizing.cruise_3.cruise_3.ml_nodal_force_evaluation.ml_f_upper']
ml_l_forces = sim['system_model.structural_sizing.cruise_3.cruise_3.ml_nodal_force_evaluation.ml_f_lower']
normals = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_oml_geo_function_normal_evaluation.evaluated_normal_wing_oml_geo_function']


ml_totals = np.sum(ml_u_forces, axis=(0,1)) + np.sum(ml_l_forces, axis=(0,1))
vlm_F_total = sim['system_model.structural_sizing.cruise_3.cruise_3.cruise_3_wing_vlm_mesh_cruise_viscous_correction_model.viscous_correction_ml.F']


vector = normals.reshape((100*40,3))
u = vector[:,0]
v = vector[:,1]
w = vector[:,2]

locations = wing_upper_surface_ml.value.reshape((100*40,3))
x = locations[:,0]
y = locations[:,1]
z = locations[:,2]

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x,y,z,u,v,w, length=0.1)
ax.set_aspect('equal')
plt.show()


vector = ml_l_forces.reshape((100*40,3), order='F')
u = vector[:,0]
v = vector[:,1]
w = vector[:,2]


locations = wing_lower_surface_ml.value.reshape((100*40,3))
x = locations[:,0]
y = locations[:,1]
z = locations[:,2]

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x,y,z,u,v,w, length=.1)
ax.set_aspect('equal')
plt.show()



vector = ml_u_forces.reshape((100*40,3), order='F')
u = vector[:,0]
v = vector[:,1]
w = vector[:,2]


locations = wing_upper_surface_ml.value.reshape((100*40,3))
x = locations[:,0]
y = locations[:,1]
z = locations[:,2]

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x,y,z,u,v,w, length=.1)
ax.set_aspect('equal')
plt.show()


# vector = np.zeros((100,3))
# for i in range(100):
#     vector[i,:] = ml_u_forces.reshape((40*100,3))[i*40,:]

vector = ml_u_forces[0,:,:]
u = vector[:,0]
v = vector[:,1]
w = vector[:,2]


locations = wing_upper_surface_ml.value[:,0,:]
x = locations[:,0]
y = locations[:,1]
z = locations[:,2]

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x,y,z,u,v,w, length=.05)


vector = ml_l_forces[0,:,:]
u = vector[:,0]
v = vector[:,1]
w = vector[:,2]


locations = wing_lower_surface_ml.value[:,0,:]
x = locations[:,0]
y = locations[:,1]
z = locations[:,2]
ax.quiver(x,y,z,u,v,w, length=.05)



ax.set_aspect('equal')
plt.show()
'''


# # plotting
# import matplotlib.pyplot as plt
# def plot(x,y,z,v):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     cmap = plt.get_cmap("coolwarm")
#     cax = ax.scatter(x, y, z, s=50, c=v, cmap=cmap)
#     ax.set_aspect('equal')
#     plt.colorbar(cax)
#     plt.show()


# # region Displacement plotting
# dir = 0

# ## original points and values

# locations = transfer_geo_nodes_ma.value
# # locations = locations.reshape(locations.shape[0]*locations.shape[1], locations.shape[2])

# x = locations[:,0]
# y = locations[:,1]
# z = locations[:,2]
# v = sim['system_model.structural_sizing.cruise_3.cruise_3.Wing_rm_shell_displacement_map.wing_shell_nodal_displacement']
# v = v[:,dir]
# plot(x,y,z,v)

# ## original points, function values

# coefficients = {}
# for name in structural_left_wing_names:
#     c_name = 'system_model.structural_sizing.cruise_3.cruise_3.wing_displacement_function_inverse_evaluation.'+name+'_wing_displacement_coefficients'
#     coefficients[name] = sim[c_name]
# displacements = wing_displacement.compute(transfer_para_mesh, coefficients)
# v = displacements[:,dir]
# plot(x,y,z,v)

# ## new points, function values

# grid_num = 20
# new_para_mesh = []
# for name in structural_left_wing_names:
#     for u in np.linspace(0,1,grid_num):
#         for v in np.linspace(0,1,grid_num):
#             new_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

# locations = spatial_rep.evaluate_parametric(new_para_mesh).value
# x = locations[:,0]
# y = locations[:,1]
# z = locations[:,2]
# displacements = wing_displacement.compute(new_para_mesh, coefficients)
# v = displacements[:,dir]
# plot(x,y,z,v)

# # endregion


# # region Force plotting

# # original points and values
# dir = 2

# locations = oml_geo_mesh.value
# # locations = locations.reshape(locations.shape[0]*locations.shape[1], locations.shape[2])

# x = locations[:,0]
# y = locations[:,1]
# z = locations[:,2]
# v = sim['system_model.structural_sizing.cruise_3.cruise_3.wing_vlm_mesh_vlm_nodal_forces_model.wing_vlm_mesh_oml_forces']
# # v = v.reshape(v.shape[0]*v.shape[1], v.shape[2])

# v = v[:,dir]
# print(np.sum(v))

# plot(x,y,z,v)

# # original points, function values

# coefficients = {}
# for name in left_wing_names:
#     c_name = 'system_model.structural_sizing.cruise_3.cruise_3.wing_force_function_inverse_evaluation.'+name+'_wing_force_coefficients'
#     coefficients[name] = sim[c_name]
# forces = wing_force.compute(left_oml_para_coords, coefficients)
# v = forces[:,dir]
# print(np.sum(v))
# plot(x,y,z,v)

# # new points, function values

# grid_num = 20
# new_para_mesh = []
# # for name in left_wing_names:
# #     for u in np.linspace(0,1,grid_num):
# #         for v in np.linspace(0,1,grid_num):
# #             new_para_mesh.append((name, np.array([u,v]).reshape((1,2))))
# for name in left_wing_names:
#     for u in np.random.rand(grid_num):
#         for v in np.random.rand(grid_num):
#             new_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

# locations = spatial_rep.evaluate_parametric(new_para_mesh).value
# x = locations[:,0]
# y = locations[:,1]
# z = locations[:,2]
# forces = wing_force.compute(new_para_mesh, coefficients)
# v = forces[:,dir]
# print(np.sum(v))
# plot(x,y,z,v)


# # endregion

# region Pressure plotting

# original points, values

# name = 'system_model.structural_sizing.cruise_3.cruise_3.left_oml_cp_upper_stack_left_oml_cp_lower_operation.left_oml_cp_upper_stack_left_oml_cp_lower'
# wing_oml_pressure_val = sim[name].reshape((num_ml_points*2*num_spanwise_ml, 1))
# locations = ml_nodes
# v = wing_oml_pressure_val
# x = locations[:,0]
# y = locations[:,1]
# z = locations[:,2]
# plot(x,y,z,v)




# name = 'system_model.structural_sizing.cruise_3.cruise_3.wing_cp_function_inverse_evaluation.test_test_fn_values'
# wing_oml_pressure_val = sim[name]
# locations = ml_nodes
# v = wing_oml_pressure_val
# x = locations[:,0]
# y = locations[:,1]
# z = locations[:,2]
# plot(x,y,z,v)

# name = 'system_model.structural_sizing.cruise_3.cruise_3.left_oml_cp_upper_stack_left_oml_cp_lower_operation.left_oml_cp_upper_stack_left_oml_cp_lower'
# wing_oml_pressure_val = sim[name][:,0]
# print(wing_oml_pressure_val.shape)
# locations = wing_oml_mesh_ml.value[:,0,:]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.scatter(x, v)
# plt.show()



# name = 'system_model.structural_sizing.cruise_3.cruise_3.airfoil_ml_model.left_cp_upper'
# wing_oml_pressure_val = sim[name][0,:]
# locations = wing_oml_mesh_ml.value[0:100,0,:]
# x = locations[:,0]
# v = wing_oml_pressure_val
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.scatter(x, v)


# name = 'system_model.structural_sizing.cruise_3.cruise_3.airfoil_ml_model.left_cp_lower'
# wing_oml_pressure_val = sim[name][0,:]
# locations = wing_oml_mesh_ml.value[0:100,0,:]
# x = locations[:,0]
# v = wing_oml_pressure_val
# cax = ax.scatter(x, v)


# old points, fn vals
# surfaces = valid_surfaces_ml
# oml_cp_para_mesh = []
# grid_num = 10
# for name in surfaces:
#     for u in np.linspace(0,1,grid_num):
#         for v in np.linspace(0,1,grid_num):
#             oml_cp_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

# # oml_cp_geo_mesh = spatial_rep.evaluate_parametric(oml_cp_para_mesh)
# locations = spatial_rep.evaluate_parametric(ml_nodes_parametric)
# x = locations[:,0]
# y = locations[:,1]
# z = locations[:,2]

# coefficients = {}
# for name in valid_surfaces_ml:
#     c_name = 'system_model.structural_sizing.cruise_3.cruise_3.wing_cp_function_inverse_evaluation.'+name+'_wing_cp_coefficients'
#     coefficients[name] = sim[c_name]
# cp = wing_cp.compute(ml_nodes_parametric, coefficients)
# v = cp
# # plot(x,y,z,v)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# cmap = plt.get_cmap("coolwarm")
# cax = ax.scatter(x, y, z, s=50, c=v, cmap=cmap)
# ax.set_aspect('equal')
# plt.colorbar(cax)
# plt.show()


# # new points, function values
# surfaces = valid_surfaces_ml
# oml_cp_para_mesh = []
# grid_num = 50
# for name in surfaces:
#     for u in np.linspace(0,1,grid_num):
#         for v in np.linspace(0,1,grid_num):
#             oml_cp_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

# oml_cp_geo_mesh = spatial_rep.evaluate_parametric(oml_cp_para_mesh)
# locations = oml_cp_geo_mesh.value
# x = locations[:,0]
# y = locations[:,1]
# z = locations[:,2]

# coefficients = {}
# for name in valid_surfaces_ml:
#     c_name = 'system_model.structural_sizing.cruise_3.cruise_3.wing_cp_function_inverse_evaluation.'+name+'_wing_cp_coefficients'
#     coefficients[name] = sim[c_name]
# cp = wing_cp.compute(oml_cp_para_mesh, coefficients)
# v = cp
# # plot(x,y,z,v)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# cmap = plt.get_cmap("coolwarm")
# cax = ax.scatter(x, y, z, s=50, c=v, cmap=cmap)
# ax.set_aspect('equal')
# plt.colorbar(cax)
# plt.show()

# coeffs = coefficients['Wing_1_203'].reshape((25,25))
# x = np.linspace(0,1,25)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(25):
#     ax.plot(x, coeffs[i,:])
# plt.show()



# endregion




# sim.check_totals(of=[system_model_name+'Wing_rm_shell_model.rm_shell.aggregated_stress_model.wing_shell_aggregated_stress'],
#                                     wrt=['h_spar', 'h_skin', 'h_rib'])

# sim.check_totals(of=[system_model_name+'Wing_rm_shell_model.rm_shell.mass_model.mass'],
#                                     wrt=['h_spar', 'h_skin', 'h_rib'])
########################## Run optimization ##################################
prob = CSDLProblem(problem_name='pav', simulator=sim)

optimizer = SLSQP(prob, maxiter=50, ftol=1E-5)

# from modopt.snopt_library import SNOPT
# optimizer = SNOPT(prob,
#                   Major_iterations = 100,
#                   Major_optimality = 1e-5,
#                   append2file=False)

optimizer.solve()
optimizer.print_results()


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
