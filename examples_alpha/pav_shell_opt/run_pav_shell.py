"""
PAV wing shell optimization setup using the new FEMO--csdl_alpha interface
"""

import dolfinx
from mpi4py import MPI
import csdl_alpha as csdl
from femo.rm_shell.rm_shell_model import RMShellModel
from femo.fea.utils_dolfinx import createCustomMeasure
import numpy as np

pav_mesh_list = ["pav_wing_6rib_caddee_mesh_2374_quad.xdmf",]

filename = "./pav_wing/"+pav_mesh_list[0]
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

# Unstiffened Aluminum 2024 (T4)
# reference: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma2024t4
E = 73.1E9 # unit: Pa
nu = 0.33
in2m = 0.0254
h = 0.05*in2m
rho = 2780 # unit: kg/m^3
y_tip = -4.2672
y_root = -1E-6
g = 9.81 # unit: m/s^2
f_d = -g*rho*h # body force per unit surface area


#### Getting facets of the LEFT and the RIGHT edge  ####
DOLFIN_EPS = 3E-16
def ClampedBoundary(x):
    return np.greater(x[1], y_root+DOLFIN_EPS)
def TipChar(x):
    return np.less(x[1], y_tip+DOLFIN_EPS)
fdim = mesh.topology.dim - 1

ds_1 = createCustomMeasure(mesh, fdim, ClampedBoundary, measure='ds', tag=100)
dS_1 = createCustomMeasure(mesh, fdim, ClampedBoundary, measure='dS', tag=100)
dx_2 = createCustomMeasure(mesh, fdim+1, TipChar, measure='dx', tag=10)

###################  m3l ########################

# create the shell dictionaries:
shells = {'E': E, 'nu': nu, 'rho': rho,# material properties
            'dss': ds_1(100), # custom ds measure for the Dirichlet BC
            'dSS': dS_1(100), # custom dS measure for the Dirichlet BC
            'dxx': dx_2(10), # custom dx measure for the tip displacement   
            'record': True}  # custom integrator: dx measure}

recorder = csdl.Recorder(inline=True)
recorder.start()

force_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
force_vector.value[:, 2] = f_d
thicknesses = csdl.Variable(value=h*np.ones(nn), name='thicknesses')

shell_model = RMShellModel(mesh, shells)
shell_outputs = shell_model.evaluate(force_vector, thicknesses)

disp_solid = shell_outputs.disp_solid
compliance = shell_outputs.compliance
mass = shell_outputs.mass
wing_von_Mises_stress = shell_outputs.stress
wing_aggregated_stress = shell_outputs.aggregated_stress

recorder.stop()

########## Output: ##########
print("Wing tip deflection (m):",max(abs(disp_solid.value)))
print("Wing total mass (kg):", mass.value)
print("Wing aggregated von Mises stress (Pascal):", wing_aggregated_stress.value)
print("Wing maximum von Mises stress (Pascal):", max(wing_von_Mises_stress.value))
print("  Number of elements = "+str(nel))
print("  Number of vertices = "+str(nn))

########## Visualization: ##############
w = shell_model.fea.states_dict['disp_solid']['function']
u_mid, _ = w.split()
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "solutions/u_mid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)




