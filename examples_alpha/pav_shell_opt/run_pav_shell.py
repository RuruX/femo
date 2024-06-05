"""
PAV wing shell optimization setup using the new FEMO--csdl_alpha interface
"""

import dolfinx
from mpi4py import MPI
import csdl_alpha as csdl
from femo.rm_shell.rm_shell_model import RMShellModel
import numpy as np
import lsdo_geo as lg



recorder = csdl.Recorder(inline=True)
recorder.start()

# pav_geometry = lg.import_geometry('pav_wing/pav.stp', parallelize=False)
# pav_geometry.plot(opacity=0.3)

pav_mesh_list = ["pav_wing_6rib_caddee_mesh_2374_quad.xdmf",]

filename = "./pav_wing/"+pav_mesh_list[0]
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

# Unstiffened Aluminum 2024 (T4)
# reference: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma2024t4
E_val = 73.1E9 # unit: Pa
nu_val = 0.33
in2m = 0.0254
h_val = 0.05*in2m
density_val = 2780 # unit: kg/m^3
y_tip = -4.2672
y_root = -1E-6
g = 9.81 # unit: m/s^2
f_d = -g*density_val*h_val # body force per unit surface area



# create the shell dictionaries for bc locations:
shell_bcs = {'y_root': y_root,
                'y_tip': y_tip} 


force_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
force_vector.value[:, 2] = f_d

# Constant material properties
thickness_0 = csdl.Variable(value=h_val, name='thickness_0')
E_0 = csdl.Variable(value=E_val, name='E_0')
nu_0 = csdl.Variable(value=nu_val, name='nu_0')
density_0 = csdl.Variable(value=density_val, name='density_0')

# Nodal material properties
thickness = thickness_0*np.ones(nn)
thickness.add_name('thickness')
E = E_0*np.ones(nn)
E.add_name('E')
nu = nu_0*np.ones(nn)
nu.add_name('nu')
density = density_0*np.ones(nn)
density.add_name('density')

shell_model = RMShellModel(mesh, shell_bcs=shell_bcs, record=True)
shell_outputs = shell_model.evaluate(force_vector, thickness, 
                                        E, nu, density, debug_mode=False)

disp_solid = shell_outputs.disp_solid
compliance = shell_outputs.compliance
mass = shell_outputs.mass
elastic_energy = shell_outputs.elastic_energy
disp_extracted = shell_outputs.disp_extracted
wing_von_Mises_stress = shell_outputs.stress
wing_aggregated_stress = shell_outputs.aggregated_stress

from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
verify_derivatives_inline([wing_aggregated_stress], 
                            [thickness_0], 
                            step_size=1e-6, raise_on_error=False)
recorder.stop()

########## Output: ##########
print("Wing tip deflection (m):",max(abs(disp_solid.value)))
print("Extracted wing tip deflection (m):",max(abs(disp_extracted.value[:,2])))
print("Wing total mass (kg):", mass.value)
print("Wing Compliance (N*m):", compliance.value)
print("Wing elastic energy (J):", elastic_energy.value)
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






# 2024-05-30 16:08:30.708 (   7.785s) [main            ]       NewtonSolver.cpp:273   WARN| Newton solver did not converge.
# Wing tip deflection (m): 0.000828640107096966
# Wing total mass (kg): [29.10897555]
# Wing aggregated von Mises stress (Pascal): [1929864.41525933]
# Wing maximum von Mises stress (Pascal): 2104232.0788074955
#   Number of elements = 2416
#   Number of vertices = 2374