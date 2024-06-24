"""
PAV wing shell optimization setup using the new FEMO--csdl_alpha interface.
This example uses pre-built Reissner-Mindlin shell model in FEMO

Author: Ru Xiang
Date: 2024-06-20
"""

import dolfinx
from mpi4py import MPI
import csdl_alpha as csdl
import numpy as np

from femo.rm_shell.rm_shell_model import RMShellModel

run_verify_forward_eval = True
run_check_derivatives = False


'''
1. Define the mesh
'''

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


'''
2. Define the boundary conditions
'''
#### Fix all displacements and rotations on the root surface  ####
DOLFIN_EPS = 3E-16
def ClampedBoundary(x):
    return np.greater(x[1], y_root+DOLFIN_EPS)

'''
3. Set up csdl recorder and run the simulation
'''
recorder = csdl.Recorder(inline=True)
recorder.start()

force_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
force_vector.value[:, 2] = f_d

# Constant material properties
thickness_0 = csdl.Variable(value=h_val, name='thickness_0')
E_0 = csdl.Variable(value=E_val, name='E_0')
nu_0 = csdl.Variable(value=nu_val, name='nu_0')
density_0 = csdl.Variable(value=density_val, name='density_0')

# Nodal material properties
thickness = csdl.expand(thickness_0, out_shape=(nn,))
thickness.add_name('thickness')
E = csdl.expand(E_0, out_shape=(nn,))
E.add_name('E')
nu = csdl.expand(nu_0, out_shape=(nn,))
nu.add_name('nu')
density = csdl.expand(density_0, out_shape=(nn,))   
density.add_name('density')
node_disp = csdl.Variable(value=0.1*np.ones((nn, 3)), name='node_disp')
node_disp.add_name('node_disp')
node_disp.value[:, 2] = 0.0 # z-displacement is zero for each



shell_model = RMShellModel(mesh, shell_bc_func=ClampedBoundary, record=True)
shell_outputs = shell_model.evaluate(force_vector, thickness, 
                                        E, nu, density, 
                                        node_disp,
                                        debug_mode=False)

disp_solid = shell_outputs.disp_solid
compliance = shell_outputs.compliance
mass = shell_outputs.mass
elastic_energy = shell_outputs.elastic_energy
disp_extracted = shell_outputs.disp_extracted
wing_von_Mises_stress = shell_outputs.stress
wing_aggregated_stress = shell_outputs.aggregated_stress

if run_verify_forward_eval:
    print("Wing tip deflection (m):",max(abs(disp_solid.value)))
    print("Extracted wing tip deflection (m):",max(abs(disp_extracted.value[:,2])))
    print("Wing total mass (kg):", mass.value)
    print("Wing Compliance (N*m):", compliance.value)
    print("Wing elastic energy (J):", elastic_energy.value)
    print("Wing aggregated von Mises stress (Pascal):", wing_aggregated_stress.value)
    print("Wing maximum von Mises stress (Pascal):", max(wing_von_Mises_stress.value))
    print("  Number of elements = "+str(nel))
    print("  Number of vertices = "+str(nn))


if run_check_derivatives:
    # Verify the derivatives
    # [RX] extra caution is needed for the step_size; 
    # rule-of-thumb: 1E-6/order_of_magnitude(analytical_derivative)

    from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
    verify_derivatives_inline([wing_aggregated_stress], 
                                [E_0], 
                                step_size=1E8, raise_on_error=False)

    verify_derivatives_inline([wing_aggregated_stress], 
                                [nu_0], 
                                step_size=1E-11, raise_on_error=False)

    verify_derivatives_inline([wing_aggregated_stress], 
                                [thickness_0], 
                                step_size=1E-11, raise_on_error=False)
recorder.stop()
