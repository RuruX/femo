'''
Thickness optimization of Cantilever Plate with Reissner-Mindlin shell elements.
This example uses pre-built Reissner-Mindlin shell model in FEMO

Author: Ru Xiang
Date: 2024-06-20
'''

import dolfinx
from mpi4py import MPI
import csdl_alpha as csdl
import numpy as np

from femo.rm_shell.rm_shell_model import RMShellModel
from femo.fea.utils_dolfinx import createCustomMeasure

run_verify_forward_eval = True
run_check_derivatives = False
run_optimization = False

'''
1. Define the mesh
'''

plate = [#### quad mesh ####
        "plate_2_10_quad_4_20.xdmf",
        "plate_2_10_quad_8_40.xdmf",
        "plate_2_10_quad_10_50.xdmf",]

filename = "./plate_meshes/"+plate[0]
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

E_val = 4.32e8
nu_val = 0.0
h_val = 0.2
rho_val = 1.0
width = 2.
length = 10.
f_d = 10.*h_val

'''
2. Define the boundary conditions
'''
# clamped root boundary condition
DOLFIN_EPS = 3E-16
def ClampedBoundary(x):
    return np.less(x[0], 0.0+DOLFIN_EPS)

'''
3. Set up csdl recorder and run the simulation
'''
recorder = csdl.Recorder(inline=True)
recorder.start()

force_vector = csdl.Variable(value=np.zeros((nn, 3)), name='force_vector')
force_vector.value[:, 2] = f_d # body force per unit surface area

thickness = csdl.Variable(value=h_val*np.ones(nn), name='thickness')
E = csdl.Variable(value=E_val*np.ones(nn), name='E')
nu = csdl.Variable(value=nu_val*np.ones(nn), name='nu')
density = csdl.Variable(value=rho_val*np.ones(nn), name='density')

node_disp = csdl.Variable(value=np.zeros((nn, 3)), name='node_disp')
node_disp.add_name('node_disp')

# All FEA variables will be saved to xdmf files if record=True
shell_model = RMShellModel(mesh, shell_bc_func=ClampedBoundary, record=True)
shell_outputs = shell_model.evaluate(force_vector, thickness, E, nu, density,
                                        node_disp,
                                        debug_mode=False)

disp_solid = shell_outputs.disp_solid
compliance = shell_outputs.compliance
aggregated_stress = shell_outputs.aggregated_stress
mass = shell_outputs.mass

if run_verify_forward_eval:
    Ix = width*h_val**3/12
    print("Euler-Beinoulli Beam theory deflection:",
        float(f_d*width*length**4/(8*E_val*Ix)))
    print("Reissner-Mindlin FE deflection:", max(disp_solid.value))

if run_check_derivatives:
    from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
    verify_derivatives_inline([compliance, aggregated_stress, mass],[node_disp], 
                                step_size=1E-11, raise_on_error=False)

if run_optimization:
    from modopt import CSDLAlphaProblem
    from modopt import SLSQP
    thickness.set_as_design_variable(upper=10, lower=1E-2)

    mass_0 = rho_val*h_val*width*length
    mass.set_as_constraint(lower=mass_0, upper=mass_0)
    compliance.set_as_objective()
    sim = csdl.experimental.PySimulator(recorder)

    prob = CSDLAlphaProblem(problem_name='plate_thickness', simulator=sim)

    optimizer = SLSQP(prob, ftol=1e-9, maxiter=1000, outputs=['x'])

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()
    print("Optimization results:")
    print(" "*4, compliance.names, compliance.value)
    print(" "*4, mass.names, mass.value)

recorder.stop()

'''
4. post-processing
'''
w = shell_model.fea.states_dict['disp_solid']['function']
u_mid = w.sub(0).collapse().x.array
theta = w.sub(1).collapse().x.array

print("Tip deflection:", max(abs(u_mid)))
print("Compliance:", compliance.value)

print("  Number of elements = "+str(nel))
print("  Number of vertices = "+str(nn))






