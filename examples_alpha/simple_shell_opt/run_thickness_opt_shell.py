"""
A simple thickness optimization problem using shell elements in FEMO
"""


"""
TODO: Convert this pure FEniCS code to FEMO code using ShellPDE, ShellModule
"""

from dolfinx.io import XDMFFile
from dolfinx.fem import (locate_dofs_topological, locate_dofs_geometrical,
                        dirichletbc, form, Constant, VectorFunctionSpace)
from dolfinx.mesh import locate_entities
import numpy as np
from mpi4py import MPI
from shell_analysis_fenicsx import *

beam = [#### quad mesh ####
        "plate_2_10_quad_4_20.xdmf",
        "plate_2_10_quad_8_40.xdmf",
        "plate_2_10_quad_10_50.xdmf",]

filename = "./plate_meshes/"+beam[2]
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

E_val = 4.32e8
nu_val = 0.0
h_val = 0.2
width = 2.
length = 10.
f_d = 10.*h_val

E = Constant(mesh,E_val) # Young's modulus
nu = Constant(mesh,nu_val) # Poisson ratio
h = Constant(mesh,h_val) # Shell thickness
f = ufl.as_vector([0,0,f_d]) # Body force per unit surface area

element_type = "CG2CG1"
#element_type = "CG2CR1"

element = ShellElement(
                mesh,
                element_type,
#                inplane_deg=3,
#                shear_deg=3
                )
W = element.W
w = Function(W)
dx_inplane, dx_shear = element.dx_inplane, element.dx_shear


#### Compute the CLT model from the material properties (for single-layer material)
material_model = MaterialModel(E=E,nu=nu,h=h)
elastic_model = ElasticModel(mesh,w,material_model.CLT)
elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane,dx_shear)
F = elastic_model.weakFormResidual(elastic_energy, f)

######### Set the BCs to have all the dofs equal to 0 on the left edge ##########
# Define BCs geometrically
locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),
                                    lambda x: np.isclose(x[0], 0. ,atol=1e-6))
ubc=  Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

bcs = [dirichletbc(ubc, locate_BC1, W.sub(0)),
        dirichletbc(ubc, locate_BC2, W.sub(1)),
       ]


########## Solve with Newton solver wrapper: ##########
solveNonlinear(F,w,bcs)

# Comparing the solution to the Kirchhoff analytical solution
uZ = computeNodalDisp(w.sub(0))[2]
Ix = width*h_val**3/12


########## Output: ##########
print("Euler-Beinoulli Beam theory deflection:",
    float(f_d*width*length**4/(8*E_val*Ix)))
print("Reissner-Mindlin FE deflection:", max(uZ))

print("  Number of elements = "+str(mesh.topology.index_map(mesh.topology.dim).size_local))
print("  Number of vertices = "+str(mesh.topology.index_map(0).size_local))

########## Visualization: ##############

u_mid, _ = w.split()
with XDMFFile(MPI.COMM_WORLD, "solutions/u_mid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_mid)

