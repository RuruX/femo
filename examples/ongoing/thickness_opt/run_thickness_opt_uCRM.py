"""
Structural analysis for the undeflected common research model (uCRM)
uCRM-9 Specifications: (units: m/ft, kg/lb)
(from https://deepblue.lib.umich.edu/bitstream/handle/2027.42/143039/6.2017-4456.pdf?sequence=1)
Maximum take-off weight	352,400kg/777,000lb
Wing span (extended)    71.75m/235.42ft
Overall length	        76.73m/251.75ft
"""

from femo.fea.fea_dolfinx import *
from femo.csdl_opt.fea_model import FEAModel
from femo.csdl_opt.state_model import StateModel
from femo.csdl_opt.output_model import OutputModel
from femo.csdl_opt.pre_processor.general_filter_model \
                                    import GeneralFilterModel
import numpy as np
import csdl
from csdl import Model
from csdl_om import Simulator
from matplotlib import pyplot as plt
import argparse
from mpi4py import MPI
from shell_analysis_fenicsX import *

quad_mesh = ["uCRM-9_wingbox_quad_coarse.xdmf",
            "uCRM-9_wingbox_quad_medium.xdmf",
            "uCRM-9_wingbox_quad_fine.xdmf",]

file_name = quad_mesh[0]
path = "../shell_analysis_fenics/mesh/mesh-examples/uCRM-9/"
mesh_file = path + file_name
with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
       mesh = xdmf.read_mesh(name="Grid")
nel = mesh.topology.index_map(mesh.topology.dim).size_local
nn = mesh.topology.index_map(0).size_local

E = 7.31E10 # unit: (N/m^2)
nu = 0.3

############ Constant body force over the mesh ##########
# Scaled body force
#f_d = 2780*9.81*h_val # force per unit area (unit: N/m^2)
#f_array = np.tile([0,0,f_d], nn)
##f = Constant(mesh,(0,0,f_d)) # Body force per unit area

################### Aerodynamic loads ###################
f_array = np.loadtxt(path+'aero_force_coarse.txt')
############# Apply distributed loads ###################
VF = VectorFunctionSpace(mesh, ("CG", 1))
f = Function(VF)
f.vector.setArray(f_array) # Body force per unit area



element_type = "CG2CG1" # with quad/tri elements
#element_type = "CG2CR1" # with tri elements

element = ShellElement(
                mesh,
                element_type,
#                inplane_deg=3,
#                shear_deg=3
                )
dx_inplane, dx_shear = element.dx_inplane, element.dx_shear


def pdeRes(h,w,E,f,CLT,dx_inplane,dx_shear):
    elastic_model = ElasticModel(mesh,w,CLT)
    elastic_energy = elastic_model.elasticEnergy(E, h, dx_inplane,dx_shear)
    return elastic_model.weakFormResidual(elastic_energy, f)


def compliance(u_mid,h):
    h_mesh = CellDiameter(mesh)
    alpha = 1e-1
    dX = ufl.Measure('dx', domain=mesh, metadata={"quadrature_degree":0})
    return 0.5*dot(u_mid,u_mid)*dX \
            + 0.5*alpha*dot(grad(h), grad(h))*(h_mesh**2)*dX

def volume(h):
    return h*dx

#######################################################
############## The optimization problem ###############
#######################################################
fea = FEA(mesh)
# Add input to the PDE problem:
input_name = 'thickness'
input_function_space = FunctionSpace(mesh, ("DG", 0))
input_function = Function(input_function_space)

# Add state to the PDE problem:
state_name = 'displacements'
state_function_space = element.W
state_function = Function(state_function_space)
material_model = MaterialModel(E=E,nu=nu,h=input_function) # Simple isotropic material
residual_form = pdeRes(input_function,state_function,
                        E,f,material_model.CLT,dx_inplane,dx_shear)

# Add output to the PDE problem:
output_name_1 = 'compliance'
output_form_1 = compliance(state_function.sub(0), input_function)
output_name_2 = 'volume'
output_form_2 = volume(input_function)


fea.add_input(input_name, input_function)
fea.add_state(name=state_name,
                function=state_function,
                residual_form=residual_form,
                arguments=[input_name])
fea.add_output(name=output_name_1,
                type='scalar',
                form=output_form_1,
                arguments=[state_name,input_name])
fea.add_output(name=output_name_2,
                type='scalar',
                form=output_form_2,
                arguments=[input_name])

############ Set the BCs for the airplane model ###################

u0 = Function(state_function_space)
u0.vector.set(0.0)

# Define BCs geometrically
locate_BC1 = locate_dofs_geometrical((state_function_space.sub(0), state_function_space.sub(0).collapse()[0]),
                                    lambda x: np.less(x[1], 3.1))
locate_BC2 = locate_dofs_geometrical((state_function_space.sub(1), state_function_space.sub(1).collapse()[0]),
                                    lambda x: np.less(x[1], 3.1))
ubc=  Function(state_function_space)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

bcs = [dirichletbc(ubc, locate_BC1, state_function_space.sub(0)),
        dirichletbc(ubc, locate_BC2, state_function_space.sub(1)),
       ]

############ Strongly enforced boundary conditions #############
# fea.add_strong_bc(ubc, locate_BC_list, state_function_space)
for i in range(len(bcs)):
    fea.bc.append(bcs[i])


########## Solve with Newton solver wrapper: ##########
# solveNonlinear(F,state_function,bcs)

########## Output: ##############

uZ = computeNodalDisp(state_function.sub(0))[2]
print("-"*50)
print("-"*8, file_name, "-"*9)
print("-"*50)
print("Tip deflection:", max(uZ))
print("  Number of elements = "+str(nel))
print("  Number of vertices = "+str(nn))
print("  Number of total dofs = ", len(state_function.vector.getArray()))
print("-"*50)

########## Visualization: ##############
u_mid, _ = state_function.split()
with XDMFFile(MPI.COMM_WORLD, "solutions/u_mid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    u_mid.name = 'u_mid'
    xdmf.write_function(u_mid)
with XDMFFile(MPI.COMM_WORLD, "solutions/aero_loads.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    f.name = 'f'
    xdmf.write_function(f)
