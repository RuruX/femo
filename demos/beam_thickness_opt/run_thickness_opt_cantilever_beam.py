# static 1D Cantilever Beam with a rectangular cross section
# this example uses Euler-Bernoulli ("classical") beam theory
from fe_csdl_opt.fea.fea_dolfinx import *
from fe_csdl_opt.csdl_opt.fea_model import FEAModel
from fe_csdl_opt.csdl_opt.state_model import StateModel
from fe_csdl_opt.csdl_opt.output_model import OutputModel
from fe_csdl_opt.csdl_opt.pre_processor.general_filter_model \
                                    import GeneralFilterModel

from dolfinx.mesh import locate_entities_boundary,create_interval
import numpy as np
import csdl
from csdl import Model
from csdl_om import Simulator as om_simulator
from python_csdl_backend import Simulator as py_simulator
from matplotlib import pyplot as plt
import argparse
import basix

'''
1. Define the mesh
'''

parser = argparse.ArgumentParser()
parser.add_argument('--nel',dest='nel',default='50',
                    help='Number of elements')

args = parser.parse_args()
########## GEOMETRIC INPUT ####################
E = 1.
L = 1.
b = 0.1
h = 0.1
volume = 0.01
rho= 2.7e-3
g = 9.81
#NOTE: floats must be converted to dolfin constants on domain below

#################################################################
########### CONSTRUCT BEAM MESH #################################
#################################################################
nel = int(args.nel)
mesh = create_interval(MPI.COMM_WORLD, nel, [0, L])

#################################################################
##### ENTER MATERIAL PARAMETERS AND CONSTITUTIVE MODEL ##########
#################################################################
x = SpatialCoordinate(mesh)

width = Constant(mesh,b)
E = Constant(mesh,E)
rho = Constant(mesh,rho)
g = Constant(mesh,g)


'''
2. Set up the PDE problem
'''

'''
    2.1 Define variational forms for PDE residual and outputs
'''

#define Moment expression
def M(u, t, E, width):
    EI = (E*width*t**3)/12
    return EI*div(grad(u))

def q(t, rho, g, width):
    #distributed load value (due to weight)
    A = t*width
    q = -rho*A*g
    return q

def pdeRes(u, v, t, f, E, width):
    # res = inner(div(grad(v)),M(u,t,E,width))*dx - q(t,rho,g,width)*v*dx
    res = inner(div(grad(v)),M(u,t,E,width))*dx - f*v*dx
    return res

def volume(t, width, L):
    return t*width*L*dx

def compliance(u, f):
    return dot(f,u)*dx
###################################################

fea = FEA(mesh)
# Add input to the PDE problem:
input_name = 'thickness'
input_function_space = FunctionSpace(mesh, ('DG', 0))
input_function = Function(input_function_space)


# Create Hermite order 3 on a interval (for more informations see:
#    https://defelement.com/elements/examples/interval-Hermite-3.html )
beam_element = basix.ufl_wrapper.create_element(basix.ElementFamily.Hermite, basix.CellType.interval, 3)

# Add state to the PDE problem:
state_name = 'displacements'
state_function_space = FunctionSpace(mesh, beam_element)
state_function = Function(state_function_space)
print("Number of DOFs: %d" % state_function_space.dofmap.index_map.size_global)
print("Number of elements (intervals): %d" % nel)
print("Number of nodes: %d" % (nel+1))

v = TestFunction(state_function_space)

# Add the same point load at the endpoint as the OpenMDAO example
#     https://github.com/OpenMDAO/OpenMDAO/blob/304d45169b4b2a20e7d0e5441f81d9c072d7af09/openmdao/test_suite/test_examples/beam_optimization/beam_group.py#L29
f = Function(state_function_space)
f.vector[nel*2] = -1.

residual_form = pdeRes(state_function,
                        v,
                        input_function,
                        f, E, width)

# Add outputs to the PDE problem:
output_name_1 = 'compliance'
output_form_1 = compliance(state_function,f)
output_name_2 = 'volume'
output_form_2 = volume(input_function,width,L)



fea.add_input(input_name, input_function)
fea.add_state(name=state_name,
                function=state_function,
                residual_form=residual_form,
                arguments=[input_name])
fea.add_output(name=output_name_1,
                type='scalar',
                form=output_form_1,
                arguments=[input_name, state_name])
fea.add_output(name=output_name_2,
                type='scalar',
                form=output_form_2,
                arguments=[input_name])
'''
    2.2. Define the boundary conditions
'''

############ Strongly enforced boundary conditions #############
ubc = Function(state_function_space)
ubc.vector.set(0.0)
startpt = locate_entities_boundary(mesh,0,lambda x : np.isclose(x[0], 0))
locate_BC1 = locate_dofs_topological(state_function_space,0,startpt)
locate_BC_list = [locate_BC1[0], locate_BC1[1]]
fea.add_strong_bc(ubc, locate_BC_list)

'''
3. Set up the CSDL model
'''
fea.REPORT = False
fea_model = FEAModel(fea=[fea])

fea_model.create_input("{}".format('thickness'),
                            shape=nel,
                            val=h) # h=0.05

fea_model.add_design_variable('thickness', upper=10., lower=1e-2)
fea_model.add_objective('compliance')
fea_model.add_constraint('volume', equals=b*h*L)
sim = py_simulator(fea_model,analytics=False)

########### Test the forward solve ##############

sim.run()

'''
4. Set up the optimization problem
'''
############# Run the optimization with modOpt #############
from modopt.csdl_library import CSDLProblem

prob = CSDLProblem(
    problem_name='beam_thickness_opt',
    simulator=sim,
)

# Both SNOPT and SLSQP work for this problem, though SNOPT is faster
from modopt.snopt_library import SNOPT
from modopt.scipy_library import SLSQP

# optimizer = SNOPT(prob,
#                   Major_iterations = 1000,
#                   Major_optimality = 1e-9,)
optimizer = SLSQP(prob, maxiter=1000, ftol=1e-9)

# Solve your optimization problem
optimizer.solve()
print("="*40)
optimizer.print_results()

# NOTE: The solution uh contains both the rotation and the displacement solutions
#The rotation and displacment solutions can be separated as follows:
#TODO: there is likely a much easier way to separate these DOFs and do so in a

disp = np.empty(0)
rot = np.empty(0)
for i,x in enumerate(state_function.x.array):
    if i % 2 != 0:
        rot = np.append(rot,x)
    else:
        disp = np.append(disp,x)
print("Maximum magnitude displacement (cantilever FEM solution) is: %e" % np.min(disp))
print("Compliance value: ", sim['compliance'])

# Reference optimized thickness distribution from the OpenMDAO example
#   https://openmdao.org/newdocs/versions/latest/examples/beam_optimization_example.html#implementation-optimization-script
thick_ref = [0.14915754,  0.14764328,  0.14611321,  0.14456715,  0.14300421,  0.14142417,
             0.13982611,  0.13820976,  0.13657406,  0.13491866,  0.13324268,  0.13154528,
             0.12982575,  0.12808305,  0.12631658,  0.12452477,  0.12270701,  0.12086183,
             0.11898809,  0.11708424,  0.11514904,  0.11318072,  0.11117762,  0.10913764,
             0.10705891,  0.10493903,  0.10277539,  0.10056526,  0.09830546,  0.09599246,
             0.09362243,  0.09119084,  0.08869265,  0.08612198,  0.08347229,  0.08073573,
             0.07790323,  0.07496382,  0.07190453,  0.06870925,  0.0653583,   0.06182632,
             0.05808044,  0.05407658,  0.04975295,  0.0450185,   0.03972912,  0.03363155,
             0.02620192,  0.01610863]

from matplotlib import pyplot as plt
fig, ax = plt.subplots()
ax.plot(np.linspace(0.0,L,50), thick_ref, "b-o", label="OpenMDAO results")
ax.plot(np.linspace(0.0,L,nel), sim['thickness'], "r-o", label="FEniCS+CSDL results")
ax.set_xlabel("x")
ax.set_ylabel("optimized thickness distribution")
ax.legend(loc="best")
plt.show()
fig.savefig("beam_thickness_distribution.png", dpi=150)


with XDMFFile(MPI.COMM_WORLD, "solutions/"+state_name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(fea.mesh)
    xdmf.write_function(fea.states_dict[state_name]['function'])
with XDMFFile(MPI.COMM_WORLD, "solutions/"+input_name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(fea.mesh)
    xdmf.write_function(fea.inputs_dict[input_name]['function'])
