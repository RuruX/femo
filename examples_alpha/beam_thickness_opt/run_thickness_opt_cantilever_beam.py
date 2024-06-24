'''
Thickness optimization of 1D Cantilever Beam with a rectangular cross section
this example uses Euler-Bernoulli beam theory

Author: Ru Xiang
Date: 2024-06-20
'''

from dolfinx.mesh import locate_entities_boundary,create_interval
import numpy as np
import csdl_alpha as csdl
import ufl
from matplotlib import pyplot as plt
import basix


from femo.fea.fea_dolfinx import *
from femo.csdl_alpha_opt.fea_model import FEAModel

run_verify_forward_eval = False
run_check_derivatives = True
run_optimization = False

'''
1. Define the mesh
'''

# Geometric inputs and material properties
E = 1.
L = 1.
b = 0.1
h = 0.1
volume = 0.01

# Construct beam mesh
nel = 50
mesh = create_interval(MPI.COMM_WORLD, nel, [0., L])

x = ufl.SpatialCoordinate(mesh)
# width = ufl.Constant(mesh,b)
# E = ufl.Constant(mesh,E)
width = b


'''
2. Set up the PDE problem
'''

'''
    2.1. Define variational forms for PDE residual and outputs
'''

# Define Moment expression
def M(u, t, E, width):
    t3 = t**3
    a = width*t**3
    EI = (E*width*t**3)/12
    return EI*div(grad(u))

def pdeRes(u, v, t, f, dss, E, width):
    res = inner(div(grad(v)),M(u,t,E,width))*dx - dot(f,v)*dss
    return res

def volume_form(t, width, L):
    return t*width*L*dx

def compliance_form(u, f, dss=ufl.ds):
    return dot(f,u)*dss

'''
    2.2. Create function spaces for the input and the state variables
'''

fea = FEA(mesh)
fea.record = True
# Add input to the PDE problem:
input_name = 'thickness'
input_function_space = FunctionSpace(mesh, ('DG', 0))
input_function = Function(input_function_space)


# Add state to the PDE problem:
# Create Hermite order 3 on a interval (for more informations see:
#    https://defelement.com/elements/examples/interval-Hermite-3.html )
beam_element = basix.ufl_wrapper.create_element(basix.ElementFamily.Hermite,
                                                basix.CellType.interval, 3)
state_name = 'displacements'
state_function_space = FunctionSpace(mesh, beam_element)
state_function = Function(state_function_space)
v = TestFunction(state_function_space)

# Add the same point load at the endpoint as the OpenMDAO example
#     https://github.com/OpenMDAO/OpenMDAO/blob/304d45169b4b2a20e7d0e5441f81d9c072d7af09/openmdao/test_suite/test_examples/beam_optimization/beam_group.py#L29

f = Constant(mesh, -1.)

# Get DOF of the endpoint
DOLFIN_EPS = 3E-16
def Endpoint(x):
    return np.isclose(abs(x[0] - L), DOLFIN_EPS*1e5)

fdim = mesh.topology.dim - 1
endpoint_node = locate_entities_boundary(mesh,fdim,Endpoint)
facet_tag = meshtags(mesh, fdim, endpoint_node,
                    np.full(len(endpoint_node),100,dtype=np.int32))
# Define measures of the endpoint
metadata = {"quadrature_degree":4}
ds_ = ufl.Measure('ds',domain=mesh,subdomain_data=facet_tag,metadata=metadata)


residual_form = pdeRes(state_function,
                        v,
                        input_function,
                        f, ds_(100),
                        E, width)

# Add outputs to the PDE problem:
output_name_1 = 'compliance'
output_form_1 = compliance_form(state_function,f,ds_(100))
output_name_2 = 'volume'
output_form_2 = volume_form(input_function,width,L)

fea.add_input(input_name, input_function)
fea.add_state(name=state_name,
                function=state_function,
                residual_form=residual_form,
                arguments=[input_name])
fea.add_output(name=output_name_1,
                type='scalar',
                form=output_form_1,
                arguments=[state_name])
fea.add_output(name=output_name_2,
                type='scalar',
                form=output_form_2,
                arguments=[input_name])

'''
    2.3. Define the boundary conditions
'''

ubc = Function(state_function_space)
ubc.vector.set(0.0)
startpt = locate_entities_boundary(mesh,0,lambda x : np.isclose(x[0], 0))
locate_BC1 = locate_dofs_topological(state_function_space,0,startpt)
locate_BC_list = [locate_BC1[0], locate_BC1[1]]
fea.add_strong_bc(ubc, locate_BC_list)


'''
3. Set up the FEMO FEA model and run simulation
'''

fea.REPORT = False
fea_model = FEAModel(fea=[fea])


# # Reference optimized thickness distribution from the OpenMDAO example
# #   https://openmdao.org/newdocs/versions/latest/examples/beam_optimization_example.html#implementation-optimization-script

thick_ref = np.array([
    0.14915754,  0.14764328,  0.14611321,  0.14456715,  0.14300421,  0.14142417,
    0.13982611,  0.13820976,  0.13657406,  0.13491866,  0.13324268,  0.13154528,
    0.12982575,  0.12808305,  0.12631658,  0.12452477,  0.12270701,  0.12086183,
    0.11898809,  0.11708424,  0.11514904,  0.11318072,  0.11117762,  0.10913764,
    0.10705891,  0.10493903,  0.10277539,  0.10056526,  0.09830546,  0.09599246,
    0.09362243,  0.09119084,  0.08869265,  0.08612198,  0.08347229,  0.08073573,
    0.07790323,  0.07496382,  0.07190453,  0.06870925,  0.0653583,   0.06182632,
    0.05808044,  0.05407658,  0.04975295,  0.0450185,   0.03972912,  0.03363155,
    0.02620192,  0.01610863])



'''
4. Set up csdl recorder and run the simulation
'''
recorder = csdl.Recorder(inline=True)
recorder.start()

thickness = csdl.Variable(value=h*np.ones(nel), name='thickness')
inputs_group = csdl.VariableGroup()
inputs_group.thickness = thickness
fea_output = fea_model.evaluate(inputs_group)

compliance = fea_output.compliance
volume = fea_output.volume
displacements = fea_output.displacements

'''
    4.1 Run forward evaluation with optimal solution
'''

if run_verify_forward_eval:
    thickness.value = np.array(thick_ref)
    fea_output = fea_model.evaluate(inputs_group)

    # thickness = fea_output.thickness
    print("Forward evaluation with optimal solution:")
    print(" "*4, compliance.names, compliance.value)
    print(" "*4, volume.names, volume.value)
    # print(" "*4, thickness.names, thickness.value)
    print("OpenMDAO compliance: "+str(23762.153677443166))


'''
    4.2 Check the derivatives
'''
if run_check_derivatives:
    from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
    verify_derivatives_inline([displacements,volume,compliance], [thickness], 
                                step_size=1e-6, raise_on_error=False)


'''
    4.3 Set up and run the optimization problem
'''

if run_optimization:
    from modopt import CSDLAlphaProblem
    from modopt import SLSQP
    thickness.set_as_design_variable(upper=10, lower=1E-2)

    volume.set_as_constraint(lower=b*h*L, upper=b*h*L)
    compliance.set_as_objective()
    sim = csdl.experimental.PySimulator(recorder)

    prob = CSDLAlphaProblem(problem_name='beam_thickness', simulator=sim)

    optimizer = SLSQP(prob, ftol=1e-9, maxiter=1000, outputs=['x'])

    # Solve your optimization problem
    optimizer.solve()
    optimizer.print_results()
    print("Optimization results:")
    print(" "*4, compliance.names, compliance.value)
    print(" "*4, volume.names, volume.value)
    # print(" "*4, thickness.names, thickness.value)
    print("OpenMDAO compliance: "+str(23762.153677443166))

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0.0,L,50), thick_ref, "b-o",
                        label="OpenMDAO results")
    ax.plot(np.linspace(0.0,L,nel), thickness.value, "r-o",
                        label="FEMO results")
    ax.set_xlabel("x")
    ax.set_ylabel("optimized thickness distribution")
    ax.legend(loc="best")
    plt.show()
    fig.savefig("beam_thickness_distribution.png", dpi=150)

recorder.stop()




