import dolfin as df
import numpy as np
import openmdao.api as om
from atomics.api import PDEProblem, AtomicsGroup
# from atomics.pdes.linear_elastic import get_residual_form
from atomics.general_filter_comp import GeneralFilterComp

def get_residual_form(u, v, rho_e, E = 1, method='SIMP'):
    if method =='SIMP':
        C = rho_e**3
    else:
        C = rho_e/(1 + 8. * (1. - rho_e))
    

    E = 1. * C # C is the design variable, its values is from 0 to 1

    nu = 0.3 # Poisson's ratio

    lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
    mu = E / 2 / (1 + nu) #lame's parameters


    w_ij = 0.5 * (df.grad(u) + df.grad(u).T)
    v_ij = 0.5 * (df.grad(v) + df.grad(v).T)
    
    d = len(u)

    sigm = lambda_*df.div(u)*df.Identity(d) + 2*mu*w_ij

    a = df.inner(sigm, v_ij) * df.dx 
    
    return a

np.random.seed(0)

'''
1. Define the mesh
'''
NUM_ELEMENTS_X = 80
NUM_ELEMENTS_Y = 40
LENGTH_X = 160.
LENGTH_Y = 80.

mesh = df.RectangleMesh.create(
    [df.Point(0.0, 0.0), df.Point(LENGTH_X, LENGTH_Y)],
    [NUM_ELEMENTS_X, NUM_ELEMENTS_Y],
    df.CellType.Type.quadrilateral,
)

'''
2. Define the traction boundary conditions
'''
# here traction force is applied on the middle of the right edge
class TractionBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return ((abs(x[1] - LENGTH_Y/2) < LENGTH_Y/NUM_ELEMENTS_Y + df.DOLFIN_EPS) and (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS*1.5e15))

# Define the traction boundary
sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
upper_edge = TractionBoundary()
upper_edge.mark(sub_domains, 6)
dss = df.Measure('ds')(subdomain_data=sub_domains)
f = df.Constant((0, -1. / 4 ))

'''
3. Setup the PDE problem
'''
# PDE problem
pde_problem = PDEProblem(mesh)

# Add input to the PDE problem:
# name = 'density', function = density_function (function is the solution vector here)
density_function_space = df.FunctionSpace(mesh, 'DG', 0)
density_function = df.Function(density_function_space)
pde_problem.add_input('density', density_function)

# Add states to the PDE problem (line 58):
# name = 'displacements', function = displacements_function (function is the solution vector here)
# residual_form = get_residual_form(u, v, rho_e) from atomics.pdes.thermo_mechanical_uniform_temp
# *inputs = density (can be multiple, here 'density' is the only input)

displacements_function_space = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
displacements_function = df.Function(displacements_function_space)
v = df.TestFunction(displacements_function_space)
method='SIMP'
residual_form = get_residual_form(
    displacements_function, 
    v, 
    density_function,
    method=method
)

residual_form -= df.dot(f, v) * dss(6)
pde_problem.add_state('displacements', displacements_function, residual_form, 'density')

# Add output-avg_density to the PDE problem:
volume = df.assemble(df.Constant(1.) * df.dx(domain=mesh))
avg_density_form = density_function / (df.Constant(1. * volume)) * df.dx(domain=mesh)
pde_problem.add_scalar_output('avg_density', avg_density_form, 'density')

# Add output-compliance to the PDE problem:
compliance_form = df.dot(f, displacements_function) * dss(6)
pde_problem.add_scalar_output('compliance', compliance_form, 'displacements')

# Add Dirichlet boundary conditions to the PDE problem:
pde_problem.add_bc(df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-0.) < DOLFIN_EPS)'))

'''
4. Setup the optimization problem
'''
# Define the OpenMDAO problem and model

prob = om.Problem()

num_dof_density = pde_problem.inputs_dict['density']['function'].function_space().dim()

comp = om.IndepVarComp()
comp.add_output(
    'density_unfiltered', 
    shape=num_dof_density, 
    val=np.random.random(num_dof_density) * 0.86,
)
prob.model.add_subsystem('indep_var_comp', comp, promotes=['*'])

comp = GeneralFilterComp(density_function_space=density_function_space)
prob.model.add_subsystem('general_filter_comp', comp, promotes=['*'])


group = AtomicsGroup(pde_problem=pde_problem)
prob.model.add_subsystem('atomics_group', group, promotes=['*'])

prob.model.add_design_var('density_unfiltered',upper=1, lower=1e-4)
prob.model.add_objective('compliance')
prob.model.add_constraint('avg_density',upper=0.40)

# set up the optimizer
prob.driver = driver = om.pyOptSparseDriver()
driver.options['optimizer'] = 'SNOPT'
driver.opt_settings['Verify level'] = 0

driver.opt_settings['Major iterations limit'] = 100000
driver.opt_settings['Minor iterations limit'] = 100000
driver.opt_settings['Iterations limit'] = 100000000
driver.opt_settings['Major step limit'] = 2.0

driver.opt_settings['Major feasibility tolerance'] = 1.0e-6
driver.opt_settings['Major optimality tolerance'] =1.e-8

prob.setup()

if False:
    prob.run_model()
    prob.check_partials(compact_print=True)
else:
    prob.run_driver()


#save the solution vector
if method =='SIMP':
    penalized_density  = df.project(density_function**3, density_function_space) 
else:
    penalized_density  = df.project(density_function/(1 + 8. * (1. - density_function)), density_function_space) 

df.File('solutions/case_1/cantilever_beam/displacement.pvd') << displacements_function
df.File('solutions/case_1/cantilever_beam/penalized_density.pvd') << penalized_density