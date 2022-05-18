
from curses import resize_term
from fe_csdl_opt.fea.fea_dolfinx import *
from fe_csdl_opt.csdl_opt.fea_model import FEAModel
from fe_csdl_opt.csdl_opt.state_model import StateModel
from fe_csdl_opt.csdl_opt.output_model import OutputModel
# from fe_csdl_opt.csdl_opt.post_processor.general_filter_model \
#                                     import GeneralFilterModel
import numpy as np
import csdl
from csdl import Model
from csdl_om import Simulator
from matplotlib import pyplot as plt
import argparse

'''
1. Define the mesh
'''

parser = argparse.ArgumentParser()
parser.add_argument('--nelx',dest='nelx',default='80',
                    help='Number of elements in x direction')
parser.add_argument('--nely',dest='nely',default='40',
                    help='Number of elements in x direction')

args = parser.parse_args()
num_el_x = int(args.nelx)
num_el_y = int(args.nely)

LENGTH_X = 160.
LENGTH_Y = 80.
mesh = createRectangleMesh(np.array([0.0,0.0]),
                            np.array([LENGTH_X, LENGTH_Y]),
                            num_el_x,
                            num_el_y)



'''
2. Set up the PDE problem
'''

'''
2.1 Define the traction boundary for the source term
'''
######################### dolfinx ###################
#### Getting facets of the bottom edge that will come in contact ####
DOLFIN_EPS = 1E-8
def TractionBoundary(x):
    return np.logical_and(abs(x[1] - LENGTH_Y/2) < LENGTH_Y/num_el_y + DOLFIN_EPS,
                            abs(x[0] - LENGTH_X ) < DOLFIN_EPS*1.5e15)

fdim = mesh.topology.dim - 1 
bottom_facets = locate_entities_boundary(mesh,fdim,TractionBoundary)

marked_facets = np.hstack([bottom_facets])
marked_values = np.hstack([np.full(len(bottom_facets),3,dtype=np.int32)])
sorted_facets = np.argsort(marked_facets)
facet_tag = meshtags(mesh,fdim,marked_facets[sorted_facets],marked_values[sorted_facets])

#### Defining measures ####
metadata = {"quadrature_degree":4}
import ufl
ds_ = ufl.Measure('ds',domain=mesh,subdomain_data=facet_tag,metadata=metadata)



######################### dolfin ####################
# here traction force is applied on the middle of the right edge
# class TractionBoundary(df.SubDomain):
#     def inside(self, x, on_boundary):
#         return ((abs(x[1] - LENGTH_Y/2) < LENGTH_Y/NUM_ELEMENTS_Y + df.DOLFIN_EPS) and (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS*1.5e15))

# # Define the traction boundary
# sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
# upper_edge = TractionBoundary()
# upper_edge.mark(sub_domains, 6)
# dss = df.Measure('ds')(subdomain_data=sub_domains)
# f = df.Constant((0, -1. / 4 ))

def pdeRes(u, v, rho_e, f, E = 1, ds_ = ds, method='SIMP'):
    if method =='SIMP':
        C = rho_e**3
    else:
        C = rho_e/(1 + 8. * (1. - rho_e))
    E = 1. * C # C is the design variable, its values is from 0 to 1
    nu = 0.3 # Poisson's ratio
    lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
    mu = E / 2 / (1 + nu) #lame's parameters

    w_ij = 0.5 * (grad(u) + grad(u).T)
    v_ij = 0.5 * (grad(v) + grad(v).T)
    d = len(u)
    sigm = lambda_*div(u)*Identity(d) + 2*mu*w_ij
    res = inner(sigm, v_ij) * dx - dot(f, v) * ds_
    return res

def averageFunc(func):
    volume = assemble(Constant(mesh,1.0)*dx)
    func1 = Function(func.function_space)
    func1.vector.set(1/volume)
    return inner(func,func1)*dx

def compliance(u, f, ds_=ds):
    return dot(u,f)*ds_
###################################################

fea = FEA(mesh)
# Add input to the PDE problem:
input_name = 'density'
input_function_space = FunctionSpace(mesh, ('DG', 0))
input_function = Function(input_function_space)

# Add state to the PDE problem:
state_name = 'displacements'
state_function_space = VectorFunctionSpace(mesh, ('CG', 1))
state_function = Function(state_function_space)
v = TestFunction(state_function_space)
method = 'SIMP'
f = Constant(mesh, (0,-1/4))
residual_form = pdeRes(state_function, 
                        v, 
                        input_function,
                        f,
                        ds_=ds(6),
                        method=method)
# Add output to the PDE problem:
output_name_1 = 'avg_density'
output_form_1 = averageFunc(input_function)
output_name_2 = 'compliance'
output_form_2 = compliance(state_function, 
                            f, 
                            ds_=ds_(6))


'''
3. Define the boundary conditions
'''

############ Strongly enforced boundary conditions #############
ubc = Function(state_function_space)
ubc.vector.set(0.0)
locate_BC1 = locate_dofs_geometrical((state_function_space, state_function_space),
                            lambda x: np.isclose(x[0], 0. ,atol=1e-6))
locate_BC_list = [locate_BC1]
fea.add_strong_bc(ubc, locate_BC_list, state_function_space)

############ Weakly enforced boundary conditions #############
############### Unsymmetric Nitsche's method #################
# residual_form = pdeRes(state_function, v, input_function, 
#                         u_exact=u_ex, weak_bc=True, sym=False)
##############################################################



fea.add_input(input_name, input_function)
fea.add_state(name=state_name,
                function=state_function,
                residual_form=residual_form,
                arguments=[input_name])
fea.add_output(name=output_name_1,
                type='scalar',
                form=output_form_1,
                arguments=[input_name])
fea.add_output(name=output_name_2,
                type='scalar',
                form=output_form_2,
                arguments=[state_name])



'''
4. Set up the CSDL model
'''

fea_model = FEAModel(fea=fea)


# post_processor_name = 'general_filter_comp'
# post_processor_model = GeneralFilterModel(density_function_space=state_function_space)
# fea_model.add_subsystem(post_processor_name, post_processor_model, promotes=['*'])

# fea_model.add_design_variable('density_unfiltered', upper=1.0, lower=1e-4)
fea_model.add_design_variable('density', upper=1.0, lower=1e-4)
fea_model.add_objective(output_name_1) # compliance
fea_model.add_constraint(output_name_2, upper=0.40) # ave_density
sim = Simulator(fea_model)

########### Test the forward solve ##############

sim.run()

############# Check the derivatives #############
sim.check_partials(compact_print=True)
sim.prob.check_totals(compact_print=True)  

'''
5. Set up the optimization problem
'''
# ############## Run the optimization with pyOptSparse #############
# import openmdao.api as om
# ####### Driver = SNOPT #########
# driver = om.pyOptSparseDriver()
# driver.options['optimizer']='SNOPT'
# driver.opt_settings['Verify level'] = 0

# driver.opt_settings['Major iterations limit'] = 100000
# driver.opt_settings['Minor iterations limit'] = 100000
# driver.opt_settings['Iterations limit'] = 100000000
# driver.opt_settings['Major step limit'] = 2.0

# driver.opt_settings['Major feasibility tolerance'] = 1e-6
# driver.opt_settings['Major optimality tolerance'] = 1e-8
# driver.options['print_results'] = False

# sim.prob.driver = driver
# sim.prob.setup()
# sim.prob.run_driver()


print("Compliance value: ", sim[output_name_1])
print("Constraint value: ", sim[output_name_2])

with XDMFFile(MPI.COMM_WORLD, "solutions/"+state_name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(fea.mesh)
    xdmf.write_function(fea.states_dict[state_name]['function'])
with XDMFFile(MPI.COMM_WORLD, "solutions/"+input_name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(fea.mesh)
    xdmf.write_function(fea.inputs_dict[input_name]['function'])
    
    
    


