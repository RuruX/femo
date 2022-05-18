from fe_csdl_opt.fea.fea_dolfinx import *
from csdl import Model, CustomExplicitOperation
import csdl
import numpy as np
from csdl_om import Simulator
import scipy.sparse 
from scipy import spatial


class GeneralFilterModel(Model):

    def initialize(self):
        self.parameters.declare('density_function_space')
        self.parameters.declare('num_element_filtered', default=2.)

    def define(self):
        density_function_space = self.parameters['density_function_space']
        num_element_filtered = self.parameters['num_element_filtered']
        NUM_ELEMENTS = density_function_space.dim()

        density_unfiltered = self.declare_variable('density_unfiltered',
                                    shape=(NUM_ELEMENTS,),
                                    val=1.0)

        e = GeneralFilterOperation(density_function_space, num_element_filtered)
        output = csdl.custom(density_unfiltered, op=e)
        self.register_output('density', output)


class GeneralFilterOperation(CustomExplicitOperation):
    """
    input: unfiltered density
    output: filtered density
    """
    def initialize(self):
        self.parameters.declare('density_function_space')
        self.parameters.declare('num_element_filtered', default=2.)

    def define(self):

        density_function_space = self.parameters['density_function_space']
        num_element_filtered = self.parameters['num_element_filtered']
        NUM_ELEMENTS = density_function_space.dim()

        self.input_size = NUM_ELEMENTS
        self.output_size = num_element_filtered
        self.add_input('density_unfiltered',
                        shape=(self.input_size,),
                        val=0.0)
        self.add_output('density',
                        shape=(self.output_size,))
        coords = density_function_space.tabulate_dof_coordinates()
        h_avg = (density_function_space.mesh.hmax() 
                + density_function_space.mesh.hmin())/2
        self.weightMat, row, col = self.compute_weight_mat(coords, h_avg,
                                                            num_element_filtered,
                                                            NUM_ELEMENTS)
        self.declare_derivatives('density', 'density_unfiltered',
                                rows=np.array(row), 
                                cols=np.array(col),
                                val=np.array(self.weightMat))

    def compute(self, inputs, outputs):
        outputs['density'] = self.weightMat.dot(inputs['density_unfiltered'])

    def compute_weight_mat(self, coords, h_avg, nel_filtered, nel):
        radius = nel_filtered * h_avg

        weight_ij = []
        col = []
        row = []

        for i in range(nel):
            current_point = coords[i,:]
            points_selection = coords
            tree = spatial.cKDTree(points_selection)
            idx = tree.query_ball_point(list(current_point), radius)
            nearest_points = points_selection[idx]
            
            weight_sum = sum(radius - np.linalg.norm(current_point - nearest_points,axis = 1))

            for j in idx:
                weight = ( radius - np.linalg.norm(current_point - points_selection[j]))/weight_sum
                row.append(i)
                col.append(j)
                weight_ij.append(weight)       
       
        weight_mtx = scipy.sparse.csr_matrix((weight_ij, (row, col)), shape=(NUM_ELEMENTS, NUM_ELEMENTS))
        return weight_mtx, row, col

if __name__ == "__main__":
    import argparse

    '''
    1. Define the mesh
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--nel',dest='nel',default='4',
                        help='Number of elements')

    args = parser.parse_args()
    num_el = int(args.nel)
    mesh = createUnitSquareMesh(num_el)



    '''
    2. Set up the PDE problem
    '''


    PI = np.pi
    ALPHA = 1E-6

    def interiorResidual(u,v,f):
        mesh = u.function_space.mesh
        n = FacetNormal(mesh)
        h_E = CellDiameter(mesh)
        x = SpatialCoordinate(mesh)
        return inner(grad(u), grad(v))*dx \
                - inner(f, v)*dx

    def boundaryResidual(u,v,u_exact,
                            sym=False,
                            beta_value=0.1,
                            overPenalize=False):

        '''
        Formulation from Github:
        https://github.com/MiroK/fenics-nitsche/blob/master/poisson/
        poisson_circle_dirichlet.py
        '''
        mesh = u.function_space.mesh
        n = FacetNormal(mesh)
        h_E = CellDiameter(mesh)
        x = SpatialCoordinate(mesh)
        beta = Constant(mesh, beta_value)
        sgn = 1.0
        if (sym is not True):
            sgn = -1.0
        retval = sgn*inner(u_exact-u, dot(grad(v), n))*ds \
                - inner(dot(grad(u), n), v)*ds
        penalty = beta*h_E**(-1)*inner(u-u_exact, v)*ds
        if (overPenalize or sym):
            retval += penalty
        return retval

    def pdeRes(u,v,f,u_exact=None,weak_bc=False,sym=False):
        """
        The variational form of the PDE residual for the Poisson's problem
        """
        retval = interiorResidual(u,v,f)
        if (weak_bc):
            retval += boundaryResidual(u,v,u_exact,sym=sym)
        return retval

    def outputForm(u, f, u_exact):
        return 0.5*inner(u-u_exact, u-u_exact)*dx + \
                        ALPHA/2*f**2*dx

    class Expression_f:
        def __init__(self):
            self.alpha = 1e-6

        def eval(self, x):
            return (1/(1+self.alpha*4*np.power(PI,4))*
                    np.sin(PI*x[0])*np.sin(PI*x[1]))

    class Expression_u:
        def __init__(self):
            pass

        def eval(self, x):
            return (1/(2*np.power(PI, 2))*
                    np.sin(PI*x[0])*np.sin(PI*x[1]))


    fea = FEA(mesh, weak_bc=True)
    # Add input to the PDE problem:
    # name = 'input', function = input_function (function is the solution vector here)
    input_function_space = FunctionSpace(mesh, ('DG', 0))
    input_function = Function(input_function_space)

    # Add states to the PDE problem (line 58):
    # name = 'displacements', function = state_function (function is the solution vector here)
    # residual_form = get_residual_form(u, v, rho_e) from atomics.pdes.thermo_mechanical_uniform_temp
    # *inputs = input (can be multiple, here 'input' is the only input)

    state_function_space = FunctionSpace(mesh, ('CG', 1))
    state_function = Function(state_function_space)
    v = TestFunction(state_function_space)
    residual_form = pdeRes(state_function, v, input_function)

    u_ex = fea.add_exact_solution(Expression_u, state_function_space)
    f_ex = fea.add_exact_solution(Expression_f, input_function_space)

    ALPHA = 1e-6

    # Add output-avg_input to the PDE problem:
    output_form = outputForm(state_function, input_function, u_ex)

    fea.add_input('input', input_function)
    fea.add_state(name='state',
                    function=state_function,
                    residual_form=residual_form,
                    arguments=['input',])
    fea.add_output(name='output',
                    type='scalar',
                    form=output_form,
                    arguments=['input','state',])


    '''
    3. Define the boundary conditions
    '''
    ubc = Function(state_function_space)
    ubc.vector.set(0.0)
    locate_BC1 = locate_dofs_geometrical((state_function_space, state_function_space),
                                lambda x: np.isclose(x[0], 0. ,atol=1e-6))
    locate_BC2 = locate_dofs_geometrical((state_function_space, state_function_space),
                                lambda x: np.isclose(x[0], 1. ,atol=1e-6))
    locate_BC3 = locate_dofs_geometrical((state_function_space, state_function_space),
                                lambda x: np.isclose(x[1], 0. ,atol=1e-6))
    locate_BC4 = locate_dofs_geometrical((state_function_space, state_function_space),
                                lambda x: np.isclose(x[1], 1. ,atol=1e-6))
    fea.add_strong_bc(ubc, locate_BC1, state_function_space)
    fea.add_strong_bc(ubc, locate_BC2, state_function_space)
    fea.add_strong_bc(ubc, locate_BC3, state_function_space)
    fea.add_strong_bc(ubc, locate_BC4, state_function_space)

    output_name = 'output'
    model = OutputModel(fea=fea,
                        output_name=output_name,
                        arg_name_list=fea.outputs_dict[output_name]['arguments'],
                        )
    sim = Simulator(model)
    print("CSDL: Running the model...")
    sim.run()
    print("="*40)
    print(sim[output_name])
    print("="*40)
    print("CSDL: Checking the partial derivatives...")
    sim.check_partials(compact_print=True)
