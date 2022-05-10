
from csdl import Model, CustomExplicitOperation
import csdl
import numpy as np
from csdl_om import Simulator
from fea_dolfinx import *

class OutputModel(Model):

    def initialize(self):
        self.parameters.declare('fea', types=FEA)
        self.parameters.declare('output_name', types=str)
        self.parameters.declare('arg_name_list', types=list)

    def define(self):
        self.fea = self.parameters['fea']
        arg_name_list = self.parameters['arg_name_list']
        output_name = self.parameters['output_name']

        args_dict = dict()
        args_list = []
        for arg_name in arg_name_list:
            if arg_name in self.fea.inputs_dict:
                args_dict[arg_name] = self.fea.inputs_dict[arg_name]
            elif arg_name in self.fea.states_dict:
                args_dict[arg_name] = self.fea.states_dict[arg_name]
            arg = self.declare_variable(arg_name,
                                        shape=(args_dict[arg_name]['shape'],),
                                        val=1.0)
            args_list.append(arg)

        e = OutputOperation(fea=self.fea,
                            args_dict=args_dict,
                            output_name=output_name,
                            )
        output = csdl.custom(*args_list, op=e)
        self.register_output(output_name, output)


class OutputOperation(CustomExplicitOperation):
    """
    input: input/state variables
    output: output
    """
    def initialize(self):
        self.parameters.declare('fea')
        self.parameters.declare('args_dict')
        self.parameters.declare('output_name')

    def define(self):
        self.fea = self.parameters['fea']
        self.output_name = output_name = self.parameters['output_name']
        self.args_dict = args_dict = self.parameters['args_dict']
        for arg_name in args_dict:
            arg = args_dict[arg_name]
            self.add_input(arg_name,
                            shape=(arg['shape'],),)
        self.output = self.fea.outputs_dict[output_name]
        self.output_size = self.output['shape']
        # for field output
        self.output_dim = 1
        # for scalar output
        if self.output_size == 1:
            self.output_dim = 0
        self.add_output(output_name,
                        shape=(self.output_size,))
        self.declare_derivatives('*', '*')

    def compute(self, inputs, outputs):
        for arg_name in inputs:
            arg = self.args_dict[arg_name]
            update(arg['function'], inputs[arg_name])

        outputs[self.output_name] = assemble(self.output['form'],
                                        dim=self.output_dim)

    def compute_derivatives(self, inputs, derivatives):
        for arg_name in inputs:
            arg = self.args_dict[arg_name]
            update(arg['function'], inputs[arg_name])

        for arg_name in self.args_dict:
            derivatives[self.output_name,arg_name] = assemble(
                                        computePartials(
                                            self.output['form'],
                                            self.args_dict[arg_name]['function']),
                                        dim=self.output_dim+1)


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
