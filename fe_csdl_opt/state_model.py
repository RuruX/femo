# Import `dolfin` first to avoid segmentation fault
# from dolfin import *

from csdl import Model, CustomImplicitOperation
import csdl
import numpy as np
from csdl_om import Simulator
from fea_dolfinx import *

class StateModel(Model):

    def initialize(self):
        self.parameters.declare('debug_mode', default=False)
        self.parameters.declare('fea', types=FEA)
        self.parameters.declare('state_name', types=str)
        self.parameters.declare('arg_name_list', types=list)

    def define(self):
        self.fea = self.parameters['fea']
        arg_name_list = self.parameters['arg_name_list']
        state_name = self.parameters['state_name']
        self.debug_mode = self.parameters['debug_mode']

        args_dict = dict()
        args_list = []
        for arg_name in arg_name_list:
            args_dict[arg_name] = self.fea.inputs_dict[arg_name]
            arg = self.declare_variable(arg_name,
                                        shape=(args_dict[arg_name]['shape'],),
                                        val=1.0)
            args_list.append(arg)

        e = StateOperation(fea=self.fea,
                            args_dict=args_dict,
                            state_name=state_name,
                            debug_mode=self.debug_mode)
        state = csdl.custom(*args_list, op=e)
        self.register_output(state_name, state)


class StateOperation(CustomImplicitOperation):
    """
    input: input variable
    output: state
    """
    def initialize(self):
        self.parameters.declare('debug_mode')
        self.parameters.declare('fea')
        self.parameters.declare('args_dict')
        self.parameters.declare('state_name')

    def define(self):
        self.debug_mode = self.parameters['debug_mode']
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running define()...")
            print("="*40)

        self.fea = self.parameters['fea']
        self.state_name = state_name = self.parameters['state_name']
        self.args_dict = args_dict = self.parameters['args_dict']
        for arg_name in args_dict:
            arg = args_dict[arg_name]
            self.add_input(arg_name,
                            shape=(arg['shape'],),)

        self.state = self.fea.states_dict[state_name]
        self.add_output(state_name,
                        shape=(self.state['shape'],),)
        self.declare_derivatives('*', '*')
        self.bcs = self.fea.bc

    def evaluate_residuals(self, inputs, outputs, residuals):
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running evaluate_residuals()...")
            print("="*40)
        for arg_name in inputs:
            arg = self.args_dict[arg_name]
            update(arg['function'], inputs[arg_name])

        update(self.state['function'], outputs[self.state_name])

        residuals[self.state_name] = getFormArray(self.state['residual_form'])


    def solve_residual_equations(self, inputs, outputs):
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running solve_residual_equations()...")
            print("="*40)
        for arg_name in inputs:
            arg = self.args_dict[arg_name]
            update(arg['function'], inputs[arg_name])

        self.fea.solve(self.state['residual_form'],
                        self.state['function'],
                        self.bcs)

        outputs[self.state_name] = getFuncArray(self.state['function'])


    def compute_derivatives(self, inputs, outputs, derivatives):
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running compute_derivatives()...")
            print("="*40)

        for arg_name in inputs:
            update(self.args_dict[arg_name]['function'], inputs[arg_name])
        update(self.state['function'], outputs[self.state_name])

        state = self.state
        args_dict = self.args_dict
        dR_du = computePartials(state['residual_form'],state['function'])
        self.dRdu = assembleMatrix(dR_du)
        dRdf_dict = dict()
        for arg_name in args_dict:
            dRdf = assembleMatrix(computePartials(
                                state['residual_form'],
                                args_dict[arg_name]['function']))
            df = createFunction(args_dict[arg_name]['function'])
            dRdf_dict[arg_name] = dict(dRdf=dRdf, df=df)
        self.dRdf_dict = dRdf_dict
        self.A,_ = assembleSystem(dR_du,
                                state['residual_form'],
                                bcs=self.bcs)

        self.dR = self.state['d_residual']
        self.du = self.state['d_state']

    def compute_jacvec_product(self, inputs, outputs,
                                d_inputs, d_outputs, d_residuals, mode):
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running compute_jacvec_product()...")
            print("="*40)

        ######################
        # Might be redundant #
        for arg_name in inputs:
            update(self.args_dict[arg_name]['function'], inputs[arg_name])
        update(self.state['function'], outputs[self.state_name])
        ######################

        state_name = self.state_name
        args_dict = self.args_dict
        if mode == 'fwd':
            if state_name in d_residuals:
                if state_name in d_outputs:
                    update(self.du, d_outputs[state_name])
                    d_residuals[state_name] += computeMatVecProductFwd(
                            self.dRdu, self.du)
                for arg_name in self.dRdf_dict:
                    if arg_name in d_inputs:
                        update(self.dRdf_dict[arg_name]['df'], d_inputs[arg_name])
                        dRdf = self.dRdf_dict[arg_name]['dRdf']
                        d_residuals[state_name] += computeMatVecProductFwd(
                                dRdf, self.dRdf_dict[arg_name]['df'])

        if mode == 'rev':
            if state_name in d_residuals:
                update(self.dR, d_residuals[state_name])
                if state_name in d_outputs:
                    d_outputs[state_name] += computeMatVecProductBwd(
                            self.dRdu, self.dR)
                for arg_name in self.dRdf_dict:
                    if arg_name in d_inputs:
                        dRdf = self.dRdf_dict[arg_name]['dRdf']
                        d_inputs[arg_name] += computeMatVecProductBwd(
                                dRdf, self.dR)

    def apply_inverse_jacobian(self, d_outputs, d_residuals, mode):
        if self.debug_mode == True:
            print("="*40)
            print("CSDL: Running apply_inverse_jacobian()...")
            print("="*40)
        state_name = self.state_name
        if mode == 'fwd':
            d_outputs[state_name] = self.fea.solveLinearFwd(
                                            self.du, self.A, self.dR, d_residuals[state_name])
        else:
            d_residuals[state_name] = self.fea.solveLinearBwd(
                                            self.dR, self.A, self.du, d_outputs[state_name])



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

    state_name = 'state'
    input_name = 'input'
    model = StateModel(fea=fea,
                        state_name=state_name,
                        arg_name_list=[input_name,],
                        debug_mode=False)
    sim = Simulator(model)
    sim['input'] = getFuncArray(f_ex)
    # print("CSDL: Running the model...")
    sim.run()
    print("="*40)
    control_error = errorNorm(f_ex, input_function)
    print("Error in controls:", control_error)
    state_error = errorNorm(u_ex, state_function)
    print("Error in states:", state_error)
    print("number of controls dofs:", fea.inputs_dict[input_name]['shape'])
    print("number of states dofs:", fea.states_dict[state_name]['shape'])
    print("="*40)
    print("CSDL: Checking the partial derivatives...")
    sim.check_partials(compact_print=True)
