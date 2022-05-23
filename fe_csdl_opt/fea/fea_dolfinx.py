"""
The FEniCS wrapper for variational forms and partial derivatives computation
"""

from fe_csdl_opt.fea.utils_dolfinx import *
from dolfinx.io import XDMFFile
import ufl

from dolfinx.fem.petsc import apply_lifting
from dolfinx.fem import (set_bc, Function, FunctionSpace, dirichletbc,
                        locate_dofs_topological, locate_dofs_geometrical,
                        Constant, VectorFunctionSpace)
from dolfinx.mesh import compute_boundary_facets
from ufl import (grad, SpatialCoordinate, CellDiameter, FacetNormal,
                    div, Identity)
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix



class AbstractFEA(object):
    """
    The abstract class of the FEniCS wrapper for defining the variational forms
    for PDE residuals and outputs, computing derivatives, and solving
    the problems.
    """
    def __init__(self, **args):

        self.mesh = None
        self.weak_bc = False
        self.sym_nitsche = False
        self.initFunctionSpace(self.mesh)
        self.res = None

    def __init__(self, mesh):
        self.mesh = mesh

        self.inputs_dict = dict()
        self.states_dict = dict()
        self.outputs_dict = dict()
        self.bcs_list = list()

    def create_function_space(self, var_name, type='CG1'):
        pass

    def add_bc(self, bc):
        self.bcs_list.append(bc)

    def add_input(self, name, function):
        if name in self.inputs_dict:
            raise ValueError('name has already been used for an input')

        function.rename(name, name)
        self.inputs_dict[name] = dict(
            function=function,
        )

    def add_state(self, name, function, residual_form, *arguments):
        function.rename(name, name)
        self.states_dict[name] = dict(
            function=function,
            residual_form=residual_form,
            arguments=arguments,
        )

    def add_output(self, name, form, *arguments):
        self.outputs_dict[name] = dict(
            form=form,
            arguments=arguments,
        )


class FEA(object):
    """
    The class of the FEniCS wrapper for the motor problem,
    with methods to compute the variational forms, partial derivatives,
    and solve the nonlinear/linear subproblems.
    """
    def __init__(self, mesh, weak_bc=False, sym_nitsche=False):

        self.mesh = mesh

        self.weak_bc = weak_bc
        self.sym_nitsche = sym_nitsche

        self.inputs_dict = dict()
        self.states_dict = dict()
        self.outputs_dict = dict()
        self.bc = []


    def add_input(self, name, function):
        if name in self.inputs_dict:
            raise ValueError('name has already been used for an input')
        self.inputs_dict[name] = dict(
            function=function,
            function_space=function.function_space,
            shape=len(getFuncArray(function)),
        )

    def add_state(self, name, function, residual_form, arguments):
        self.states_dict[name] = dict(
            function=function,
            residual_form=residual_form,
            function_space=function.function_space,
            shape=len(getFuncArray(function)),
            d_residual=Function(function.function_space),
            d_state=Function(function.function_space),
            arguments=arguments,
        )

    def add_output(self, name, type, form, arguments):
        if type == 'field':
            shape = len(getFormArray(form))
        elif type == 'scalar':
            shape = 1
        partials = []
        for argument in arguments:
            if argument in self.inputs_dict:
                partial = derivative(form, self.inputs_dict[argument]['function'])
            elif argument in self.states_dict:
                partial = derivative(form, self.states_dict[argument]['function'])
            partials.append(partial)
        self.outputs_dict[name] = dict(
            form=form,
            shape=shape,
            arguments=arguments,
            partials=partials,
        )

    def add_exact_solution(self, Expression, function_space):
        f_analytic = Expression()
        f_ex = Function(function_space)
        f_ex.interpolate(f_analytic.eval)
        return f_ex

    def add_strong_bc(self, ubc, locate_BC_list, function_space):
        for locate_BC in locate_BC_list:
            self.bc.append(dirichletbc(ubc, locate_BC, function_space))

    def solve(self, res, func, bc, report=False):
        """
        Solve the PDE problem
        """
        if report == True:
            print(80*"=")
            print(" FEA: Solving the PDE problem")
            print(80*"=")
        from timeit import default_timer
        start = default_timer()
#        solveNonlinear(res, func, bc, report=report)
        solveNonlinear(res, func, bc,
                     abs_tol=1e-16, max_it=100, report=report)
        stop = default_timer()
        if report == True:
            print("Solve nonlinear finished in ",start-stop, "seconds")


    def solveLinearFwd(self, du, A, dR, dR_array):
        """
        solve linear system dR = dR_du (A) * du in DOLFIN type
        """
        setFuncArray(dR, dR_array)

        du.vector.set(0.0)

        solveKSP(A, dR.vector, du.vector)
        du.vector.assemble()
        du.vector.ghostUpdate()
        return du.vector.getArray()

    def solveLinearBwd(self, dR, A, du, du_array):
        """
        solve linear system du = dR_du.T (A_T) * dR in DOLFIN type
        """
        setFuncArray(du, du_array)

        dR.vector.set(0.0)
        solveKSP(transpose(A), du.vector, dR.vector)
        dR.vector.assemble()
        dR.vector.ghostUpdate()
        return dR.vector.getArray()
