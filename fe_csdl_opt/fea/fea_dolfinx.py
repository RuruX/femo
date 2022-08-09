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
from ufl import (grad, SpatialCoordinate, CellDiameter, FacetNormal,
                    div, Identity)
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

import os.path


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


    def add_strong_bc(self, bc):
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


        self.inputs_dict = dict()
        self.states_dict = dict()
        self.outputs_dict = dict()
        self.bc = []

        self.PDE_SOLVER = "Newton"
        self.SOLVE_INCREMENTAL = False
        self.REPORT = True

        self.weak_bc = weak_bc
        self.sym_nitsche = sym_nitsche

        self.ubc = None
        self.solver = None

        self.opt_iter = 0
        self.record = False

    def add_input(self, name, function):
        if name in self.inputs_dict:
            raise ValueError('name has already been used for an input')
        self.inputs_dict[name] = dict(
            function=function,
            function_space=function.function_space,
            shape=len(getFuncArray(function)),
            recorder=self.createRecorder(name, self.record)
        )

    def add_state(self, name, function, residual_form, arguments, dR_du=None, dR_df_list=None):
        self.states_dict[name] = dict(
            function=function,
            residual_form=residual_form,
            function_space=function.function_space,
            shape=len(getFuncArray(function)),
            d_residual=Function(function.function_space),
            d_state=Function(function.function_space),
            dR_du=dR_du,
            dR_df_list=dR_df_list,
            arguments=arguments,
            recorder=self.createRecorder(name, self.record)
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

    def add_strong_bc(self, ubc, locate_BC_list, 
                    function_space=None):
        if function_space == None:
            for locate_BC in locate_BC_list:
                self.bc.append(dirichletbc(ubc, locate_BC))
        else:
            for locate_BC in locate_BC_list:
                self.bc.append(dirichletbc(ubc, locate_BC, function_space))

    def solve(self, res, func, bc):
        """
        Solve the PDE problem
        """
        solver_type=self.PDE_SOLVER
        incremental=self.SOLVE_INCREMENTAL
        report=self.REPORT
        if incremental is not True:
            solveNonlinear(res,func,bc,solver_type,report)
        else:
            # self.solveNonlinear(res,func,bc,solver,report)
            if (incremental is True and solver_type=='SNES'):

                func_old = self.ubc

                # func_bc is the final step solution of the bc values
                func_bc = Function(func.function_space)
                func_bc.vector[:] = func_old.vector

                # Get the relative movements from the previous step
                relative_edge_deltas = func_bc.vector[:] - func.vector[:]
                STEPS, increment_deltas = getDisplacementSteps(func_bc, 
                                                            relative_edge_deltas,
                                                            self.mesh)
                print("Nonzero edge movements:",increment_deltas[np.nonzero(increment_deltas)])
                # newton_solver = NewtonSolver(res, func, bc, rel_tol=1e-6, report=report)

                snes_solver = SNESSolver(res, func, bc, rel_tol=1e-6, report=report)
                # Incrementally set the BCs to increase to `edge_deltas`
                print(80*"=")
                print(' FEA: total steps for mesh motion:', STEPS)
                print(80*"=")
                for i in range(STEPS):
                    if report == True:
                        print(80*"=")
                        print("  FEA: Step "+str(i+1)+" of mesh movement")
                        print(80*"=")
                    func_old.vector[:] = func.vector

                    # func_old.vector[:] += increment_deltas
                    func_old.vector[np.nonzero(relative_edge_deltas)] = (i+1)*increment_deltas[np.nonzero(relative_edge_deltas)]
                    print(func_old.x.array[np.nonzero(relative_edge_deltas)])
                    print(func.x.array[np.nonzero(relative_edge_deltas)])
                    print(assemble_vector(form(res)))
                    # newton_solver.solve(func)
                    snes_solver.solve(None, func.vector)

                if report == True:
                    print(80*"=")
                    print(' FEA: L2 error of the mesh motion on the edges:',
                                np.linalg.norm(func.vector[np.nonzero(relative_edge_deltas)]
                                         - relative_edge_deltas[np.nonzero(relative_edge_deltas)]))
                    print(80*"=")



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

    def createRecorder(self, name, record=True):
        recorder = None
        if record:
            recorder = XDMFFile(MPI.COMM_WORLD, "records/record_"+name+".xdmf", "w")
            recorder.write_mesh(self.mesh)
        return recorder
